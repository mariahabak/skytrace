"""Local demo web page: upload a crop, shrink it, compare two models.

Author: Mariah Habak

Usage:
  python -m dronebird.demo results/resnet18/seed0 results/resnet18_distance/seed0
  then open http://localhost:8000

The page sends the uploaded image to this small server (Python's own
http.server, no extra library). The server loads each model's
checkpoint (saved with torch.save during training), cuts a square crop
from the middle of the image, shrinks it with the same TEST recipe as
the synthetic test set, and returns each model's answer with a
Grad-CAM heat map. Research prototype: the scores are model scores, not
guaranteed probabilities, and this is not a warning system.
"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402
import torch  # noqa: E402
from torchvision.io import ImageReadMode, decode_image, encode_png  # noqa: E402

from dronebird.augment import synthetic_test_degradation  # noqa: E402
from dronebird.config import Config  # noqa: E402
from dronebird.data.images import to_canvas  # noqa: E402
from dronebird.gradcam import GradCAM  # noqa: E402
from dronebird.models import build_model  # noqa: E402

NAMES = ("bird", "drone")
PAGE = Path(__file__).with_name("demo_page.html")


def png_data_url(image: torch.Tensor) -> str:
    """Float (3, H, W) in [0, 1] -> 'data:image/png;base64,...'."""
    data = encode_png((image.clamp(0, 1) * 255).round().to(torch.uint8))
    return "data:image/png;base64," + base64.b64encode(data.numpy().tobytes()).decode()


class DemoModel:
    """One trained model loaded from its run folder."""

    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        config = Config.from_file(run_dir / "config.json")
        data = Config.from_file(config.get("data_config", "configs/data.json")).section("data")
        self.__crops = data["crops"]
        self.__evaluation = data["evaluation"]
        seed = int(run_dir.name.replace("seed", ""))
        self.__name = config.name
        self.__model = build_model(config.section("model"), seed, self.__crops["margin"],
                                   self.__crops["canvas"])
        self.__model.load(run_dir)  # torch.load of the checkpoint saved with torch.save
        net = self.__model.network
        self.__cam = GradCAM(net, net.cam_layer())

    @property
    def name(self) -> str:
        return self.__name

    @property
    def crops(self) -> dict:
        return dict(self.__crops)

    @property
    def evaluation(self) -> dict:
        return dict(self.__evaluation)

    def answer(self, image: torch.Tensor) -> dict:
        heat, cls, scores = self.__cam(image)
        overlay = torch.from_numpy(colormaps["jet"](heat.numpy())[..., :3]).permute(2, 0, 1)
        blended = 0.55 * image + 0.45 * overlay.float()
        return {"model": self.__name, "label": NAMES[cls],
                "scores": {NAMES[i]: round(float(scores[i]), 3) for i in range(2)},
                "heatmap": png_data_url(blended)}


class DemoService:
    """Turns an uploaded image and a size into model answers."""

    def __init__(self, models: list):
        self.__models = models
        self.__crops = models[0].crops
        self.__evaluation = models[0].evaluation

    @property
    def sizes(self) -> list:
        return ["full"] + [str(s) for s in self.__evaluation["synthetic_sizes"]]

    def square_crop(self, raw: bytes) -> torch.Tensor:
        image = decode_image(torch.frombuffer(bytearray(raw), dtype=torch.uint8),
                             mode=ImageReadMode.RGB)
        _, h, w = image.shape
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        crop = image[:, top:top + side, left:left + side].float() / 255.0
        return to_canvas(crop, self.__crops["canvas"]).clamp(0, 1)

    def predict(self, raw: bytes, size: str) -> dict:
        image = self.square_crop(raw)
        if size != "full":
            recipe = synthetic_test_degradation(int(size), self.__evaluation["test_degradation"],
                                                self.__crops["margin"], self.__crops["canvas"])
            image = recipe(image.unsqueeze(0))[0]
        return {"input": png_data_url(image), "size": size,
                "answers": [m.answer(image) for m in self.__models]}


def make_handler(service: DemoService):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = PAGE.read_text(encoding="utf-8").replace(
                "__SIZES__", json.dumps(service.sizes)).encode()
            self.__send(200, "text/html; charset=utf-8", body)

        def do_POST(self):
            if self.path != "/predict":
                return self.__send(404, "text/plain", b"not found")
            length = int(self.headers.get("Content-Length", 0))
            if length > 15_000_000:
                return self.__send(413, "text/plain", b"image too large")
            request = json.loads(self.rfile.read(length))
            raw = base64.b64decode(request["image"].split(",", 1)[-1])
            try:
                result = service.predict(raw, str(request.get("size", "full")))
            except RuntimeError as error:
                return self.__send(400, "application/json",
                                   json.dumps({"error": str(error)}).encode())
            self.__send(200, "application/json", json.dumps(result).encode())

        def __send(self, code, kind, body):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="one or two results/<model>/seed<k> folders")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    service = DemoService([DemoModel(d) for d in args.run_dirs])
    print(f"Open http://localhost:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service)).serve_forever()


if __name__ == "__main__":
    main()
