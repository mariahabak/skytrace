"""Controlled image degradations for the robustness tests and the demo.

Authors: Maria Habak, [Partner name]

Each degradation takes uint8 clips (N, T, S, S) and a severity, and
returns new uint8 clips of the same shape. Severity 0 changes nothing.
They are deterministic, so every model sees exactly the same inputs.
"""

import cv2
import numpy as np
import torch


def _per_frame(clips: torch.Tensor, function) -> torch.Tensor:
    array = clips.numpy()
    out = np.empty_like(array)
    for n in range(array.shape[0]):
        for t in range(array.shape[1]):
            out[n, t] = function(array[n, t])
    return torch.from_numpy(out)


def blur(clips: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur with standard deviation ``sigma`` pixels."""
    if sigma <= 0:
        return clips.clone()
    return _per_frame(clips, lambda f: cv2.GaussianBlur(f, (0, 0), sigma))


def reduce_resolution(clips: torch.Tensor, factor: float) -> torch.Tensor:
    """Shrink by ``factor`` and enlarge back, as if the object were farther."""
    if factor <= 1:
        return clips.clone()

    def shrink(frame):
        size = frame.shape[0]
        small = max(2, int(round(size / factor)))
        tiny = cv2.resize(frame, (small, small), interpolation=cv2.INTER_AREA)
        return cv2.resize(tiny, (size, size), interpolation=cv2.INTER_LINEAR)

    return _per_frame(clips, shrink)


def reduce_contrast(clips: torch.Tensor, amount: float) -> torch.Tensor:
    """Pull every pixel towards the frame mean; amount in [0, 1)."""
    if amount <= 0:
        return clips.clone()
    x = clips.float()
    mean = x.mean(dim=(-2, -1), keepdim=True)
    y = mean + (1.0 - amount) * (x - mean)
    return y.round().clamp(0, 255).to(torch.uint8)


DEGRADATIONS = {
    "blur": blur,
    "resolution": reduce_resolution,
    "contrast": reduce_contrast,
}


def apply(clips: torch.Tensor, kind: str, severity: float) -> torch.Tensor:
    if kind not in DEGRADATIONS:
        raise ValueError(f"Unknown degradation '{kind}'")
    return DEGRADATIONS[kind](clips, severity)
