# Drone or Bird? A size-aware classifier for small flying targets

*Author: Mariah Habak (273074) — Programming for AI and ML, ESIB–USJ, 2026.*
*AI use: code drafted with Claude (Anthropic) with the instructor's permission.*

**In five lines.** Drones and birds in surveillance footage are often tiny, but classifiers are usually trained and tested on close-ups. We classify object crops as bird or drone and ask how far the object can shrink before the model fails. Our twist is a `DistanceAugment` (downscale, blur, noise) used during training. We measure accuracy and F1 at full size, 48, 32, 24 and 16 px, on synthetic small targets made with a *different* recipe and on objects that are naturally small, against two baselines, over 3 seeds.

Research prototype on public data (perception only). Scores are model scores, not guaranteed probabilities.

## Install and test

```bash
python -m venv venv && venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
pytest -q                                           # 13 tests, no dataset needed
python -m dronebird.pipeline --smoke                # whole pipeline on a tiny fake dataset, ~1 min on CPU
```

## Run on the real data

Dataset: download from Mendeley (DOI 10.17632/6ghdz52pd7) and unzip into `data/raw/bird_vs_drone`. Training is meant for a free Colab GPU: see `notebooks/train_on_colab.ipynb`.

| Step | Command |
|---|---|
| 1. Look at the data, check `class_ids` | `python -m dronebird.audit data/raw/bird_vs_drone` |
| 2. Prepare: boxes, near-duplicates, split, crops, figures | `python -m dronebird.prepare configs/data.json` |
| 3. Check 200 labels by hand | `python -m dronebird.labelcheck sheet`, fill the CSV, then `... score` |
| 4. Train one model, 3 seeds, validation curves | `python -m dronebird.run configs/<model>.json --stage fit` |
| 5. Size-curve tables (mean ± std) | `python -m dronebird.tables --split val` |
| 6. Wrong predictions with Grad-CAM | `python -m dronebird.errors results/resnet18_distance/seed0 --test-set real_small` |
| 7. Test, **once**, at the end | `python -m dronebird.run configs/<model>.json --stage test --final` |
| 8. Demo web page | `python -m dronebird.demo results/resnet18/seed0 results/resnet18_distance/seed0` |
| All of 2, 4, 5 (and 7 with `--final`) | `python -m dronebird.pipeline` |

Models (`configs/`): `majority` (baseline 1), `small_cnn` (baseline 2, from scratch), `resnet18` (pretrained, no DistanceAugment) and `resnet18_distance`. Both ResNet-18 versions use the same settings and a random flip; the only difference is DistanceAugment.

## Method details

**Crop pipeline (the same for every model, the demo included).** Each box gives an object size L = longest box side, in pixels of the original frame. We take a square window around the box centre with side ⌈1.5 × L⌉, so the aspect ratio is never changed. Where the window leaves the frame, the border pixels are repeated. The window is cut with tensor slicing `padded[:, top:top+side, left:left+side]` → `(3, side, side)`, then resized to `(3, 96, 96)` with bilinear interpolation (antialiased when shrinking). The object then spans 64 canvas pixels: this is "full size". Crops are stored as `uint8 (N, 3, 96, 96)`. The networks get `(N, 3, 96, 96)` floats normalised with the ImageNet mean and std by broadcasting (`(3, 1, 1)` against `(N, 3, 96, 96)`).

**Two different recipes.** Shrinking an object to s px means shrinking the canvas to round(1.5 × s) px, then going back to 96 px with the shared bilinear step.

| | Training: `DistanceAugment` (p = 0.5) | Test: synthetic small targets |
|---|---|---|
| Shrink | `F.interpolate(mode="area")`, s ~ U[12, 64] | bicubic, antialiased (torchvision `resize`) |
| Blur | Gaussian, σ ~ U[0, 1] px, depthwise `conv2d` | none |
| Noise | Gaussian, std ~ U[0, 0.04], `randn_like` | JPEG compression, quality 50 |
| Back to 96 px | bilinear (shared) | bilinear (shared) |

**Test sets.** *Synthetic*: test objects that are large in the frame (L ≥ 64 px), shown at full size and shrunk to 48/32/24/16 px with the test recipe. *Real small*: test objects that are naturally small (L < 48 px), unchanged, in bins 36–48, 28–36, 20–28 and < 20 px. F1 of a bin that contains only one class is not meaningful (at most 50%).

**Near-duplicates.** Every frame is shrunk to a 16 × 16 colour thumbnail. Pairs with an RMS difference ≤ 0.03 (found with `torch.cdist`), and frames whose file names share a source id, are joined with union-find. Each group goes into one partition (70/15/15, seed 2026), and Python sets prove that no image, group or source appears in two partitions. `data/processed/summary.json` reports how many were found.

**Grad-CAM** is our own code (`dronebird/gradcam.py`): channel-mean of the gradients, weighted sum of the activations, ReLU, bilinear upscaling. We use ResNet-18 `layer3` (6 × 6) and the last SmallCNN block (12 × 12).

## Repository

```
configs/            data.json + one JSON per model (configs/smoke/: tiny fake run)
dronebird/
  data/             yolo.py (boxes dict) · duplicates.py · split.py · crops.py · datasets.py (FlyingObjectCrops) · images.py
  models/           base.py (TargetClassifier) · majority.py · networks.py (nn.Modules) · network_classifier.py
  augment.py        Augmentation, DistanceAugment, BicubicShrink, JpegCompression, ShotNoise, CanvasResize
  training.py       Trainer      evaluation.py  Evaluator (size curve)      metrics.py  (tensor metrics)
  gradcam.py · prepare.py · run.py · tables.py · errors.py · labelcheck.py · audit.py · demo.py · pipeline.py · fake.py
tests/              pytest tests on synthetic data
notebooks/          train_on_colab.ipynb
```

## Data and licences

See [DATASHEET.md](DATASHEET.md). Dataset: Srivastav, Shandilya et al. (2023), CC BY 4.0. Libraries: PyTorch and torchvision (ResNet-18 ImageNet weights and image input/output, approved by the instructor), NumPy, matplotlib, pytest.
