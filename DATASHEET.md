# Data sheet

*Authors: Maria Habak, [Partner name]*

1. **Source:** Halmstad multi-sensor drone detection dataset — Svanström, Alonso-Fernandez and Englund (2021), *A dataset for multi-sensor drone detection*, Data in Brief 39, 107521. https://github.com/DroneDetectionThesis/Drone-detection-dataset. We use the infrared (IR, 320×256, 30 fps) videos of the BIRD and DRONE classes only.
2. **Licence / consent:** CC0-1.0 (public domain dedication), stated in the repository's LICENSE file. No people appear in the clips; no personal data.
3. **Size:** 228 IR videos kept (76 bird, 152 drone) after dropping 8 videos with fewer than 40 labelled frames. They give 2,713 clips of 8 frames: train 2,182 (562 bird / 1,620 drone), validation 288 (115 / 173), test 243 (119 / 124).
4. **Labelling:** class labels and per-frame boxes come from the dataset authors (MATLAB Video Labeler), read with `mcos-decoder`. A clip is 8 frames taken every 3rd frame (0.8 s); each frame is a square crop around the box (2× its longest side), resized to 64×64. Sessions were rebuilt from file creation times (same class, < 30 min apart) and each session is in exactly one partition; the split uses a fixed seed (2026) and is made before any training (`data/processed/split.json`).
5. **Known limitation:** bird and drone videos were recorded on **different days** (no session contains both classes), so the sky, weather and camera settings can differ by class. A model may learn these instead of the object. Drones are only 3 models, mostly the F450 quadcopter, and drone videos come from few sessions (2 in training, 1 in validation). Daylight only, Swedish airports.
