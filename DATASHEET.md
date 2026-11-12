# Data sheet

*Author: Mariah Habak*

*Fill in the [brackets] from `data/processed/summary.json` after running `python -m dronebird.prepare`.*

1. **Source:** "Segmented Dataset Based on YOLOv7 for Drone vs. Bird Identification for Deep and Machine Learning Algorithms", Srivastav, Shandilya, Datta, Yemets and Nagar (2023), Mendeley Data, DOI 10.17632/6ghdz52pd7 (also described in *Data in Brief*). 20,925 JPEG images, 640 × 640, frames from Pexels images and videos, with YOLO labels.
2. **Licence / consent:** CC BY 4.0, used with attribution as above. No personal data; no images taken by us.
3. **Size:** [N] labelled images and [N] object crops (bird [N], drone [N]). Train / validation / test: [N] / [N] / [N] crops. Test sets: synthetic (objects ≥ 64 px) [N] crops; real small (objects < 48 px) [N] crops.
4. **Labelling and preparation:** labels come from the dataset (class id → bird/drone checked against the BT/DT folders). We checked 200 random training crops by hand: [N] wrong, error rate [x]% (95% CI [a]–[b]%). Crops: square, 1.5 × the longest box side, border pixels repeated, resized to 96 × 96 (README, "Method details"). Split 70/15/15 by near-duplicate group with seed 2026, made before any training: [N] near-duplicate groups with [N] images; largest group [N] frames.
5. **Known limitation:** most objects are large close-ups from stock footage, not real surveillance views, which is why we test on shrunk and on naturally small objects. Frames of one video that look different can still land in different partitions. [Add what the audit and the 200-label check showed.]
