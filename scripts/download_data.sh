#!/usr/bin/env bash
# Download the Halmstad multi-sensor drone detection dataset (CC0-1.0).
# Authors: Maria Habak, [Partner name]
set -euo pipefail
mkdir -p data/raw
if [ ! -d data/raw/Drone-detection-dataset ]; then
  git clone --depth 1 https://github.com/DroneDetectionThesis/Drone-detection-dataset.git \
    data/raw/Drone-detection-dataset
fi
echo "Dataset ready in data/raw/Drone-detection-dataset"
