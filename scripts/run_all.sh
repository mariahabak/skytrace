#!/usr/bin/env bash
# Rerun the whole main experiment: data, all models over 3 seeds, summary.
# Authors: Maria Habak, [Partner name]
#
#   bash scripts/run_all.sh          # prepare data + train + validation table
#   bash scripts/run_all.sh --final  # also the ONE-TIME test evaluation
set -euo pipefail
FINAL="${1:-}"

bash scripts/download_data.sh
python -m skytrace.prepare configs/data.json

# Order matters: frame_single trains the encoder that the others reuse.
EXPERIMENTS="baseline baseline_no_geometry frame_single frame_mean \
temporal_mean temporal_gru temporal_gru_shuffled"

for e in $EXPERIMENTS; do
  python -m skytrace.run "configs/$e.json" --stage fit
done
python -m skytrace.summarize --split val

if [ "$FINAL" = "--final" ]; then
  for e in $EXPERIMENTS; do
    python -m skytrace.run "configs/$e.json" --stage test --final
  done
  python -m skytrace.summarize --split test
fi
