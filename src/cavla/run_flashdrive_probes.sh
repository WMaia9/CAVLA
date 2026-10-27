#!/bin/bash
# Runs the matched-temperature FlashDrive probe for both checkpoints, each into its own resumable directory.
set -e
cd /mnt/datasets/CAVLA
export HF_TOKEN=$(cat .deps/hf_token) HF_HOME=/mnt/datasets/hf-cache/perceptron-lab PYTHONPATH=src
for pair in "nvidia nvidia/Alpamayo-1.5-10B" "zlab z-lab/Alpamayo-1.5-10B"; do
  set -- $pair
  .deps/flashdrive/.venv/bin/python -P -m cavla.flashdrive_temperature_probe --model "$2" \
    --records artifacts/physicalai_test_contexts_v1_20261022/records.json \
    --out "artifacts/flashdrive_probe_$1_v1" --range 100 160 --windows 4 --flashdrive .deps/flashdrive --resume \
    >> "artifacts/flashdrive_probe_$1_v1.log" 2>&1
done
