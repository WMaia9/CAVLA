#!/bin/bash
# Qwen-Drive v2 probe (reference inside the trained time range) for both planners, resumable.
set -e
cd /mnt/datasets/CAVLA
export HF_TOKEN=$(cat .deps/hf_token) PYTHONPATH=src
M=/home/perceptron-lab/.cache/huggingface/hub/models--Qwen--Qwen-Drive-1.0-4B/snapshots/28484089a7cc8c335cf5089fb0745cf7c49b6eaa
for planner in rl sft; do
  .deps/qwendrive_venv/bin/python -m cavla.qwendrive_temperature_probe_v2 --records artifacts/physicalai_test_contexts_v1_20261022/records.json \
    --out artifacts/qwendrive_probe_${planner}_v2_20261027 --model $M --planner planner-$planner --range 100 160 --expert-fp32 --resume \
    >> artifacts/qwendrive_probe_${planner}_v2_20261027.log 2>&1
done
