#!/bin/bash
# Addendum 13: A1.5, R1, A2 temperature probes on clips held out of Alpamayo training, resumable.
set -e
cd /mnt/datasets/CAVLA
export HF_TOKEN=$(cat .deps/hf_token) PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/mnt/datasets/Explainability/alpamayo/ar1_venv/bin/python;D=artifacts/physicalai_heldout_contexts_v3_20261027
$PY -m cavla.temperature_probe --data $D --out artifacts/temperature_probe_heldout_v3_20261027 --range 0 120 --resume >> artifacts/temperature_probe_heldout_v3_20261027.log 2>&1
for m in r1 a2; do
  $PY -m cavla.multimodel_temperature_probe --model $m --data $D --out artifacts/temperature_probe_${m}_heldout_v3_20261027 --range 0 120 --resume \
    >> artifacts/temperature_probe_${m}_heldout_v3_20261027.log 2>&1
done
