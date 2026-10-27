#!/bin/bash
# Addendum 13, R1 then A2 on held-out clips: three processes of the unchanged probe on disjoint clip ranges
# (one model on the GPU at a time), then merge. Replaces the sequential run_heldout_v3.sh for the remaining clips.
set -e
cd /mnt/datasets/CAVLA
export HF_TOKEN=$(cat .deps/hf_token) PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/mnt/datasets/Explainability/alpamayo/ar1_venv/bin/python;D=artifacts/physicalai_heldout_contexts_v3_20261027
run(){ # model start stop outdir
  $PY -m cavla.multimodel_temperature_probe --model $1 --data $D --out $4 --range $2 $3 --resume >> $4.log 2>&1
}
for m in r1 a2; do
  M=artifacts/temperature_probe_${m}_heldout_v3_20261027
  if [ $m = r1 ]; then a=83;b=102; else a=40;b=80; fi
  run $m 0 $a $M & p1=$!; sleep 120
  run $m $a $b ${M}_shard2 & p2=$!; sleep 120
  run $m $b 120 ${M}_shard3 & p3=$!
  wait $p1; wait $p2; wait $p3
  $PY -m cavla.merge_probe_shards --main $M --shards ${M}_shard2 ${M}_shard3 --range 0 120 >> $M.log 2>&1
  echo "[sharded] $m done $(date)" >> artifacts/heldout_v3_sharded.log
done
