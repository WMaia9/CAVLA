#!/bin/bash
# Run from the workstation: resubmit the one-hour multi-GPU job until every shard has a manifest (written when a shard finishes).
MODEL=$1;PREFIX=$2;START=$3;STOP=$4;R=login.rc.ucmerced.edu
while true; do
  done_shards=$(ssh -o BatchMode=yes -n $R "cd /data/wesleyferreiramaia/CAVLA; ls ${PREFIX}_c*/manifest.json 2>/dev/null | wc -l" 2>/dev/null)
  [ "$done_shards" = 4 ] && { echo "all shards done $(date)"; break; }
  ssh -o BatchMode=yes -n $R "cd /data/wesleyferreiramaia/CAVLA; sbatch --wait -J ${MODEL}-heldout cluster/probe_multi.sbatch $MODEL $PREFIX $START $STOP" 2>&1 | grep -v -i "post-quantum\|store now\|upgraded\|warning"
  sleep 20
done
