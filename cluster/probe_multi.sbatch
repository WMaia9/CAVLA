#!/bin/bash
# One-hour test-partition job: 4 H200 (2 nodes x 2), one probe shard per GPU, resumable.
# Usage: sbatch cluster/probe_multi.sbatch MODEL OUT_PREFIX START STOP [DATA]; the clip range is split into 4 contiguous shards OUT_PREFIX_c{0..3}.
#SBATCH --account=project_ucm_rossgreer
#SBATCH --partition=test
#SBATCH --qos=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=00:59:00
#SBATCH --output=/data/wesleyferreiramaia/CAVLA/artifacts/slurm-%x-%j.log
source /data/wesleyferreiramaia/CAVLA/cluster/env.sh
cd $CAVLA_ROOT
export MODEL=$1 PREFIX=$2 START=$3 STOP=$4 DATA=${5:-artifacts/physicalai_heldout_contexts_v3_20261027}
srun --ntasks=4 --ntasks-per-node=2 --gpus-per-task=1 bash -c '
  n=$(( (STOP-START+3)/4 )); a=$(( START+SLURM_PROCID*n )); b=$(( a+n )); [ $b -gt $STOP ] && b=$STOP
  echo "task $SLURM_PROCID on $(hostname) gpu $CUDA_VISIBLE_DEVICES clips $a-$b"
  .venv-probe/bin/python -m cavla.multimodel_temperature_probe --model $MODEL --data $DATA --out ${PREFIX}_c$SLURM_PROCID --range $a $b --resume'
