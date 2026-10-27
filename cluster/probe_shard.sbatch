#!/bin/bash
# One shard of a temperature probe on one GPU. Usage: sbatch cluster/probe_shard.sbatch MODEL START STOP OUT [DATA]
#SBATCH --account=project_ucm_rossgreer
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/CAVLA/artifacts/slurm-%x-%j.log
set -e
source /data/wesleyferreiramaia/CAVLA/cluster/env.sh
cd $CAVLA_ROOT
MODEL=$1;START=$2;STOP=$3;OUT=$4;DATA=${5:-artifacts/physicalai_heldout_contexts_v3_20261027}
nvidia-smi -L
.venv-probe/bin/python -m cavla.multimodel_temperature_probe --model $MODEL --data $DATA --out $OUT --range $START $STOP --resume
