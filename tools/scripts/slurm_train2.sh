#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
PY_ARGS=${@:4}


CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gpus-per-node=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --ntasks=${GPUS} \
    --mem-per-gpu=4g \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --container-mounts=/netscratch/nozarian:/netscratch/nozarian,/ds-av:/ds-av \
    --container-image=/netscratch/nozarian/openpcdet_original.sqsh \
    --container-workdir=/netscratch/nozarian/OpenPCDet_Original/tools \
    --container-save="/netscratch/nozarian/openpcdet_original2.sqsh" \
    ${SRUN_ARGS} \
    python -u train.py --launcher slurm ${PY_ARGS}

