#!/usr/bin/env bash

set -e
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=127.0.0.1:${PORT} train.py ${PY_ARGS}

