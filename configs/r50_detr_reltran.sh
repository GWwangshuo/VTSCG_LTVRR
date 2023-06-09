#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_gvqa_SCG_RLTrans_CE
PY_ARGS=${@:1}

python -u main.py \
    --batch_size 6 \
    --dataset gvqa \
    --output_dir ${EXP_DIR} \
    --loss-type "cross_entropy" \
    --rel-head \
    ${PY_ARGS}