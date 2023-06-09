#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_gvqa_SCG_WCE
PY_ARGS=${@:1}

python -u main.py \
    --backbone_name resnet50 \
    --dataset gvqa \
    --batch_size 4 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
