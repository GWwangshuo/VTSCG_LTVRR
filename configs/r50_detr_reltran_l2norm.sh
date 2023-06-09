#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_gvqa_SCG_RLTrans_WCE_L2NORM
PY_ARGS=${@:1}

python -u main.py \
    --batch_size 6 \
    --dataset gvqa \
    --output_dir ${EXP_DIR} \
    --rel-head \
    ${PY_ARGS}