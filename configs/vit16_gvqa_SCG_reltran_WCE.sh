#!/usr/bin/env bash

set -x

EXP_DIR=exps/vit16_gvqa_SCG_RLTrans_WCE_v2
PY_ARGS=${@:1}

python -u main.py \
    --backbone_name CLIP_ViT_16 \
    --dataset gvqa \
    --batch_size 2 \
    --rel-head \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
