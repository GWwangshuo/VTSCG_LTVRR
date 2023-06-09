#!/usr/bin/env bash

set -x

EXP_DIR=nexps/vit32_gvqa_SCG_WCE
PY_ARGS=${@:1}

python -u main.py \
    --backbone_name CLIP_ViT_32 \
    --dataset gvqa \
    --batch_size 8 \
    --seperate-classifier \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
