#!/usr/bin/env bash

set -x

EXP_DIR=nexps/vit32_vg8k_SCG_WCE
PY_ARGS=${@:1}

python -u main.py \
    --backbone_name CLIP_ViT_32 \
    --dataset vg8k \
    --batch_size 8 \
    --seperate-classifier \
    --resume ${EXP_DIR}/checkpoint0002.pth \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
