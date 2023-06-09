#!/usr/bin/env bash

set -x

EXP_DIR=nexps/vit16_vg8k_SCG_WCE
PY_ARGS=${@:1}

python -u main.py \
    --backbone_name CLIP_ViT_16 \
    --dataset vg8k \
    --batch_size 2 \
    --seperate-classifier \
    --output_dir ${EXP_DIR} \

    ${PY_ARGS}
