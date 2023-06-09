#!/usr/bin/env bash

set -x

EXP_DIR=exps/vit16_vg8k_SCG_RLTrans_WCE
PY_ARGS=${@:1}

python -u main.py \
    --backbone_name CLIP_ViT_16 \
    --dataset vg8k \
    --batch_size 2 \
    --loss-type "weighted_cross_entropy" \
    --output_dir ${EXP_DIR} \
    --rel-head \
    --resume exps/vit16_vg8k_SCG_RLTrans_WCE/checkpoint0004.pth \
    ${PY_ARGS}
