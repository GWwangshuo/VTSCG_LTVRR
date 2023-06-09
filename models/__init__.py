# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .ema import ModelEma

def build_model(args):
    return build(args)
