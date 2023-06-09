# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import test
from models import build_model, ModelEma
from metric import SetCriterion


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float,
                        help="The multiplier by which the learning rate is reduced")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--milestones', nargs='+', default=[6,], type=int,
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # * Backbone
    parser.add_argument('--backbone_name', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    # SCG
    parser.add_argument('--box-score-thresh', default=0.99, type=float)
    parser.add_argument('--fg-iou-thresh', default=1.0, type=float)
    parser.add_argument('--ignore-iou-thresh', default=0.5, type=float)
    parser.add_argument('--box-nms-thresh', default=0.5, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    
    # separate human and object classfier 
    parser.add_argument('--seperate-classifier', action='store_true', default=False,
                        help='apply separate human and object classifier')
    
    # RelHead
    parser.add_argument('--rel-head', action='store_true', default=False,
                        help='apply relhead')
    
    # EMA
    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=10, type=int, metavar='M',
                        help='start ema epoch')

    # Loss
    parser.add_argument('--loss-type', default='weighted_cross_entropy', type=str)

    # dataset parameters
    parser.add_argument('--dataset', default='vg8k')
    parser.add_argument('--data-root', default='data', type=str)

    parser.add_argument('--output_dir', default='logs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='exps/vit16_SCG_WCE/checkpoint0006.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
 
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser



def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

 
    dataset_test = build_dataset(image_set='test', args=args)

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)


    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 pin_memory=True,
                                 drop_last=False, collate_fn=utils.custom_collate, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    dataset_stat = {
        'name': args.dataset,
        'obj_categories': dataset_test.dataset._objects,
        'prd_categories': dataset_test.dataset._verbs,
        'prd_freq_dict': dataset_test.dataset._prd_freq_dict,
        'obj_freq_dict': dataset_test.dataset._obj_freq_dict
    }
    

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    test(model, data_loader_test, device, dataset_stat, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
