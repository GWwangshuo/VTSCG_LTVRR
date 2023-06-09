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
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from util.config import get_raw_dict
from util.regularizers import Normalizer
from datasets import build_dataset
from engine import evaluate, train_one_epoch
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
    parser.add_argument('--epochs', default=8, type=int)
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
    parser.add_argument('--resume', default='', help='resume from checkpoint')
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
    
    if dist.get_rank() == 0:
        path = os.path.join(args.output_dir, "config.json")
        
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    model.to(device)
    
    # applying L2 Normalization
    # L2_norm = Normalizer(tau=1)
    # L2_norm.apply_on(model)
    
    ema_m = ModelEma(model, args.ema_decay) # 0.9997

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    lambda1 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    lambda2 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda1, lambda2]
    )

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=True,
                                   collate_fn=utils.custom_collate, 
                                   num_workers=args.num_workers)
    
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 pin_memory=True,
                                 drop_last=False, 
                                 collate_fn=utils.custom_collate, 
                                 num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    dataset_stat = {
        'obj_categories': dataset_train.dataset._objects,
        'prd_categories': dataset_train.dataset._verbs,
        'prd_freq_dict': dataset_train.dataset._prd_freq_dict,
        'obj_freq_dict': dataset_train.dataset._obj_freq_dict
    }
    
    weight_dict = {'loss_sbj': 1, 'loss_obj': 1, 'loss_prd': 1}
    criterion = SetCriterion(dataset_stat, weight_dict, loss_type=args.loss_type)
    
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(model, criterion, data_loader_val,  device, dataset_stat, output_dir)
        return

    print("Start training")
    start_time = time.time()
    
    best = {
        'avg_per_class_acc': 0.0,
        'epoch': 0
    }

    json.dump(best, open(os.path.join(output_dir, 'best.json'), 'w'))
    
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model.module, args.ema_decay)
        
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, ema_m, criterion, 
            data_loader_train, optimizer, 
            device, epoch,
            args.clip_max_norm,
            amp=args.amp,
            ema_epoch=args.ema_epoch)
        
        lr_scheduler.step()
      
        # test_stats = evaluate(model, criterion, data_loader_val, device, dataset_stat, output_dir)
        
        # # # test_stats_ema = evaluate(ema_m.module, criterion, data_loader_val, device, dataset_stat, output_dir)
        
        if args.output_dir and utils.is_main_process():
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    #  **{f'test_{k}': v for k, v in test_stats.items()},
                    #  **{f'test_ema_{k}': v for k, v in test_stats_ema.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
            
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        #     # flag = test_stats_ema['avg_acc'] > test_stats['avg_acc']
        #     # if flag:
        #     #     test_stats = test_stats_ema
                
        #     avg_acc = test_stats['avg_acc']
        #     overall_metrics = test_stats['overall_metrics']
        #     per_class_metrics = test_stats['per_class_metrics']
        #     csv_path = test_stats['csv_path']
        #     best = json.load(open(os.path.join(output_dir, 'best.json')))
            
        #     if avg_acc > best['avg_per_class_acc']:
        #         print('Found new best validation accuracy at {:2.2f}%'.format(avg_acc))
        #         print('Saving best model..')
        #         best['epoch'] = epoch
        #         best['avg_per_class_acc'] = avg_acc
        #         best['per_class_metrics'] = {'obj_top1': per_class_metrics[f'{csv_path}_obj_top1'],
        #                         'sbj_top1': per_class_metrics[f'{csv_path}_sbj_top1'],
        #                         'prd_top1': per_class_metrics[f'{csv_path}_rel_top1']}
        #         best['overall_metrics'] = {'obj_top1': overall_metrics[f'{csv_path}_obj_top1'],
        #                 'sbj_top1': overall_metrics[f'{csv_path}_sbj_top1'],
        #                 'prd_top1': overall_metrics[f'{csv_path}_rel_top1']}
                
        #         json.dump(best, open(os.path.join(output_dir, 'best.json'), 'w'))
    
            checkpoint_path = output_dir /  f'checkpoint{epoch:04}.pth'
            
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                # 'model': model_without_ddp.state_dict() if not flag else ema_m.module.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
