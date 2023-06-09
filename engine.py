# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import json
import json_numpy
import numpy as np
from typing import Iterable

import torch
import torchvision.ops.boxes as box_ops
import torch.nn.functional as F
import torch.distributed as dist
import util.misc as utils
from metric.vrr_eval import generate_csv_file_from_det_obj, get_metrics_from_csv, get_many_medium_few_scores


def train_one_epoch(model: torch.nn.Module, 
                    ema_m: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    amp: bool = False, ema_epoch: int = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = [s.to(device) for s in samples]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
  
        if epoch >= ema_epoch:
            ema_m.update(model)
       
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
     
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, dataset_stat, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    all_results = []
    
    with torch.no_grad():
        for samples, targets in metric_logger.log_every(data_loader, 20, header):
            samples = [s.to(device) for s in samples]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                 **loss_dict_reduced_scaled,
                                 **loss_dict_reduced_unscaled)
            
            results = synchronise_and_log_results(outputs)
            
            all_results += results

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    saved_data_name = 'saved_data_tmp.{}.json'.format(dist.get_rank())
    
    with open(os.path.join(output_dir, saved_data_name), 'w') as f:
        encoded_arr_str = json_numpy.dumps(all_results)
        f.write(encoded_arr_str)

    if dist.get_world_size() > 1:
        dist.barrier()
        
    csv_path = os.path.join(output_dir, 'eval.csv')
    if dist.get_rank() == 0:
          
        merged_results = []
        
        filenamelist = ['saved_data_tmp.{}.json'.format(ii) for ii in range(dist.get_world_size())]
        
        for filename in filenamelist:
            merged_results += json_numpy.load(open(os.path.join(output_dir, filename), 'r'))
 
        obj_categories = dataset_stat['obj_categories']
        prd_categories = dataset_stat['prd_categories']
        prd_freq_dict = dataset_stat['prd_freq_dict']
        obj_freq_dict = dataset_stat['obj_freq_dict']
        
        generate_csv_file_from_det_obj(merged_results, 
                                       csv_path, 
                                       obj_categories, 
                                       prd_categories, 
                                       obj_freq_dict, 
                                       prd_freq_dict)
    
        overall_metrics, per_class_metrics = get_metrics_from_csv(csv_path)
   
        obj_acc = per_class_metrics[f'{csv_path}_obj_top1']
        sbj_acc = per_class_metrics[f'{csv_path}_sbj_top1']
        prd_acc = per_class_metrics[f'{csv_path}_rel_top1']
        avg_obj_sbj = (obj_acc + sbj_acc) / 2.0
        avg_acc = (prd_acc + avg_obj_sbj) / 2.0
    
        stats = {
            'obj_acc': obj_acc,
            'sbj_acc': sbj_acc,
            'prd_acc': prd_acc,
            'avg_obj_sbj': avg_obj_sbj,
            'avg_acc': avg_acc,
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics,
            'csv_path': csv_path
        }
        
        return stats


@torch.no_grad()
def test(model, data_loader, device, dataset_stat, output_dir):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_results = []
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = [s.to(device) for s in samples]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, targets)
        
        results = synchronise_and_log_results_sync(outputs)
        all_results += results

    if dist.get_rank() == 0:
        csv_path = os.path.join(output_dir, 'test.csv')
        dataset_name = dataset_stat['name']
        obj_categories = dataset_stat['obj_categories']
        prd_categories = dataset_stat['prd_categories']
        prd_freq_dict = dataset_stat['prd_freq_dict']
        obj_freq_dict = dataset_stat['obj_freq_dict']
        
        generate_csv_file_from_det_obj(all_results, 
                                       csv_path, 
                                       obj_categories, 
                                       prd_categories, 
                                       obj_freq_dict, 
                                       prd_freq_dict)
        
        cutoffs = [0.8, 0.95]
        if 'vg8k' in data_loader.dataset.dataset.root:
            data_dir = 'data/vg8k/'
            ann_dir = 'data/vg8k/seed3/'
        elif 'gvqa' in data_loader.dataset.dataset.root:
            data_dir = 'data/gvqa/'
            ann_dir = 'data/gvqa/seed0/'
        csv_file = csv_path
        get_many_medium_few_scores(csv_file, cutoffs, dataset_name, data_dir, ann_dir, syn=True)
    
        

def synchronise_and_log_results(output):
    
        detections = []
        for result in output:
            # unique gt
            unique_boxes_h = result['boxes_h']
            unique_boxes_o = result['boxes_o']
            
            # all gt   
            gt_boxes_h = result['gt_boxes_h']
            gt_boxes_o = result['gt_boxes_o']
            
            x, y = torch.nonzero(torch.min(
                box_ops.box_iou(gt_boxes_h, unique_boxes_h),
                box_ops.box_iou(gt_boxes_o, unique_boxes_o)
            ) >= 1).unbind(1)

            # predication
            sbj_scores_out = F.softmax(result['scores_h'][y], dim=1).detach().cpu().numpy()
            obj_scores_out = F.softmax(result['scores_o'][y], dim=1).detach().cpu().numpy()
            prd_scores_out = F.softmax(result['scores_p'][y], dim=1).detach().cpu().numpy()
            
            gt_sbj_labels = result['gt_labels_h'].detach().cpu().numpy()
            gt_obj_labels = result['gt_labels_o'].detach().cpu().numpy()
            gt_prd_labels = result['gt_labels_p'].detach().cpu().numpy()
            # remove non interaction category
            gt_prd_labels = gt_prd_labels - 1
            
            id = result['fname'].item()
            fname = f'{id}.jpg'
        
            detections.append(
                {
                    'image': fname,
                    'sbj_scores_out': sbj_scores_out,
                    'obj_scores_out': obj_scores_out,
                    'prd_scores_out': prd_scores_out,
                    'gt_sbj_labels': gt_sbj_labels,
                    'gt_obj_labels': gt_obj_labels,
                    'gt_prd_labels': gt_prd_labels
                }
            )
    
        return detections
    
    
    
def synchronise_and_log_results_sync(output):
    
    detections = []
    for result in output:
        # unique gt
        unique_boxes_h = result['boxes_h']
        unique_boxes_o = result['boxes_o']
        
        # all gt   
        gt_boxes_h = result['gt_boxes_h']
        gt_boxes_o = result['gt_boxes_o']

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(gt_boxes_h, unique_boxes_h),
            box_ops.box_iou(gt_boxes_o, unique_boxes_o)
        ) >= 1).unbind(1)

        # predication
        sbj_scores_out = F.softmax(result['scores_h'][y], dim=1).detach().cpu().numpy()
        obj_scores_out = F.softmax(result['scores_o'][y], dim=1).detach().cpu().numpy()
        prd_scores_out = F.softmax(result['scores_p'][y], dim=1).detach().cpu().numpy()
        
        gt_sbj_labels = result['gt_labels_h'].detach().cpu().numpy()
        gt_obj_labels = result['gt_labels_o'].detach().cpu().numpy()
        gt_prd_labels = result['gt_labels_p'].detach().cpu().numpy()
        # remove non interaction category
        gt_prd_labels = gt_prd_labels - 1
        
        id = result['fname'].item()
        fname = f'{id}.jpg'
    
        detections.append(
            {
                'image': fname,
                'sbj_scores_out': sbj_scores_out,
                'obj_scores_out': obj_scores_out,
                'prd_scores_out': prd_scores_out,
                'gt_sbj_labels': gt_sbj_labels,
                'gt_obj_labels': gt_obj_labels,
                'gt_prd_labels': gt_prd_labels
            }
        )

    # Sync across subprocesses
    all_detection_sync = utils.all_gather(detections)
    all_results_sync = []
    for item in all_detection_sync:
        all_results_sync += item
    # Collate and log results in master process
    return all_results_sync
