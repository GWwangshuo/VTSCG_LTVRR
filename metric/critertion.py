import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from .focal_loss import focal_loss


def get_freq_from_dict(freq_dict, categories):
    freqs = np.zeros(len(categories))
    for i, cat in enumerate(categories):
        if cat in freq_dict.keys():
            freqs[i] = freq_dict[cat]
        else:
            freqs[i] = 0
    return freqs


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, dataset_stat, weight_dict, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        
        self._rank = dist.get_rank()
        self.weight_dict = weight_dict
        
        self.prd_weights = None
        self.obj_weights = None

        self.obj_categories = dataset_stat['obj_categories']
        self.prd_categories = dataset_stat['prd_categories']
        self.prd_freq_dict = dataset_stat['prd_freq_dict']
        self.obj_freq_dict=  dataset_stat['obj_freq_dict']
        
        self.loss_type = kwargs['loss_type']
        
        if self.loss_type == 'weighted_cross_entropy':
            self.freq_prd = get_freq_from_dict(self.prd_freq_dict, self.prd_categories)
            self.freq_obj = get_freq_from_dict(self.obj_freq_dict, self.obj_categories)
            freq_prd = self.freq_prd + 1
            freq_obj = self.freq_obj + 1
            prd_weights = np.sum(freq_prd) / freq_prd
            obj_weights = np.sum(freq_obj) / freq_obj

            self.prd_weights = (prd_weights / np.mean(prd_weights)).astype(np.float32)
            self.obj_weights = (obj_weights / np.mean(obj_weights)).astype(np.float32)
            temp = np.zeros(shape=self.prd_weights.shape[0] + 1, dtype=np.float32) 
            temp[1:] = self.prd_weights
            temp[0] = min(self.prd_weights)
            self.prd_weights = temp


    def get_cls_loss(self, cls_scores, labels, weight=None, loss_type='cross_entropy'):
        if weight is not None:
            weight = torch.tensor(weight).to(labels.device)
        if loss_type == 'cross_entropy':
            return F.cross_entropy(cls_scores, labels)
        elif loss_type == 'weighted_cross_entropy':
            return F.cross_entropy(cls_scores, labels, weight=weight)
        elif loss_type == 'focal':
            cls_scores_exp = cls_scores.unsqueeze(2)
            cls_scores_exp = cls_scores_exp.unsqueeze(3)
            labels_exp = labels.unsqueeze(1)
            labels_exp = labels_exp.unsqueeze(2)
            return focal_loss(cls_scores_exp, labels_exp, alpha=0.25, gamma=2.0, reduction='mean')
        else:
            raise NotImplementedError
        
    def get_nan_idx(self, scores, labels, weights):
        num = len(scores)
        ret = []
        for i in range(num):
            loss = self.get_cls_loss(scores[i], labels[i], weights, self.loss_type)
            if loss.isnan():
                ret.append(i)
        return ret
        
        
    def loss_factory(self, results, weights, loss_type, target='subject'):
        
        scores = []; labels = []
        for result in results:
            if target == 'subject':
                scores_ret =  result['scores_h']
                labels_ret = result['labels_h']
            elif target == 'object':
                scores_ret =  result['scores_o']
                labels_ret = result['labels_o']
            elif target == 'predicate':
                scores_ret =  result['scores_p']
                labels_ret = result['labels_p']
            
            scores.append(scores_ret)
            labels.append(labels_ret)

        preds_ = torch.cat(scores)
        labels_ = torch.cat(labels)
        
        loss = self.get_cls_loss(preds_, labels_, weights, loss_type)
        
        if loss.isnan():
            nan_idx = self.get_nan_idx(scores, labels, weights)[0]
            nan_target = results[nan_idx]
            # image = np.uint8(self.unnormalize(nan_target['image']).permute(1, 2, 0).detach().cpu().numpy() * 255)
            # image = Image.fromarray(image)
            id = nan_target['fname'].item()
            filename = f'{id}.jpg'
            for i, (sbj_box, obj_box, sbj, obj, prd) in enumerate(zip(nan_target['boxes_h'], nan_target['boxes_o'],
                                                    nan_target['labels_h'], nan_target['labels_o'],
                                                    nan_target['labels_p'])):
                
                sbj_box = sbj_box.detach().cpu().numpy().tolist()
                obj_box = obj_box.detach().cpu().numpy().tolist()
                
                sbj = self.obj_categories[sbj.item()]
                obj = self.obj_categories[obj.item()]
                prd = (['non_interaction'] + self.prd_categories)[prd.item()]
                
                # save_name = filename.replace('.jpg', f'_{i}_{sbj}_{prd}_{obj}.jpg')
                # pil_draw(image, [sbj_box, obj_box], [sbj, obj], save_name)
                print(filename, sbj, prd, obj)
                
            raise ValueError(f"The prd loss of {filename} is NaN for rank {self._rank}")
    
        
        cls_preds = preds_.max(dim=1)[1].type_as(labels_)
        accuracy_cls = cls_preds.eq(labels_).float().mean(dim=0)
        
        return loss, accuracy_cls

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
 
        loss_sbj, acc_sbj = self.loss_factory(outputs, self.obj_weights, self.loss_type, 'subject')
        loss_obj, acc_obj = self.loss_factory(outputs, self.obj_weights, self.loss_type, 'object')
        loss_prd, acc_prd = self.loss_factory(outputs, self.prd_weights, self.loss_type, 'predicate')

        loss_dict = dict(
            loss_sbj=loss_sbj,
            loss_obj=loss_obj,
            loss_prd=loss_prd,
            acc_sbj=acc_sbj,
            acc_obj=acc_obj,
            acc_prd=acc_prd
            )
        
        return loss_dict