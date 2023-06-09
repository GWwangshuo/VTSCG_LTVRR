# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import torch.utils.data
from torch.utils.data import Dataset
from torchvision.ops import clip_boxes_to_image
from torchvision.transforms.functional import hflip
from torchvision.transforms import ColorJitter
from .vg8k import VG8K
from .gvqa import GVQA

import pocket

class DataFactory(Dataset):
    def __init__(self,
            name, partition,
            data_root,
            flip=False,
            color_jitter=False,
            box_score_thresh_h=0.2,
            box_score_thresh_o=0.2
            ):
        if name not in ['vg8k', 'gvqa']:
            raise ValueError("Unknown dataset ", name)

        assert partition in ['train', 'val', 'test'], \
            f"Unknown {name} partition " + partition
                
        if name == 'vg8k':
            self.dataset = VG8K(
                root=os.path.join(data_root, 'vg8k', 'VG_100K'),
                anno_file=os.path.join(data_root, 'vg8k', 'seed3', 'vg8k_{}_annos.json'.format(partition)),
                # anno_file=os.path.join(data_root, 'vg8k', 'seed3', '{}_annos_split0.json'.format(partition)),
                det_file=os.path.join(data_root, 'vg8k', 'seed3', 'detections_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        elif name == 'gvqa':
            self.dataset = GVQA(
                root=os.path.join(data_root, 'gvqa', 'images'),
                anno_file=os.path.join(data_root, 'gvqa', 'seed0', 'gvqa_{}_annos.json'.format(partition)),
                det_file=os.path.join(data_root, 'gvqa', 'seed0', 'detections_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        self.name = name
        self.box_score_thresh_h = box_score_thresh_h
        self.box_score_thresh_o = box_score_thresh_o
        
        self._flip = torch.randint(0, 2, (len(self.dataset),)) if flip \
            else torch.zeros(len(self.dataset))
        self._brightness = torch.randint(0, 2, (len(self.dataset),)) if color_jitter \
            else torch.zeros(len(self.dataset))
        self._contrast = torch.randint(0, 2, (len(self.dataset),)) if color_jitter \
            else torch.zeros(len(self.dataset))
        self._saturation = torch.randint(0, 2, (len(self.dataset),)) if color_jitter \
            else torch.zeros(len(self.dataset))
        self._hue = torch.randint(0, 2, (len(self.dataset),)) if color_jitter \
            else torch.zeros(len(self.dataset))
        #self._random
        self.aug_bri = ColorJitter(brightness=0.5)
        self.aug_con = ColorJitter(contrast=0.5)
        self.aug_sat = ColorJitter(saturation=0.5)
        self.aug_hue = ColorJitter(hue=0.3)
    
            
    def __len__(self):
        return len(self.dataset)
    
    def flip_boxes(self, target, w):
        target['boxes_h'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_h'])
        target['boxes_o'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_o'])
        
        # trans label
        if self.name == 'gvqa':
            idx1 = torch.where(target['verb'] == 63) # to the right of 
            idx2 = torch.where(target['verb'] == 85) # to the left of
            target['verb'][idx1] = 85
            target['verb'][idx2] = 63
            
        elif self.name == 'vg8k':
            # to left of, to right of
            # idx1 = torch.where(target['verb'] == 149) # to the right of 
            # idx2 = torch.where(target['verb'] == 133) # to the left of
            # target['verb'][idx1] = 133
            # target['verb'][idx2] = 149
            
            # # left of, right of
            # idx1 = torch.where(target['verb'] == 184) # to the right of 
            # idx2 = torch.where(target['verb'] == 259) # to the left of
            # target['verb'][idx1] = 259
            # target['verb'][idx2] = 184
            
            # # on left side of, on right side of
            # idx1 = torch.where(target['verb'] == 317) # to the right of 
            # idx2 = torch.where(target['verb'] == 315) # to the left of
            # target['verb'][idx1] = 315
            # target['verb'][idx2] = 317
            
            # # on left of, on right of
            # idx1 = torch.where(target['verb'] == 366) # to the right of 
            # idx2 = torch.where(target['verb'] == 407) # to the left of
            # target['verb'][idx1] = 407
            # target['verb'][idx2] = 366
            
            # # has left, has right
            # idx1 = torch.where(target['verb'] == 741) # to the right of 
            # idx2 = torch.where(target['verb'] == 898) # to the left of
            # target['verb'][idx1] = 898
            # target['verb'][idx2] = 741
            
             # on left side, on right side
            idx1 = torch.where(target['verb'] == 1822) # to the right of 
            idx2 = torch.where(target['verb'] == 1529) # to the left of
            target['verb'][idx1] = 1529
            target['verb'][idx2] = 1822

    def __getitem__(self, i):
        image, target = self.dataset[i]
            
        # Convert ground truth boxes to zero-based index and the
        # representation from pixel indices to coordinates
        if self.name in ['vg8k', 'gvqa']:
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1

        w, h = image.size
    
        if self._flip[i]:
            image = hflip(image)
            self.flip_boxes(target, w)
            
        # random color jittering
        if self._brightness[i]:
            image = self.aug_bri(image)
        if self._contrast[i]:
            image = self.aug_con(image)
        if self._saturation[i]:
            image = self.aug_sat(image)
        if self._hue[i]:
            image = self.aug_hue(image)
  

        target['labels'] = target['verb']
        
        image = pocket.ops.to_tensor(image, 'pil')

        return image, target


def build_dataset(image_set, args):
    dataset = DataFactory(
        name=args.dataset, 
        partition=image_set,
        data_root=args.data_root, 
        flip=True if image_set == 'train' else False,
        # flip=True,
        # color_jitter=True if image_set == 'train' else False,
        color_jitter=False
    )
    return dataset