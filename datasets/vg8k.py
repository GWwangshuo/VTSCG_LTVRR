"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Callable, Tuple
from torchvision.ops import clip_boxes_to_image
from pocket.data import ImageDataset, DataSubset

from util.box_ops import unique_boxes
from PIL import Image, ImageDraw, ImageFont



def pil_draw(img_path, bboxes, labels, save_name):
    # 1.读取图片
    im = Image.open(img_path)
    font = ImageFont.truetype(font='./Gemelli.ttf', size=np.floor(1.5e-2 * np.shape(im)[1] + 15).astype('int32'))
    draw = ImageDraw.Draw(im)
    
    for bbox, label in zip(bboxes, labels):
        # 获取label长宽
        label_size = draw.textsize(label, font)
        # 设置label起点
        text_origin = np.array([bbox[0], bbox[1] - label_size[1]])

        # 绘制矩形框，加入label文本
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]],outline='red',width=2)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill='red')
        draw.text(text_origin, str(label), fill=(255, 255, 255), font=font)

    del draw
    
    im.save(save_name)

class VG8KSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]
    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]
    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno
    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno
    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno

class VG8K(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        anno_file(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """
    def __init__(self, root: str, anno_file: str, det_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        super(VG8K, self).__init__(root, transform, target_transform, transforms)
        
        self.root = root
        object_path = '/'.join(anno_file.split('/')[:-1] + ['objects.json'])
        predicate_path = '/'.join(anno_file.split('/')[:-1] + ['predicates.json'])
        
        objects_freqs_path = '/'.join(anno_file.split('/')[:-1] + ['train_objects_freqs.json'])
        prd_freqs_path = '/'.join(anno_file.split('/')[:-1] + ['train_predicates_freqs.json'])
        
        with open(objects_freqs_path, 'r') as f:
            self._obj_freq_dict = json.load(f)
            
        with open(prd_freqs_path, 'r') as f:
            self._prd_freq_dict = json.load(f)

        with open(det_file, 'r') as f:
            dets = json.load(f)
            
        with open(object_path, 'r') as f:
            self._objects = json.load(f)
            
        with open(predicate_path, 'r') as f:
            self._verbs = json.load(f)

        with open(anno_file, 'r') as f:
            anno = json.load(f)
            
        self.num_object_cls = len(self._objects)
        self.num_interation_cls = len(self._verbs)
        self._anno_file = anno_file
        self._dets = dets

        # Load annotations
        self._load_annotation_and_metadata(anno)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._idx)

    def clip_boxes(self, target, w, h):
        target['boxes_h'] = clip_boxes_to_image(target['boxes_h'], (h, w))
        target['boxes_o'] = clip_boxes_to_image(target['boxes_o'], (h, w))
        return target
    
    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image
        
        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
        """
        intra_idx = self._idx[i]

        # print(os.path.join(self.root, self._filenames[intra_idx]))
        file_name = self._filenames[intra_idx]
        image, anno = self._transforms(
            self.load_image(os.path.join(self._root, file_name)), 
            self._anno[intra_idx]
            )
        
        width, height = image.size
        anno = self.clip_boxes(anno, width, height)
        
        id = torch.tensor([int(file_name.split('.')[0])])
        anno['id'] = id
        return image, anno

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno


    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self) -> List[int]:
        """
        Number of annotated box pairs for each object class

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def objects(self) -> List[str]:
        """
        Object names 

        Returns:
            list[str]
        """
        return self._objects.copy()

    @property
    def verbs(self) -> List[str]:
        """
        Verb (action) names

        Returns:
            list[str]
        """
        return self._verbs.copy()

    @property
    def interactions(self) -> List[str]:
        """
        Combination of verbs and objects

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i] 
            for _, i, j in self._class_corr]

    def split(self, ratio: float) -> Tuple[VG8KSubset, VG8KSubset]:
        """
        Split the dataset according to given ratio

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset)
            val(Dataset)
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return VG8KSubset(self, perm[:n]), VG8KSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, f: dict) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """

        invalid_filenames = ['2316810.jpg', '2322882.jpg', '2412.jpg', '2315762.jpg', 
                             '3258.jpg', '2326431.jpg', '2388565.jpg', '2337206.jpg', 
                             '2338230.jpg', '2326508.jpg', '2381559.jpg']
        
        # # vit invalid filenames
        # invalid_filenames += ['2361702.jpg', '2342066.jpg', '285761.jpg', '2388920.jpg', '2370581.jpg',
        #                       '2388535.jpg', '2349001.jpg', '2383703.jpg', '2389006.jpg', '2405738.jpg',
        #                       '2413169.jpg', '2413905.jpg', '2414404.jpg', '2414375.jpg', '2388524.jpg']
        
        self._filenames = [anno['file_name'] for anno in f if anno['file_name'] not in invalid_filenames]
        self._image_sizes = [(anno['height'], anno['width']) for anno in f if anno['file_name'] not in invalid_filenames]

        idx = list(range(len(self._filenames)))

        annos = []     
     
        for anno in tqdm(f):
            if anno['file_name'] not in invalid_filenames:
     
                annos.append(
                    {   
                        'boxes_h': anno['sbj_gt_boxes'],
                        'boxes_o': anno['obj_gt_boxes'],
                        'h_keep_idx': list(unique_boxes(np.array(anno['sbj_gt_boxes']))),
                        'o_keep_idx': list(unique_boxes(np.array(anno['obj_gt_boxes']))),
                        'subject': anno['sbj_gt_classes'],
                        'object': anno['obj_gt_classes'],
                        'verb': list(np.array(anno['prd_gt_classes']) + 1)
                    }
                )
                
        self._idx = idx
        self._anno = annos
        self._class_corr = None
        self._empty_idx = []
      
