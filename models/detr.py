"""
Models

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import clip
import torch
import torchvision.ops.boxes as box_ops
from collections import OrderedDict
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops._utils import _cat
from typing import Optional, List, Tuple
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import transform
from torchvision.transforms import Compose, ToTensor, ToPILImage

import pocket.models as models
from datasets.transforms import HOINetworkTransform
from .interaction_head import InteractionHead, GraphHead
from .reltransformer import reldn_head
from .ops import warpaffine_image, warp_affine_joints



class GenericHOINetwork(nn.Module):
    """A generic architecture for HOI classification

    Parameters:
    -----------
        backbone: nn.Module
        interaction_head: nn.Module
        transform: nn.Module
        postprocess: bool
            If True, rescale bounding boxes to original image size
    """
    def __init__(self,
        backbone_name: str,
        backbone: nn.Module, 
        interaction_head: nn.Module,
        transform: nn.Module, postprocess: bool = True,
        rank: int = 0
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = backbone
        self.interaction_head = interaction_head
        self.transform = transform
        self.postprocess = postprocess
        self.rank = rank
        self.topilimage = ToPILImage()
        self.instance_norm = nn.InstanceNorm2d(256, affine=False)

    def preprocess(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> Tuple[
        List[Tensor], List[dict],
        List[dict], List[Tuple[int, int]]
    ]:
        # device = torch.device(f"cuda:{self.rank}")
        device = images[0].device
        original_image_sizes = [img.shape[-2:] for img in images]
        
        if self.backbone_name == "resnet50":
            images, targets = self.transform(images, targets)

            # for debug
            for i, target in enumerate(targets):
                target['transformed_image'] = images.tensors[i]
                
        elif self.backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            topilimage = ToPILImage()
            totensor = ToTensor()
            processed_image_list = []
            trans_list = []
            img_meta_list = []
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(-1, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(-1, 1, 1)
            for img in images:
                img = topilimage(img)
                img, trans, img_meta = warpaffine_image(img, n_px=672, device=device)
                img = totensor(img).to(device)
                processed_image_list.append(img.sub_(mean).div_(std))
                trans_list.append(trans)
                img_meta_list.append(img_meta)
            
            processed_image_sizes = [img.shape[-2:] for img in processed_image_list]
            images = torch.stack(processed_image_list, dim=0)
            
            for i, (tar, o_im_s, im_s, trans) in enumerate(zip(
                targets, original_image_sizes, processed_image_sizes, trans_list
            )):      
                    target_h = tar['boxes_h']
                    tar['boxes_h'] = warp_affine_joints(target_h, trans)
                    target_o = tar['boxes_o']
                    tar['boxes_o'] = warp_affine_joints(target_o, trans)
                    
                    tar['transformed_image'] = images[i]

            
                   
        return images, targets, original_image_sizes
    

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
            images: List[Tensor]
            targets: List[dict]

        Returns:
        --------
            results: List[dict]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images, targets, original_image_sizes = self.preprocess(
                images, targets)
 
        if self.backbone_name == "resnet50":
            features = self.backbone(images.tensors)
            results = self.interaction_head(features, images.image_sizes, targets)

        elif self.backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            image_sizes = [img.shape[-2:] for img in images]
            features = OrderedDict()
            results = self.interaction_head(features, image_sizes, targets, images)
        
     
        if self.postprocess and results is not None:
            return self.transform.postprocess(
                results,
                images.image_sizes,
                original_image_sizes
            )
        else:
            return results

class SpatiallyConditionedGraph(GenericHOINetwork):
    def __init__(self,
        # Backbone parameters
        backbone_name: str = "resnet50",
        use_rel_head: bool = False,
        use_separate_classifier: bool = False,
        pretrained: bool = True,
        # Pooler parameters
        output_size: int = 7,
        sampling_ratio: int = 2,
        # Box pair head parameters
        node_encoding_size: int = 1024,
        representation_size: int = 1024,
        num_object_classes: int = 5330,
        num_classes: int = 117,
        box_score_thresh: float = 0.2,
        fg_iou_thresh: float = 0.95,
        ignore_iou_thresh: float = 0.5,
        num_iterations: int = 2,
        distributed: bool = False,
        # Transformation parameters
        min_size: int = 800, max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        postprocess: bool = True,
        # Preprocessing parameters
        box_nms_thresh: float = 0.5,
        max_human: int = 15,
        max_object: int = 15,
        rank: int = 0,
    ) -> None:
        
        if backbone_name == "resnet50":
            detector = models.fasterrcnn_resnet_fpn(backbone_name,
                pretrained=pretrained)
            backbone = detector.backbone
            out_channels = backbone.out_channels        
            
        elif backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            patch_size = int(backbone_name.split('_')[-1])
            if patch_size in [16, 32]:
                backbone, _ = clip.load(f"ViT-B/{patch_size}", jit=False)
            elif patch_size in [14]:
                backbone, _ = clip.load(f"ViT-L/{patch_size}", jit=False)
            del backbone.token_embedding
            del backbone.positional_embedding
            del backbone.ln_final
            del backbone.text_projection
            del backbone.transformer
            del backbone.vocab_size
            del backbone.visual.proj

            del backbone.logit_scale
            pretrained_img_size = 224
            input_img_size = 672
            scale_factor = input_img_size // pretrained_img_size
            pretrained_width = pretrained_img_size // patch_size
            input_width = scale_factor * pretrained_width
            out_channels = 768
            backbone = backbone.float()
            cls_pos_embedding = backbone.visual.positional_embedding[:1]
            pre_pos_embedding = backbone.visual.positional_embedding[1:].view(pretrained_width,pretrained_width,-1).permute(2,0,1)
            
            post_pos_embedding = F.interpolate(pre_pos_embedding.unsqueeze(0), scale_factor=scale_factor, mode='bilinear')[0]
            expanded_pos_embedding = torch.cat([cls_pos_embedding, post_pos_embedding.permute(1,2,0).view(input_width*input_width,-1)], dim=0)
            backbone.visual.positional_embedding = torch.nn.Parameter(expanded_pos_embedding)
        else:
            raise ValueError("Not supported backbone name")
        
        if backbone_name == "resnet50":
            box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=output_size,
            sampling_ratio=sampling_ratio
        )
        elif backbone_name == "CLIP":
            box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=output_size, sampling_ratio=sampling_ratio)
        elif backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=5, sampling_ratio=sampling_ratio)
        else:
            raise ValueError("Not supported backbone name")

        box_pair_head = GraphHead(
            backbone_name=backbone_name,
            out_channels=out_channels,
            roi_pool_size=output_size,
            node_encoding_size=node_encoding_size,
            representation_size=representation_size,
            num_cls=num_classes,
            num_obj_cls=num_object_classes,
            fg_iou_thresh=fg_iou_thresh,
            ignore_iou_thresh=ignore_iou_thresh,
            num_iter=num_iterations
        )
        
        if use_rel_head:
            rel_transformer_head = reldn_head(representation_size * 4)  # concat of SPO
        else:
            rel_transformer_head = None
        
        if use_separate_classifier:
            box_h_predictor = nn.Linear(representation_size, num_object_classes)
            box_o_predictor = nn.Linear(representation_size, num_object_classes)
        else:
            box_h_predictor = nn.Linear(representation_size, num_object_classes)
            box_o_predictor = None
        
        if rel_transformer_head is not None:
            box_pair_predictor = nn.Linear(representation_size, num_classes)
        else:
            box_pair_predictor = nn.Linear(representation_size * 2, num_classes)

        interaction_head = InteractionHead(
            backbone_name=backbone_name,
            backbone=backbone if backbone_name != "resnet50" else None,
            box_roi_pool=box_roi_pool,
            box_pair_head=box_pair_head,
            rel_transformer_head=rel_transformer_head,
            box_pair_predictor=box_pair_predictor,
            box_h_predictor=box_h_predictor,
            box_o_predictor=box_o_predictor,
            num_classes=num_classes,
            box_nms_thresh=box_nms_thresh,
            box_score_thresh=box_score_thresh,
            fg_iou_thresh=fg_iou_thresh,
            max_human=max_human,
            max_object=max_object,
            distributed=distributed
        )
        
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        if backbone_name == "resnet50":
            transform = HOINetworkTransform(min_size, max_size,
            image_mean, image_std)
        elif backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            transform  = Compose([
                        ToTensor()
                    ])
        super().__init__(backbone_name, backbone, interaction_head, transform, postprocess, rank)


def build(args):
    if args.dataset == 'vg8k':
        num_classes = 2000 + 1 # add non-interaction category
        num_object_classes = 5330
    elif args.dataset == 'gvqa':
        num_classes = 310 + 1 # add non-interaction category
        num_object_classes = 1703
        
    model = SpatiallyConditionedGraph(
        backbone_name=args.backbone_name,
        use_rel_head=args.rel_head,
        use_separate_classifier=args.seperate_classifier,
        num_classes=num_classes,
        num_object_classes=num_object_classes,
        num_iterations=args.num_iter, postprocess=False,
        max_human=args.max_human, max_object=args.max_object,
        fg_iou_thresh=args.fg_iou_thresh,
        ignore_iou_thresh=args.ignore_iou_thresh,
        box_score_thresh=args.box_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        distributed=True
    )
    
    device = torch.device(args.device)

    return model
