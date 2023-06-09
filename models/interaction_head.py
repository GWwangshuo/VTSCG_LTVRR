"""
Interaction head and its submodules

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops

from torch.nn import Module
from torch import nn, Tensor
from pocket.ops import Flatten
from typing import Optional, List, Tuple
from collections import OrderedDict

from .ops import compute_spatial_encodings

class InteractionHead(Module):
    """Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_roi_pool: Module
        Module that performs RoI pooling or its variants
    box_pair_head: Module
        Module that constructs and computes box pair features
    box_pair_suppressor: Module
        Module that computes unary weights for each box pair
    box_pair_predictor: Module
        Module that classifies box pairs
    human_idx: int
        The index of human/person class in all objects
    num_classes: int
        Number of target classes
    box_nms_thresh: float, default: 0.5
        Threshold used for non-maximum suppression
    box_score_thresh: float, default: 0.2
        Threshold used to filter out low-quality boxes
    max_human: int, default: 15
        Number of human detections to keep in each image
    max_object: int, default: 15
        Number of object (excluding human) detections to keep in each image
    distributed: bool, default: False
        Whether the model is trained under distributed data parallel. If True,
        the number of positive logits will be averaged across all subprocesses
    """
    def __init__(self,
        # Network components
        backbone_name: str,
        backbone: Module,
        box_roi_pool: Module,
        box_pair_head: Module,
        rel_transformer_head: Module,
        box_pair_predictor: Module,
        box_h_predictor: Module,
        box_o_predictor: Module,
        # Dataset properties
        num_classes: int,
        # Hyperparameters
        box_nms_thresh: float = 0.5,
        box_score_thresh: float = 0.2,
        fg_iou_thresh: float = 0.5,
        max_human: int = 15,
        max_object: int = 15,
        # Misc
        distributed: bool = False
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name
        self.backbone = backbone
        self.box_roi_pool = box_roi_pool
        self.box_pair_head = box_pair_head
        self.rel_transformer_head = rel_transformer_head
        
        self.box_pair_predictor = box_pair_predictor
        
        self.box_h_predictor = box_h_predictor
        self.box_o_predictor = box_o_predictor

        self.num_classes = num_classes

        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.max_human = max_human
        self.max_object = max_object

        self.distributed = distributed

    def preprocess(self,
        targets: List[dict],
        append_gt: Optional[bool] = None
    ) -> None:

        if not self.training:
            self.max_human = 1000
            self.max_object = 1000

        results = []
        for target in targets:

            h_keep_idx = target['h_keep_idx']
            o_keep_idx = target['o_keep_idx']
            
            boxes = torch.cat(
                [
                    target["boxes_h"][h_keep_idx][:self.max_human], 
                    target["boxes_o"][o_keep_idx][:self.max_object]
                ]
            )
            
            labels = torch.cat(
                [
                    target['subject'][h_keep_idx][:self.max_human],
                    target["object"][o_keep_idx][:self.max_object],
                ]
            )

            indicators = torch.cat(
                [
                    torch.ones(len(h_keep_idx[:self.max_human]), device=labels.device),
                    torch.zeros(len(o_keep_idx[:self.max_object]), device=labels.device),
                ]
            )

            results.append(dict(
                boxes=boxes.view(-1, 4),
                labels=labels.view(-1),
                indicators=indicators.view(-1),
            ))
            
        return results
    

    def postprocess(self,
        indicators: List[Tensor],
        logits_h: Tensor,
        logits_o: Tensor,
        logits_p: Tensor,
        boxes_h: List[Tensor],
        boxes_o: List[Tensor],
        subject_labels: List[Tensor],
        object_labels: List[Tensor],
        labels: List[Tensor],
        filenames: List[Tensor],
        targets: List[dict]
    ) -> List[dict]:
        """
        Parameters:
        -----------
        logits_p: Tensor
            (N, K) Classification logits on each action for all box pairs
        logits_s: Tensor
            (N, 1) Logits for unary weights
        prior: List[Tensor]
            Prior scores organised by images. Each tensor has shape (2, M, K).
            M could be different for different images
        boxes_h: List[Tensor]
            Human bounding box coordinates organised by images (M, 4)
        boxes_o: List[Tensor]
            Object bounding box coordinates organised by images (M, 4)
        object_classes: List[Tensor]
            Object indices for each pair organised by images (M,)
        labels: List[Tensor]
            Binary labels on each action organised by images (M, K)

        Returns:
        --------
        results: List[dict]
            Results organised by images, with keys as below
            `boxes_h`: Tensor[M, 4]
            `boxes_o`: Tensor[M, 4]
            `index`: Tensor[L]
                Expanded indices of box pairs for each predicted action
            `prediction`: Tensor[L]
                Expanded indices of predicted actions
            `scores`: Tensor[L]
                Scores for each predicted action
            `object`: Tensor[M]
                Object indices for each pair
            `prior`: Tensor[2, L]
                Prior scores for expanded pairs
            `weights`: Tensor[M]
                Unary weights for each box pair
            `labels`: Tensor[L], optional
                Binary labels on each action
            `unary_labels`: Tensor[M], optional
                Labels for the unary weights
        """
        num_boxes = [len(b) for b in boxes_h]
        
        scores_p = logits_p
        scores_h = logits_h
        scores_o = logits_o
        
        scores_p = scores_p.split(num_boxes)
        scores_h = scores_h.split(num_boxes)
        scores_o = scores_o.split(num_boxes)
        
        if len(labels) == 0:
            labels = [None for _ in range(len(num_boxes))]

        results = []
        for sh, so, sp, b_h, b_o, l_h, l_o, l, fname, tgt in zip(
            scores_h, scores_o, scores_p, boxes_h, boxes_o, subject_labels, object_labels, labels, filenames, targets
        ):
       
            result_dict = dict(
                boxes_h=b_h, boxes_o=b_o,
                scores_h=sh, scores_o=so, scores_p=sp,
                labels_h=l_h, labels_o=l_o, fname=fname
            )
        
            # If binary labels are provided
            if l is not None:
                result_dict['image'] = tgt['transformed_image']
                result_dict['labels_p'] = l

            if not self.training:
                if tgt is not None:
                    result_dict['gt_boxes_h'] = tgt['boxes_h']
                    result_dict['gt_boxes_o'] = tgt['boxes_o']
                    result_dict['gt_labels_h'] = tgt['subject']
                    result_dict['gt_labels_o'] = tgt['object']
                    result_dict['gt_labels_p'] = tgt['labels']

            results.append(result_dict)

        return results

    def forward(self,
        features: OrderedDict,
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[dict]] = None,
        images = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        image_shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        targets: List[dict], optional
            Interaction targets with the following keys
            `boxes_h`: Tensor[G, 4]
            `boxes_o`: Tensor[G, 4]
            `object`: Tensor[G]
                Object class indices for each pair
            `labels`: Tensor[G]
                Target class indices for each pair

        Returns:
        --------
        results: List[dict]
            Results organised by images. During training the loss dict is appended to the
            end of the list, resulting in the length being larger than the number of images
            by one. For the result dict of each image, refer to `postprocess` for documentation.
            The loss dict has two keys
            `hoi_loss`: Tensor
                Loss for HOI classification
            `interactiveness_loss`: Tensor
                Loss incurred on learned unary weights
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        detections = self.preprocess(targets)

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        indicators = [detection['indicators'] for detection in detections]
        
        if self.backbone_name == "resnet50":
            box_features = self.box_roi_pool(features, box_coords, image_shapes)
 
        elif self.backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            box_features, global_features = self.backbone.encode_image(images, box_coords, box_segs=None, need_patch=False)
            features['global'] = global_features
        
        (box_features, box_h_features, box_o_features, 
         box_pair_features, boxes_h, boxes_o,
         subject_labels, object_labels, 
         box_pair_labels, filenames) = self.box_pair_head(
            features, image_shapes, box_features,
            box_coords, box_labels, targets,
            indicators
        )
        
        box_h_features = torch.cat(box_h_features)
        box_o_features = torch.cat(box_o_features)
        box_pair_features = torch.cat(box_pair_features)
        
        if self.rel_transformer_head is not None:
            concat_feat = torch.cat((box_h_features, box_o_features, box_pair_features), dim=1)
            
            if len(concat_feat) < 100:
                # padding
                num_feat, feat_size = concat_feat.shape
                target_tensor = torch.zeros(100, feat_size).to(box_features.device)
                target_tensor[:num_feat, :] = concat_feat
                all_unique_objects = target_tensor
            else:
                all_unique_objects = concat_feat[:100, :]

            all_unique_objects = all_unique_objects.unsqueeze(0)
        
            box_h_features, box_o_features, box_pair_features = self.rel_transformer_head(concat_feat, box_h_features, box_o_features, all_unique_objects)
        
        logits_h = self.box_h_predictor(box_h_features)
        
        if self.box_o_predictor is None:
            logits_o = self.box_h_predictor(box_o_features)
        else:
            logits_o = self.box_o_predictor(box_o_features)
        
        logits_p = self.box_pair_predictor(box_pair_features)
     
        results = self.postprocess(
            indicators,
            logits_h, logits_o,
            logits_p, 
            boxes_h, boxes_o, 
            subject_labels, 
            object_labels,
            box_pair_labels,
            filenames,
            targets
        )

        return results

class MultiBranchFusion(Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        representation_size: int, cardinality: int
    ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(representation_size / cardinality)
        assert sub_repr_size * cardinality == representation_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, representation_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))


class MessageMBF(MultiBranchFusion):
    """
    MBF for the computation of anisotropic messages

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    node_type: str
        Nature of the sending node. Choose between `human` amd `object`
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int,
        spatial_size: int,
        representation_size: int,
        node_type: str,
        cardinality: int
    ) -> None:
        super().__init__(appearance_size, spatial_size, representation_size, cardinality)

        if node_type == 'human':
            self._forward_method = self._forward_human_nodes
        elif node_type == 'object':
            self._forward_method = self._forward_object_nodes
        else:
            raise ValueError("Unknown node type \"{}\"".format(node_type))

    def _forward_human_nodes(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n_h, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n, 1, 1)
                * fc_2(spatial).permute([1, 0, 2])
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)
    def _forward_object_nodes(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n_h, 1, 1)
                * fc_2(spatial)
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)

    def forward(self, *args) -> Tensor:
        return self._forward_method(*args)


class GraphHead(Module):
    """
    Graphical model head

    Parameters:
    -----------
    output_channels: int
        Number of output channels of the backbone
    roi_pool_size: int
        Spatial resolution of the pooled output
    node_encoding_size: int
        Size of the node embeddings
    num_cls: int
        Number of targe classes
    human_idx: int
        The index of human/person class in all objects
    object_class_to_target_class: List[list]
        The mapping (potentially one-to-many) from objects to target classes
    fg_iou_thresh: float, default: 0.5
        The IoU threshold to identify a positive example
    num_iter: int, default 2
        Number of iterations of the message passing process
    """
    def __init__(self,
        backbone_name: str,
        out_channels: int,
        roi_pool_size: int,
        node_encoding_size: int, 
        representation_size: int, 
        num_cls: int, 
        num_obj_cls: int,
        fg_iou_thresh: float = 0.5,
        ignore_iou_thresh: float = 0.5,
        num_iter: int = 2
    ) -> None:

        super().__init__()

        self.backbone_name = backbone_name
        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.num_cls = num_cls
        self.num_obj_cls = num_obj_cls

        self.fg_iou_thresh = fg_iou_thresh
        self.ignore_iou_thresh = ignore_iou_thresh
        self.num_iter = num_iter

        if backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            self.box_head = nn.Sequential(nn.Linear(out_channels, node_encoding_size),
                                        nn.ReLU(),
                                        nn.Linear(node_encoding_size, node_encoding_size),
                                        nn.ReLU())
            
        else:
            # Box head to map RoI features to low dimensional
            self.box_head = nn.Sequential(
                Flatten(start_dim=1),
                nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
                nn.ReLU(),
                nn.Linear(node_encoding_size, node_encoding_size),
                nn.ReLU()
            )
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Compute adjacency matrix
        self.adjacency = nn.Linear(representation_size, 1)

        # Compute messages
        self.sub_to_obj = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='human',
            cardinality=16
        )
        self.obj_to_sub = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='object',
            cardinality=16
        )

        self.norm_h = nn.LayerNorm(node_encoding_size)
        self.norm_o = nn.LayerNorm(node_encoding_size)

        # Map spatial encodings to the same dimension as appearance features
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        # Spatial attention head
        self.attention_head = MultiBranchFusion(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )
        
        if backbone_name == 'resnet50':
            self.attention_head_g = MultiBranchFusion(
                256, 1024,
                representation_size, cardinality=16
            )
            
        elif backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            self.attention_head_g = MultiBranchFusion(
                out_channels, 1024,
                representation_size, cardinality=16
            )
            

    def associate_with_ground_truth(self,
        boxes_h: Tensor,
        boxes_o: Tensor,
        targets: List[dict]
    ) -> Tensor:
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_cls, device=boxes_h.device)
        
        pair_iou_min = torch.min(
            box_ops.box_iou(boxes_h, targets["boxes_h"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
            )

        # assign label with ground truth pair
        x, y = torch.nonzero(pair_iou_min >= self.fg_iou_thresh).unbind(1)
        
        # ignore pairs with iou threshold in [self.ignore_iou_thresh, self.fg_iou_thresh)
        x_ignore, _ = torch.nonzero(
            torch.logical_and(pair_iou_min < self.fg_iou_thresh, pair_iou_min >= self.ignore_iou_thresh)
            ).unbind(1)

        labels[x, targets["labels"][y]] = 1
        
        # add non interaction category
        non_interaction_x = torch.nonzero(labels.sum(dim=1) == 0).flatten()
        non_interaction_y = torch.zeros(len(non_interaction_x), device=boxes_h.device).long()
        labels[non_interaction_x, non_interaction_y] = 1
        
        if self.training:
            if len(non_interaction_x) > 50:
                ignore_non_interaction_x = non_interaction_x[50:]
                x_ignore = torch.cat((x_ignore, ignore_non_interaction_x), dim=0)
        # else:
        #     x_ignore = torch.cat((x_ignore, non_interaction_x), dim=0)
            
        
        y_ignore = torch.zeros(len(x_ignore), device=boxes_h.device).long()
        labels[x_ignore, y_ignore] = 0

        return labels

    def forward(self,
        features: OrderedDict, image_shapes: List[Tuple[int, int]],
        box_features: Tensor, box_coords: List[Tensor],
        box_labels: List[Tensor], 
        targets: Optional[List[dict]] = None,
        indicators: List[Tensor] = None
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor],
        List[Tensor], List[Tensor], List[Tensor]
    ]:
        """
        Parameters:
        -----------
            features: OrderedDict
                Feature maps returned by FPN
            box_features: Tensor
                (N, C, P, P) Pooled box features
            image_shapes: List[Tuple[int, int]]
                Image shapes, heights followed by widths
            box_coords: List[Tensor]
                Bounding box coordinates organised by images
            box_labels: List[Tensor]
                Bounding box object types organised by images
            targets: List[dict]
                Interaction targets with the following keys
                `boxes_h`: Tensor[G, 4]
                `boxes_o`: Tensor[G, 4]
                `labels`: Tensor[G]
            indicators: List[Tensor]
                indicators of subjects and objects

        Returns:
        --------
            all_box_pair_features: List[Tensor]
            all_boxes_h: List[Tensor]
            all_boxes_o: List[Tensor]
            all_object_class: List[Tensor]
            all_labels: List[Tensor]
            all_prior: List[Tensor]
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"

        if self.backbone_name == 'resnet50':
            global_features = self.avg_pool(features['3']).flatten(start_dim=1)
        elif self.backbone_name in ["CLIP_ViT_32", "CLIP_ViT_16", "CLIP_ViT_14"]:
            global_features = features['global']
        else:
            assert False, "Not supported backbone!"
            
        box_features = self.box_head(box_features)

        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]
        
        counter = 0
        all_boxes_h = []
        all_boxes_o = []
        all_subject_labels = []
        all_object_labels = []
        all_labels = []
        all_box_h_features = []
        all_box_o_features = []
        all_box_pair_features = []
        all_file_names = []
        
        for b_idx, (coords, labels, indicator) in enumerate(zip(box_coords, box_labels, indicators)):
            n = num_boxes[b_idx]
            device = box_features.device
    
            subject_idx = torch.nonzero(
                indicator > 0
            ).squeeze(1)
            
            n_h = len(subject_idx)
            n_o = n - n_h
            
            sbj_coords = coords[:n_h]
            obj_coords = coords[-n_o:]
            
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            # if n_h == 0 or n <= 1:
            #     all_box_pair_features.append(torch.zeros(
            #         0, 2 * self.representation_size,
            #         device=device)
            #     )
            #     all_boxes_h.append(torch.zeros(0, 4, device=device))
            #     all_boxes_o.append(torch.zeros(0, 4, device=device))
            #     all_subject_labels.append(torch.zeros(0, device=device, dtype=torch.int64))
            #     all_object_labels.append(torch.zeros(0, device=device, dtype=torch.int64))
            #     all_prior.append(torch.zeros(2, 0, self.num_cls, device=device))
            #     all_labels.append(torch.zeros(0, self.num_cls, device=device))
            #     continue
            # if not torch.all(labels[:n_h]==self.human_idx):
            #     raise ValueError("Human detections are not permuted to the top")

            node_encodings = box_features[counter: counter+n]
            h_node_encodings = node_encodings[:n_h]
            o_node_encodings = node_encodings[-n_o:]
            
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                    torch.arange(n_h, device=device),
                    torch.arange(n_o, device=device)
                )
            
            x_keep = x.flatten()
            y_keep = y.flatten()

            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten(); y = y.flatten()

            # Compute spatial features
            box_pair_spatial = compute_spatial_encodings(
                [sbj_coords[x]], [obj_coords[y]], [image_shapes[b_idx]]
            )
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            # Reshape the spatial features
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n_h, n_o, -1)

            adjacency_matrix = torch.ones(n_h, n_o, device=device)

            for _ in range(self.num_iter):
                # Compute weights of each edge
                weights = self.attention_head(
                    torch.cat([
                        h_node_encodings[x],
                        o_node_encodings[y]
                    ], 1),
                    box_pair_spatial
                )
                adjacency_matrix = self.adjacency(weights).reshape(n_h, n_o)

                # Update human nodes
                messages_to_h = F.relu(torch.sum(
                    adjacency_matrix.softmax(dim=1)[..., None] *
                    self.obj_to_sub(
                        o_node_encodings,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                h_node_encodings = self.norm_h(
                    h_node_encodings + messages_to_h
                )

                # Update object nodes (including human nodes)
                messages_to_o = F.relu(torch.sum(
                    adjacency_matrix.t().softmax(dim=1)[..., None] *
                    self.sub_to_obj(
                        h_node_encodings,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                o_node_encodings = self.norm_o(
                    o_node_encodings + messages_to_o
                )
 
            if targets is not None:
                # ignore pairs with iou threshold in [0.6, 1.0) using neg keep
                all_labels_onehot = self.associate_with_ground_truth(sbj_coords[x_keep], obj_coords[y_keep], targets[b_idx])
               
                neg_keep = torch.nonzero(all_labels_onehot.sum(dim=1) != 0).flatten()
                
                all_labels.append(
                    all_labels_onehot[neg_keep].max(dim=1)[1]
                )
                
                all_subject_labels.append(labels[:n_h][x_keep][neg_keep])
                all_object_labels.append(labels[-n_o:][y_keep][neg_keep])
                
                all_file_names.append(targets[b_idx]['id'])
            
            all_box_pair_features.append(torch.cat([
                self.attention_head(
                    torch.cat([
                        h_node_encodings[x_keep][neg_keep],
                        o_node_encodings[y_keep][neg_keep]
                        ], 1),
                    box_pair_spatial_reshaped[x_keep, y_keep][neg_keep]
                ), self.attention_head_g(
                    global_features[b_idx, None],
                    box_pair_spatial_reshaped[x_keep, y_keep][neg_keep]
                    )
            ], dim=1))
            
            all_box_h_features.append(h_node_encodings[x_keep][neg_keep])
            all_box_o_features.append(o_node_encodings[y_keep][neg_keep])
            
            all_boxes_h.append(sbj_coords[x_keep][neg_keep])
            all_boxes_o.append(obj_coords[y_keep][neg_keep])

            counter += n

        return box_features, all_box_h_features, all_box_o_features, \
            all_box_pair_features, all_boxes_h, all_boxes_o, \
            all_subject_labels, all_object_labels, \
            all_labels, all_file_names
