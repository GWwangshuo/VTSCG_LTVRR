"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import cv2
import numpy as np
import torchvision.ops.boxes as box_ops

from torch import Tensor
from typing import List, Tuple



def warpaffine_image(image, n_px, device):
    
    width, height = image.size
    image_size = torch.tensor([n_px, n_px]).to(device)
    x,y = torch.tensor(0).to(device), torch.tensor(0).to(device)
    w = torch.tensor(width-1).to(device)
    h = torch.tensor(height-1).to(device)

    aspect_ratio = image_size[0] / image_size[1]
    #center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    center = torch.tensor([x + w * 0.5, y + h * 0.5], dtype=torch.float32).to(device)
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    #scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    scale = torch.tensor([w / 200.0, h / 200.0], dtype=torch.float32).to(device)
    trans = get_warp_matrix(center * 2.0, image_size - 1.0, scale * 200.0, device)
    processed_img = cv2.warpAffine(
                        np.array(image),
                        np.array(trans.cpu()), (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR)
    img_meta = {'center': center, 'scale': scale, 'n_px': n_px}

    return processed_img, trans, img_meta


def get_warp_matrix(size_input, size_dst, size_target, device):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """

    matrix = torch.zeros((2,3), dtype=torch.float32).to(device)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]

    matrix[0, 0] = 1.0 * scale_x
    matrix[0, 1] = -0.0 * scale_x 

    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * 1.0 +
                               0.5 * size_input[1] * 0.0 +
                              0.5 * size_target[0])

    matrix[1, 0] = 0.0 * scale_y
    matrix[1, 1] = 1.0 * scale_y

    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * 0.0 -
                               0.5 * size_input[1] * 1.0 +
                               0.5 * size_target[1])
    return matrix



def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        np.ndarray[..., 2]: Result coordinate of joints.
    """
    #joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return torch.matmul(torch.cat((joints, joints[:, 0:1] * 0 +1), dim=1), mat.T).reshape(shape)



def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
        boxes_1: List[Tensor]
            First set of bounding boxes (M, 4)
        boxes_1: List[Tensor]
            Second set of bounding boxes (M, 4)
        shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        eps: float
            A small constant used for numerical stability

    Returns:
    --------
        Tensor
            Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)

def binary_focal_loss(
    x: Tensor, y: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-6
) -> Tensor:
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Parameters:
    -----------
        x: Tensor[N, K]
            Post-normalisation scores
        y: Tensor[N, K]
            Binary labels
        alpha: float
            Hyper-parameter that balances between postive and negative examples
        gamma: float
            Hyper-paramter suppresses well-classified examples
        reduction: str
            Reduction methods
        eps: float
            A small constant to avoid NaN values from 'PowBackward'

    Returns:
    --------
        loss: Tensor
            Computed loss tensor
    """
    loss = (1 - y - alpha).abs() * ((y-x).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy(
            x, y, reduction='none'
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))
