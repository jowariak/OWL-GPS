from functools import partial
#import mmcv
import random
import functools
from mmcv.runner import load_checkpoint
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Block
from typing import List
import numbers
from math import cos, pi
from typing import Callable, List, Optional, Union
import torch
import torch.nn as nn
import yaml
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from itertools import product
import numpy as np
from sklearn.metrics import confusion_matrix
from einops import rearrange
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as Fprint

from collections import OrderedDict
#import mmcv
import numpy as np
import torch
import logging
import time
#np.random.seed(421)
# NumPy
seed = 100
np.random.seed(seed)

# Python Random
random.seed(seed)

# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up logging
logging.basicConfig(filename='neur343.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
import torch
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.current_device())  # Currently active GPU

def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))



    if isinstance(label, str):
        #label = torch.from_numpy(
         #   mmcv.imread(label, flag='unchanged', backend='pillow'))
         label = torch.from_numpy(label)
    else:
        label = torch.from_numpy(label)

    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id



    
    # Create a mask where label is not equal to 2
    mask = (label != 2)
    
    # Apply the mask to filter pred_label and label
    pred_label = pred_label[mask]
    label = label[mask]
   

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean F-Score (mFscore)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
    
    

class Residual3DConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size, stride, padding),
            nn.BatchNorm3d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.residual_conv = nn.Conv3d(in_chans, out_chans, 1) if in_chans != out_chans else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual_conv(x)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _convTranspose2dOutput(
    input_size: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    output_padding: int,
):
    """
    Calculate the output size of a ConvTranspose2d.
    Taken from: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    return (
        (input_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )

class PatchEmbed(nn.Module):
    """ Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            num_frames=1,
            tubelet_size=1,
            in_chans=43,
            embed_dim=1024,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (num_frames // tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_size[0], patch_size[1]), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x




class TemporalViTEncoder(nn.Module):
    """Encoder from an ViT with capability to take in temporal input.

    This class defines an encoder taken from a ViT architecture.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 1,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 24,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        pretrained: str = None
    ):
        """

        Args:
            img_size (int, optional): Input image size. Defaults to 224.
            patch_size (int, optional): Patch size to be used by the transformer. Defaults to 16.
            num_frames (int, optional): Number of frames (temporal dimension) to be input to the encoder. Defaults to 1.
            tubelet_size (int, optional): Tubelet size used in patch embedding. Defaults to 1.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 1024.
            depth (int, optional): Encoder depth. Defaults to 24.
            num_heads (int, optional): Number of heads used in the encoder blocks. Defaults to 16.
            mlp_ratio (float, optional): Ratio to be used for the size of the MLP in encoder blocks. Defaults to 4.0.
            norm_layer (nn.Module, optional): Norm layer to be used. Defaults to nn.LayerNorm.
            norm_pix_loss (bool, optional): Whether to use Norm Pix Loss. Defaults to False.
            pretrained (str, optional): Path to pretrained encoder weights. Defaults to None.
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.num_frames = num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
       # self.multi_head_self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm_pix_loss = norm_pix_loss
        self.pretrained = pretrained

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if isinstance(self.pretrained, str):
            self.apply(self._init_weights)
            logging.info(f"load from {self.pretrained}")
            load_checkpoint(self, self.pretrained, strict=False, map_location="cpu")
            
        elif self.pretrained is None:
            # # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        
       
        x = self.patch_embed(x)
        

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return tuple([x])
        
        
class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """
    Neck that transforms the token-based output of transformer into a single embedding suitable for processing with standard layers.
    Performs 4 ConvTranspose2d operations on the rearranged input with kernel_size=2 and stride=2
    """

    def __init__(
        self,
        embed_dim: int,
        output_embed_dim: int,
        # num_frames: int = 1,
        Hp: int = 14,
        Wp: int = 14,
        drop_cls_token: bool = True,
    ):
        """

        Args:
            embed_dim (int): Input embedding dimension
            output_embed_dim (int): Output embedding dimension
            Hp (int, optional): Height (in patches) of embedding to be upscaled. Defaults to 14.
            Wp (int, optional): Width (in patches) of embedding to be upscaled. Defaults to 14.
            drop_cls_token (bool, optional): Whether there is a cls_token, which should be dropped. This assumes the cls token is the first token. Defaults to True.
        """
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = Hp
        self.W_out = Wp
        # self.num_frames = num_frames

        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        for _ in range(4):
            self.H_out = _convTranspose2dOutput(
                self.H_out, stride, padding, dilation, kernel_size, output_padding
            )
            self.W_out = _convTranspose2dOutput(
                self.W_out, stride, padding, dilation, kernel_size, output_padding
            )

        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )

    def forward(self, x):
        
        x = x[0]
        #print(x.shape)
        if self.drop_cls_token:
            x = x[:, 1:, :]
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1 , self.Hp, self.Wp)

        x = self.fpn1(x)
        x = self.fpn2(x)

        x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))

        out = tuple([x])

        return out
        
        
class Norm2d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
        


import torch.nn.functional as F




import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels
        self.norm_cfg=dict(type="BN", requires_grad=True)
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


encoder = TemporalViTEncoder(
            img_size=256,
    patch_size=16,
    num_frames=1,
    embed_dim=768,
    pretrained=None
        )
        




import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import glob
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import yaml
from tqdm import tqdm



        




def preprocess_image(image, means, stds):
    means = np.array(means).reshape(-1, 1, 1)
    stds = np.array(stds).reshape(-1, 1, 1)
    # normalize image
    normalized = image.copy()
    normalized = ((image - means) / stds)
    normalized = torch.from_numpy(normalized.reshape(normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized



class MultibandTiffDataset(Dataset):
    def __init__(self, image_directories, label_directories, lb, bands, data_mean, data_std, random_cropping=False):
        """
        Multiband TIFF Dataset with image paths, labels, and optional random cropping.

        Args:
            image_directories (list of str): Directories containing image files.
            label_directories (list of str): Directories containing corresponding label masks.
            bands (list of int): List of bands to extract from each image.
            data_mean (list of float): Mean values for normalization (one per band).
            data_std (list of float): Std dev values for normalization (one per band).
            random_cropping (bool): Whether to apply random cropping.
        """
        self.image_paths = []
        self.label_paths = []
        self.label_paths1 = []

        # Collect image and label file paths
        for image_dir, label_dir, label_dir1 in zip(image_directories, label_directories, lb):
            image_files = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
            for image_file in image_files:
                # Derive the corresponding label path
                label_file = os.path.join(label_dir, os.path.basename(image_file).replace('_merged.tif', '_mask.tif'))
                label_file1 = os.path.join(label_dir1, os.path.basename(image_file).replace('_merged.tif', '_mask.tif'))
                
                if os.path.exists(label_file1):  # Ensure label exists for the image
                    self.image_paths.append(image_file)
                    self.label_paths.append(label_file)
                    self.label_paths1.append(label_file1)
                    

        self.bands = bands
        self.data_mean = torch.tensor(data_mean, dtype=torch.float32)
        self.data_std = torch.tensor(data_std, dtype=torch.float32)
        self.random_cropping = random_cropping
        if self.random_cropping:
            self.random_crop = transforms.RandomCrop(224)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        label_path1 = self.label_paths1[idx]

        # Load image
        with rasterio.open(image_path) as src:
            img = np.stack([src.read(band) for band in self.bands], axis=0)  # Shape: (num_bands, H, W)

        # Load label mask
        with rasterio.open(label_path) as src:
            label = src.read(1)  # Assuming the label mask is a single-band TIFF (H, W)
        
        with rasterio.open(label_path1) as src:
            label1 = src.read(1)  # Assuming the label mask is a single-band TIFF (H, W)

        # Preprocess image
        img = preprocess_image(img, self.data_mean, self.data_std)
        
        #img = tuple(preprocess_image(band, self.data_mean, self.data_std) for band in img)

        # Random cropping (applied to both image and label)
        if self.random_cropping:
            crop = self.random_crop(torch.tensor(img))  # Apply random crop to image
            img, label = crop, self.random_crop(torch.tensor(label))

        # Return image, label, and image path
        return {
            'images': img,
            'label': torch.tensor(label, dtype=torch.long),
            'label1': torch.tensor(label1, dtype=torch.long),
            'image_paths': image_path
        }





train_image_dirs1 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_1"]
train_image_dirs2 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_2"]
train_image_dirs3 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_3"]
train_image_dirs4 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_4"]
train_image_dirs5 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_5"]
train_image_dirs6 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_6"]
train_image_dirs7 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_7"]
train_image_dirs8 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_8"]
train_image_dirs9 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_9"]
train_image_dirs10 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_10"]
train_image_dirs11 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_11"]
train_image_dirs12 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_12"]
train_image_dirs13 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_13"]
train_image_dirs14 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_14"]
train_image_dirs15 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_15"]
train_image_dirs16 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_16"]
train_image_dirs17 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_17"]
train_image_dirs18 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_18"]
train_image_dirs19 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_19"]
train_image_dirs20 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_20"]
train_image_dirs21 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_21"]
train_image_dirs22 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_22"]
train_image_dirs23 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_23"]
train_image_dirs24 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_24"]
train_image_dirs25 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_25"]
train_image_dirs26 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_26"]
train_image_dirs27 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_27"]
train_image_dirs28 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_28"]
train_image_dirs29 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_29"]
train_image_dirs30 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_30"]
train_image_dirs31 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_31"]
train_image_dirs32 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_32"]
train_image_dirs33 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_33"]
train_image_dirs34 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_34"]
train_image_dirs35 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_35"]
train_image_dirs36 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_36"]
train_image_dirs37 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_37"]
train_image_dirs38 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_38"]
train_image_dirs39 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_39"]
train_image_dirs40 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_40"]
train_image_dirs41 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_41"]
train_image_dirs42 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_42"]
train_image_dirs43 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_43"]
train_image_dirs44 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_44"]
train_image_dirs45 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/band_45"]




train_label_dirs = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/masks"]
train_label1_dirs = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/train_latest_new_256n/updated_masks_sparse"]



test_image_dirs1 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_1"]
test_image_dirs2 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_2"]
test_image_dirs3 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_3"]
test_image_dirs4 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_4"]
test_image_dirs5 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_5"]
test_image_dirs6 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_6"]
test_image_dirs7 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_7"]
test_image_dirs8 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_8"]
test_image_dirs9 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_9"]
test_image_dirs10 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_10"]
test_image_dirs11 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_11"]
test_image_dirs12 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_12"]
test_image_dirs13 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_13"]
test_image_dirs14 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_14"]
test_image_dirs15 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_15"]
test_image_dirs16 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_16"]
test_image_dirs17 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_17"]
test_image_dirs18 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_18"]
test_image_dirs19 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_19"]
test_image_dirs20 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_20"]
test_image_dirs21 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_21"]
test_image_dirs22 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_22"]
test_image_dirs23 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_23"]
test_image_dirs24 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_24"]
test_image_dirs25 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_25"]
test_image_dirs26 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_26"]
test_image_dirs27 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_27"]
test_image_dirs28 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_28"]
test_image_dirs29 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_29"]
test_image_dirs30 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_30"]
test_image_dirs31 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_31"]
test_image_dirs32 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_32"]
test_image_dirs33 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_33"]
test_image_dirs34 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_34"]
test_image_dirs35 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_35"]
test_image_dirs36 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_36"]
test_image_dirs37 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_37"]
test_image_dirs38 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_38"]
test_image_dirs39 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_39"]
test_image_dirs40 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_40"]
test_image_dirs41 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_41"]
test_image_dirs42 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_42"]
test_image_dirs43 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_43"]
test_image_dirs44 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_44"]
test_image_dirs45 = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/band_45"]




test_label_dirs = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/masks"]
test_label1_dirs = ["/nfs/turbo/train_combined_nsra/train_latest_water_elev_2019_new_filtered/test_latest_new_256n/updated_masks_sparse"]




data_mean1 = [56.69645316384055],
data_std1 = [26.2330500704307]

data_mean2 = [2531.34789375087],
data_std2 = [2791.9198815614836]

data_mean3 = [4928.593316058732],
data_std3 = [5759.562549486662]

data_mean4 = [3644.8906218797597],
data_std4 = [2874.1674343220598]

data_mean5 = [1657.285808553724],
data_std5 = [1643.7351312871856]

data_mean6 = [8000.301350726498],
data_std6 = [8716.18631357973]

data_mean7 = [1112.7808418709355],
data_std7 = [1130.431893825277]

data_mean8 = [3365.3761542606585],
data_std8 = [3611.742070670478]

data_mean9 = [1759.049240982931],
data_std9 = [1794.366404197245]

data_mean10 = [950.5207645543347],
data_std10 = [1058.7264247529574]

data_mean11 = [4330.122080129108],
data_std11 = [5047.569028544736]

data_mean12 = [5631.455037528524],
data_std12 = [5145.99569095791]

data_mean13 = [1768.6456173421175],
data_std13 = [1790.5764810981814]

data_mean14 = [4425.639290395274],
data_std14 = [4095.554762689101]

data_mean15 = [3357.74766017886],
data_std15 = [5157.178838902773]

data_mean16 = [1133.431735954465],
data_std16 = [1134.9741138735517]

data_mean17 = [2046.0259741424202],
data_std17 = [2743.419287860129]

data_mean18 = [1757.1552920053666],
data_std18 = [1794.9100825645605]

data_mean19 = [272.1845398827822],
data_std19 = [222.38146597838113]

data_mean20 = [1952.4514555955566],
data_std20 = [1453.0487543230004]

data_mean21 = [1738.6345073496357],
data_std21 = [1153.762448146885]

data_mean22 = [1856.3540331292802],
data_std22 = [2029.2569889330437]

data_mean23 = [603.156958980655],
data_std23 = [752.9620481768502]

data_mean24 = [1664.93586414233],
data_std24 = [1375.2462071567916]

data_mean25 = [753.6451236223812],
data_std25 = [732.6188750189423]

data_mean26 = [3887.043763536925],
data_std26 = [3296.7154762048226]

data_mean27 = [1989.92755639595],
data_std27 = [1729.4434716204003]

data_mean28 = [554.9850951083951],
data_std28 = [627.0356783412022]

data_mean29 = [8408.438590436977],
data_std29 = [4813.091027643458]

data_mean30 = [2992.565692608203],
data_std30 = [2903.6172090605924]

data_mean31 = [1054.5985629467536],
data_std31 = [1275.065108658805]

data_mean32 = [3877.5199011540517],
data_std32 = [3376.267841247456]

data_mean33 = [4635.183071694014],
data_std33 = [3229.585717542962]

data_mean34 = [6223.50207820429],
data_std34 = [5125.612037698973]

data_mean35 = [9351.760481826448],
data_std35 = [9051.840412747693]

data_mean36 = [6816.581408141371],
data_std36 = [6554.639344540255]

data_mean37 = [9640.437677204096],
data_std37 = [8248.276346346365]

data_mean38 = [5363.289585851493],
data_std38 = [3958.3951775654527]

data_mean39 = [7790.014835174909],
data_std39 = [6314.738552916341]

data_mean40 = [7536.195878599414],
data_std40 = [6041.563392709451]

data_mean41 = [8617.65864586381],
data_std41 = [6372.735738688901]

data_mean42 = [31049.824015690967],
data_std42 = [19874.963951566206]

data_mean43 = [44338.37224493198],
data_std43 = [21052.40534488174]

data_mean44 = [28.941783072731713],
data_std44 = [40.068691659443076]

data_mean45 = [2.3372645984996447],
data_std45 = [0.4727760453941362]


      





bands = [1]


train_dataset1 = MultibandTiffDataset(
    image_directories=train_image_dirs1,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean1,
    data_std=data_std1,
    random_cropping=False
)

train_dataset2 = MultibandTiffDataset(
    image_directories=train_image_dirs2,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean2,
    data_std=data_std2,
    random_cropping=False
)

train_dataset3 = MultibandTiffDataset(
    image_directories=train_image_dirs3,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean3,
    data_std=data_std3,
    random_cropping=False
)

train_dataset4 = MultibandTiffDataset(
    image_directories=train_image_dirs4,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean4,
    data_std=data_std4,
    random_cropping=False
)

train_dataset5 = MultibandTiffDataset(
    image_directories=train_image_dirs5,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean5,
    data_std=data_std5,
    random_cropping=False
)

train_dataset6 = MultibandTiffDataset(
    image_directories=train_image_dirs6,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean6,
    data_std=data_std6,
    random_cropping=False
)

train_dataset7 = MultibandTiffDataset(
    image_directories=train_image_dirs7,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean7,
    data_std=data_std7,
    random_cropping=False
)

train_dataset8 = MultibandTiffDataset(
    image_directories=train_image_dirs8,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean8,
    data_std=data_std8,
    random_cropping=False
)

train_dataset9 = MultibandTiffDataset(
    image_directories=train_image_dirs9,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean9,
    data_std=data_std9,
    random_cropping=False
)

train_dataset10 = MultibandTiffDataset(
    image_directories=train_image_dirs10,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean10,
    data_std=data_std10,
    random_cropping=False
)

train_dataset11 = MultibandTiffDataset(
    image_directories=train_image_dirs11,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean11,
    data_std=data_std11,
    random_cropping=False
)

train_dataset12 = MultibandTiffDataset(
    image_directories=train_image_dirs12,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean12,
    data_std=data_std12,
    random_cropping=False
)

train_dataset13 = MultibandTiffDataset(
    image_directories=train_image_dirs13,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean13,
    data_std=data_std13,
    random_cropping=False
)

train_dataset14 = MultibandTiffDataset(
    image_directories=train_image_dirs14,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean14,
    data_std=data_std14,
    random_cropping=False
)

train_dataset15 = MultibandTiffDataset(
    image_directories=train_image_dirs15,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean15,
    data_std=data_std15,
    random_cropping=False
)

train_dataset16 = MultibandTiffDataset(
    image_directories=train_image_dirs16,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean16,
    data_std=data_std16,
    random_cropping=False
)

train_dataset17 = MultibandTiffDataset(
    image_directories=train_image_dirs17,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean17,
    data_std=data_std17,
    random_cropping=False
)

train_dataset18 = MultibandTiffDataset(
    image_directories=train_image_dirs18,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean18,
    data_std=data_std18,
    random_cropping=False
)

train_dataset19 = MultibandTiffDataset(
    image_directories=train_image_dirs19,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean19,
    data_std=data_std19,
    random_cropping=False
)

train_dataset20 = MultibandTiffDataset(
    image_directories=train_image_dirs20,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean20,
    data_std=data_std20,
    random_cropping=False
)

train_dataset21 = MultibandTiffDataset(
    image_directories=train_image_dirs21,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean21,
    data_std=data_std21,
    random_cropping=False
)

train_dataset22 = MultibandTiffDataset(
    image_directories=train_image_dirs22,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean22,
    data_std=data_std22,
    random_cropping=False
)

train_dataset23 = MultibandTiffDataset(
    image_directories=train_image_dirs23,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean23,
    data_std=data_std23,
    random_cropping=False
)

train_dataset24 = MultibandTiffDataset(
    image_directories=train_image_dirs24,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean24,
    data_std=data_std24,
    random_cropping=False
)

train_dataset25 = MultibandTiffDataset(
    image_directories=train_image_dirs25,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean25,
    data_std=data_std25,
    random_cropping=False
)

train_dataset26 = MultibandTiffDataset(
    image_directories=train_image_dirs26,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean26,
    data_std=data_std26,
    random_cropping=False
)

train_dataset27 = MultibandTiffDataset(
    image_directories=train_image_dirs27,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean27,
    data_std=data_std27,
    random_cropping=False
)

train_dataset28 = MultibandTiffDataset(
    image_directories=train_image_dirs28,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean28,
    data_std=data_std28,
    random_cropping=False
)

train_dataset29 = MultibandTiffDataset(
    image_directories=train_image_dirs29,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean29,
    data_std=data_std29,
    random_cropping=False
)

train_dataset30 = MultibandTiffDataset(
    image_directories=train_image_dirs30,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean30,
    data_std=data_std30,
    random_cropping=False
)

train_dataset31 = MultibandTiffDataset(
    image_directories=train_image_dirs31,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean31,
    data_std=data_std31,
    random_cropping=False
)

train_dataset32 = MultibandTiffDataset(
    image_directories=train_image_dirs32,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean32,
    data_std=data_std32,
    random_cropping=False
)

train_dataset33 = MultibandTiffDataset(
    image_directories=train_image_dirs33,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean33,
    data_std=data_std33,
    random_cropping=False
)

train_dataset34 = MultibandTiffDataset(
    image_directories=train_image_dirs34,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean34,
    data_std=data_std34,
    random_cropping=False
)

train_dataset35 = MultibandTiffDataset(
    image_directories=train_image_dirs35,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean35,
    data_std=data_std35,
    random_cropping=False
)

train_dataset36 = MultibandTiffDataset(
    image_directories=train_image_dirs36,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean36,
    data_std=data_std36,
    random_cropping=False
)

train_dataset37 = MultibandTiffDataset(
    image_directories=train_image_dirs37,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean37,
    data_std=data_std37,
    random_cropping=False
)

train_dataset38 = MultibandTiffDataset(
    image_directories=train_image_dirs38,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean38,
    data_std=data_std38,
    random_cropping=False
)

train_dataset39 = MultibandTiffDataset(
    image_directories=train_image_dirs39,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean39,
    data_std=data_std39,
    random_cropping=False
)

train_dataset40 = MultibandTiffDataset(
    image_directories=train_image_dirs40,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean40,
    data_std=data_std40,
    random_cropping=False
)

train_dataset41 = MultibandTiffDataset(
    image_directories=train_image_dirs41,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean41,
    data_std=data_std41,
    random_cropping=False
)

train_dataset42 = MultibandTiffDataset(
    image_directories=train_image_dirs42,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean42,
    data_std=data_std42,
    random_cropping=False
)

train_dataset43 = MultibandTiffDataset(
    image_directories=train_image_dirs43,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean43,
    data_std=data_std43,
    random_cropping=False
)

train_dataset44 = MultibandTiffDataset(
    image_directories=train_image_dirs44,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean44,
    data_std=data_std44,
    random_cropping=False
)

train_dataset45 = MultibandTiffDataset(
    image_directories=train_image_dirs45,
    label_directories=train_label_dirs,
    lb=train_label1_dirs,
    bands=bands,
    data_mean=data_mean45,
    data_std=data_std45,
    random_cropping=False
)


# Testing datasets for all 45 bands
test_dataset1 = MultibandTiffDataset(
    image_directories=test_image_dirs1,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean1,
    data_std=data_std1,
    random_cropping=False
)

test_dataset2 = MultibandTiffDataset(
    image_directories=test_image_dirs2,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean2,
    data_std=data_std2,
    random_cropping=False
)

test_dataset3 = MultibandTiffDataset(
    image_directories=test_image_dirs3,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean3,
    data_std=data_std3,
    random_cropping=False
)

test_dataset4 = MultibandTiffDataset(
    image_directories=test_image_dirs4,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean4,
    data_std=data_std4,
    random_cropping=False
)

test_dataset5 = MultibandTiffDataset(
    image_directories=test_image_dirs5,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean5,
    data_std=data_std5,
    random_cropping=False
)

test_dataset6 = MultibandTiffDataset(
    image_directories=test_image_dirs6,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean6,
    data_std=data_std6,
    random_cropping=False
)

test_dataset7 = MultibandTiffDataset(
    image_directories=test_image_dirs7,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean7,
    data_std=data_std7,
    random_cropping=False
)

test_dataset8 = MultibandTiffDataset(
    image_directories=test_image_dirs8,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean8,
    data_std=data_std8,
    random_cropping=False
)

test_dataset9 = MultibandTiffDataset(
    image_directories=test_image_dirs9,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean9,
    data_std=data_std9,
    random_cropping=False
)

test_dataset10 = MultibandTiffDataset(
    image_directories=test_image_dirs10,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean10,
    data_std=data_std10,
    random_cropping=False
)

test_dataset11 = MultibandTiffDataset(
    image_directories=test_image_dirs11,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean11,
    data_std=data_std11,
    random_cropping=False
)

test_dataset12 = MultibandTiffDataset(
    image_directories=test_image_dirs12,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean12,
    data_std=data_std12,
    random_cropping=False
)

test_dataset13 = MultibandTiffDataset(
    image_directories=test_image_dirs13,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean13,
    data_std=data_std13,
    random_cropping=False
)

test_dataset14 = MultibandTiffDataset(
    image_directories=test_image_dirs14,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean14,
    data_std=data_std14,
    random_cropping=False
)

test_dataset15 = MultibandTiffDataset(
    image_directories=test_image_dirs15,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean15,
    data_std=data_std15,
    random_cropping=False
)

test_dataset16 = MultibandTiffDataset(
    image_directories=test_image_dirs16,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean16,
    data_std=data_std16,
    random_cropping=False
)

test_dataset17 = MultibandTiffDataset(
    image_directories=test_image_dirs17,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean17,
    data_std=data_std17,
    random_cropping=False
)

test_dataset18 = MultibandTiffDataset(
    image_directories=test_image_dirs18,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean18,
    data_std=data_std18,
    random_cropping=False
)

test_dataset19 = MultibandTiffDataset(
    image_directories=test_image_dirs19,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean19,
    data_std=data_std19,
    random_cropping=False
)

test_dataset20 = MultibandTiffDataset(
    image_directories=test_image_dirs20,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean20,
    data_std=data_std20,
    random_cropping=False
)

test_dataset21 = MultibandTiffDataset(
    image_directories=test_image_dirs21,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean21,
    data_std=data_std21,
    random_cropping=False
)

test_dataset22 = MultibandTiffDataset(
    image_directories=test_image_dirs22,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean22,
    data_std=data_std22,
    random_cropping=False
)

test_dataset23 = MultibandTiffDataset(
    image_directories=test_image_dirs23,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean23,
    data_std=data_std23,
    random_cropping=False
)

test_dataset24 = MultibandTiffDataset(
    image_directories=test_image_dirs24,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean24,
    data_std=data_std24,
    random_cropping=False
)

test_dataset25 = MultibandTiffDataset(
    image_directories=test_image_dirs25,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean25,
    data_std=data_std25,
    random_cropping=False
)

test_dataset26 = MultibandTiffDataset(
    image_directories=test_image_dirs26,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean26,
    data_std=data_std26,
    random_cropping=False
)

test_dataset27 = MultibandTiffDataset(
    image_directories=test_image_dirs27,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean27,
    data_std=data_std27,
    random_cropping=False
)

test_dataset28 = MultibandTiffDataset(
    image_directories=test_image_dirs28,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean28,
    data_std=data_std28,
    random_cropping=False
)

test_dataset29 = MultibandTiffDataset(
    image_directories=test_image_dirs29,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean29,
    data_std=data_std29,
    random_cropping=False
)

test_dataset30 = MultibandTiffDataset(
    image_directories=test_image_dirs30,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean30,
    data_std=data_std30,
    random_cropping=False
)

test_dataset31 = MultibandTiffDataset(
    image_directories=test_image_dirs31,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean31,
    data_std=data_std31,
    random_cropping=False
)

test_dataset32 = MultibandTiffDataset(
    image_directories=test_image_dirs32,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean32,
    data_std=data_std32,
    random_cropping=False
)

test_dataset33 = MultibandTiffDataset(
    image_directories=test_image_dirs33,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean33,
    data_std=data_std33,
    random_cropping=False
)

test_dataset34 = MultibandTiffDataset(
    image_directories=test_image_dirs34,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean34,
    data_std=data_std34,
    random_cropping=False
)

test_dataset35 = MultibandTiffDataset(
    image_directories=test_image_dirs35,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean35,
    data_std=data_std35,
    random_cropping=False
)

test_dataset36 = MultibandTiffDataset(
    image_directories=test_image_dirs36,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean36,
    data_std=data_std36,
    random_cropping=False
)

test_dataset37 = MultibandTiffDataset(
    image_directories=test_image_dirs37,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean37,
    data_std=data_std37,
    random_cropping=False
)

test_dataset38 = MultibandTiffDataset(
    image_directories=test_image_dirs38,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean38,
    data_std=data_std38,
    random_cropping=False
)

test_dataset39 = MultibandTiffDataset(
    image_directories=test_image_dirs39,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean39,
    data_std=data_std39,
    random_cropping=False
)

test_dataset40 = MultibandTiffDataset(
    image_directories=test_image_dirs40,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean40,
    data_std=data_std40,
    random_cropping=False
)

test_dataset41 = MultibandTiffDataset(
    image_directories=test_image_dirs41,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean41,
    data_std=data_std41,
    random_cropping=False
)

test_dataset42 = MultibandTiffDataset(
    image_directories=test_image_dirs42,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean42,
    data_std=data_std42,
    random_cropping=False
)

test_dataset43 = MultibandTiffDataset(
    image_directories=test_image_dirs43,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean43,
    data_std=data_std43,
    random_cropping=False
)

test_dataset44 = MultibandTiffDataset(
    image_directories=test_image_dirs44,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean44,
    data_std=data_std44,
    random_cropping=False
)

test_dataset45 = MultibandTiffDataset(
    image_directories=test_image_dirs45,
    label_directories=test_label_dirs,
    lb=test_label1_dirs,
    bands=bands,
    data_mean=data_mean45,
    data_std=data_std45,
    random_cropping=False
)

observed_datasets = [
    train_dataset1, train_dataset2, train_dataset3, train_dataset4, train_dataset5,
    train_dataset6, train_dataset7, train_dataset8, train_dataset9, train_dataset10,
    train_dataset11, train_dataset12, train_dataset13, train_dataset14, train_dataset15,
    train_dataset16, train_dataset17, train_dataset18, train_dataset19, train_dataset20,
    train_dataset21, train_dataset22, train_dataset23, train_dataset24, train_dataset25,
    train_dataset26, train_dataset27, train_dataset28, train_dataset29, train_dataset30,
    train_dataset31, train_dataset32, train_dataset33, train_dataset34, train_dataset35,
    train_dataset36, train_dataset37, train_dataset38, train_dataset39, train_dataset40,
    train_dataset41, train_dataset42, train_dataset43, train_dataset44, train_dataset45
]

observed_test_datasets = [
    test_dataset1, test_dataset2, test_dataset3, test_dataset4, test_dataset5,
    test_dataset6, test_dataset7, test_dataset8, test_dataset9, test_dataset10,
    test_dataset11, test_dataset12, test_dataset13, test_dataset14, test_dataset15,
    test_dataset16, test_dataset17, test_dataset18, test_dataset19, test_dataset20,
    test_dataset21, test_dataset22, test_dataset23, test_dataset24, test_dataset25,
    test_dataset26, test_dataset27, test_dataset28, test_dataset29, test_dataset30,
    test_dataset31, test_dataset32, test_dataset33, test_dataset34, test_dataset35,
    test_dataset36, test_dataset37, test_dataset38, test_dataset39, test_dataset40,
    test_dataset41, test_dataset42, test_dataset43, test_dataset44, test_dataset45
]



# DataLoader parameters
batch_size = 4
num_workers = 7




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import os


def evaluate_segmentation(predictions, labels, num_classes=3, ignore_index=-1, beta=1):
    """
    Evaluates segmentation predictions using IoU, F1 score, accuracy, precision, and recall.
    
    Args:
        predictions: List of predicted segmentation maps or tensors.
        labels: List of ground truth segmentation maps or tensors.
        num_classes: Number of classes (default is 3).
        ignore_index: Index to ignore in evaluation (e.g., background class).
        beta: Beta value for F-score calculation (default 1 for F1 score).
    
    Returns:
        Dictionary containing the evaluation metrics: IoU, F1 score, accuracy, precision, and recall.
    """
    # Convert predictions and labels into a suitable format
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    #predictions = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in predictions]
    #labels = [l.cpu().numpy() if isinstance(l, torch.Tensor) else l for l in labels]

    # Calculate mean IoU
    iou_result = mean_iou(
        results=predictions,
        gt_seg_maps=labels,
        num_classes=num_classes,
        ignore_index=ignore_index
    )

    # Calculate mean F-score
    fscore_result = mean_fscore(
        results=predictions,
        gt_seg_maps=labels,
        num_classes=num_classes,
        ignore_index=ignore_index,
        beta=beta
    )

    # Assemble the results into a dictionary
    eval_metrics = {
        'iou': iou_result['IoU'],  # Mean IoU for each class
        'fscore': fscore_result['Fscore'],  # F1 score for each class
        'precision': fscore_result['Precision'],  # Precision (macro average)
        'recall': fscore_result['Recall'],  # Recall (macro average)
        'aAcc': fscore_result['aAcc']
    }

    return eval_metrics


import numbers
from math import cos, pi
from typing import Callable, List, Optional, Union

import mmcv
from mmcv import runner




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Observed and unlabeled dataset initialization
observed_indices = np.random.choice(len(train_dataset1), 1, replace=False)
unlabeled_indices = np.setdiff1d(np.arange(len(train_dataset1)), observed_indices)





import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

observed_loader1 = DataLoader(
    Subset(train_dataset1, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader2 = DataLoader(
    Subset(train_dataset2, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader3 = DataLoader(
    Subset(train_dataset3, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader4 = DataLoader(
    Subset(train_dataset4, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader5 = DataLoader(
    Subset(train_dataset5, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader6 = DataLoader(
    Subset(train_dataset6, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader7 = DataLoader(
    Subset(train_dataset7, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader8 = DataLoader(
    Subset(train_dataset8, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader9 = DataLoader(
    Subset(train_dataset9, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader10 = DataLoader(
    Subset(train_dataset10, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader11 = DataLoader(
    Subset(train_dataset11, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader12 = DataLoader(
    Subset(train_dataset12, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader13 = DataLoader(
    Subset(train_dataset13, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader14 = DataLoader(
    Subset(train_dataset14, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader15 = DataLoader(
    Subset(train_dataset15, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader16 = DataLoader(
    Subset(train_dataset16, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader17 = DataLoader(
    Subset(train_dataset17, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader18 = DataLoader(
    Subset(train_dataset18, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader19 = DataLoader(
    Subset(train_dataset19, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader20 = DataLoader(
    Subset(train_dataset20, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader21 = DataLoader(
    Subset(train_dataset21, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader22 = DataLoader(
    Subset(train_dataset22, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader23 = DataLoader(
    Subset(train_dataset23, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader24 = DataLoader(
    Subset(train_dataset24, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader25 = DataLoader(
    Subset(train_dataset25, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader26 = DataLoader(
    Subset(train_dataset26, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader27 = DataLoader(
    Subset(train_dataset27, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader28 = DataLoader(
    Subset(train_dataset28, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader29 = DataLoader(
    Subset(train_dataset29, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader30 = DataLoader(
    Subset(train_dataset30, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader31 = DataLoader(
    Subset(train_dataset31, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader32 = DataLoader(
    Subset(train_dataset32, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader33 = DataLoader(
    Subset(train_dataset33, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader34 = DataLoader(
    Subset(train_dataset34, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader35 = DataLoader(
    Subset(train_dataset35, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader36 = DataLoader(
    Subset(train_dataset36, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader37 = DataLoader(
    Subset(train_dataset37, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader38 = DataLoader(
    Subset(train_dataset38, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader39 = DataLoader(
    Subset(train_dataset39, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader40 = DataLoader(
    Subset(train_dataset40, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader41 = DataLoader(
    Subset(train_dataset41, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader42 = DataLoader(
    Subset(train_dataset42, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader43 = DataLoader(
    Subset(train_dataset43, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader44 = DataLoader(
    Subset(train_dataset44, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)

observed_loader45 = DataLoader(
    Subset(train_dataset45, observed_indices),
    batch_size=4,
    shuffle=True,
    num_workers=4
)









import torch
import torch.nn as nn
import torch.nn.functional as F



class RelevanceEncoder2(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        """
        Relevance Encoder that predicts:
        - mu: Mean relevance vector (B, 45), which assigns importance to each of the 45 encoders.
        - sigma: Standard deviation vector (B, 45), which represents uncertainty for each encoder.

        Arguments:
        - embed_dim: Number of channels in each input embedding (e.g., 768).
        - hidden_dim: Number of hidden channels for convolutional layers.
        """
        super(RelevanceEncoder2, self).__init__()

        # 2D Convolutional feature extraction backbone (treating each encoder independently)
        self.conv1 = nn.Conv2d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        # Max Pooling (2D) to downsample spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for mu and sigma (output size is 1 for each encoder)
        self.mu_layer = nn.Linear(hidden_dim * 64 * 64, 1)  # Output (B, 1) after flattening the spatial dims
        self.sigma_layer = nn.Linear(hidden_dim * 64 * 64, 1)  # Output (B, 1)

    def forward(self, x):
        """
        Forward pass of the Relevance Encoder.

        Arguments:
        - x: Tensor of shape (B, 45, embed_dim, 256, 256), where:
          - B: Batch size
          - 45: Number of different encoders
          - embed_dim: Feature embedding size (e.g., 768)
          - 256x256: Spatial dimensions

        Returns:
        - mu: Mean relevance vector, shape (B, 45).
        - sigma: Standard deviation vector, shape (B, 45).
        """
        B, N, C, H, W = x.shape  # (B, 45, embed_dim, 256, 256)

        # Apply convolutions and pooling to each of the 45 encoders independently
        mu_list = []
        sigma_list = []

        for i in range(N):
            encoder_output = x[:, i, :, :, :]  # Select the i-th encoder's embeddings (shape: B, embed_dim, 256, 256)

            # Apply 2D convolution layers with activation
            encoder_output = F.relu(self.bn1(self.conv1(encoder_output)))
            encoder_output = F.relu(self.bn2(self.conv2(encoder_output)))
            encoder_output = F.relu(self.bn3(self.conv3(encoder_output)))

            # Max Pooling to reduce spatial dimensions (256x256 -> 128x128 -> 64x64)
            encoder_output = self.pool(encoder_output)  # Shape: (B, hidden_dim, 128, 128)
            encoder_output = self.pool(encoder_output)  # Shape: (B, hidden_dim, 64, 64)

            # Flatten the tensor before passing it to the fully connected layers
            encoder_output = torch.flatten(encoder_output, start_dim=1)  # Shape: (B, hidden_dim * 64 * 64)

            # Compute mu and sigma for this encoder
            mu = F.softplus(self.mu_layer(encoder_output))  # Shape: (B, 1)
            sigma = F.softplus(self.sigma_layer(encoder_output)) + 1e-6  # Shape: (B, 1), ensures sigma > 0

            mu_list.append(mu)
            sigma_list.append(sigma)

        # Stack the list of mu and sigma vectors across encoders (N=45) and take the mean across embeddings
        mu = torch.cat(mu_list, dim=1)  # Shape: (B, 45, 1)
        sigma = torch.cat(sigma_list, dim=1)  # Shape: (B, 45, 1)
        
        # Final shape should be (B, 45)
        #mu = mu.squeeze()  # Remove the singleton dimension to get shape (B, 45)
        #sigma = sigma.squeeze()  # Remove the singleton dimension to get shape (B, 45)

        return mu, sigma




    
def gram_schmidt_optimized(embeddings):
    """
    Applies Gram-Schmidt orthogonalization across encoder axis in a memory-efficient way.

    Args:
        embeddings: Tensor of shape (N, B, C, H, W)

    Returns:
        Tensor of shape (N, B, C, H, W), orthogonalized.
    """
    N, B, C, H, W = embeddings.shape
    embeddings = embeddings.to(dtype=torch.float32)  # Ensure precision for inner products

    orthogonal = torch.zeros_like(embeddings)

    for i in range(N):
        vi = embeddings[i]

        for j in range(i):
            vj = orthogonal[j]
            dot = (vi * vj).sum(dim=1, keepdim=True)  # Shape: (B, 1, H, W)
            norm = (vj * vj).sum(dim=1, keepdim=True).clamp(min=1e-6)
            proj = (dot / norm) * vj
            vi = vi - proj

        orthogonal[i] = vi

    return orthogonal.to(dtype=torch.float16)




    
    
def save_embeddings_npy(embeddings_stacked, npy_dir, cache_key):
    os.makedirs(npy_dir, exist_ok=True)
    npy_path = os.path.join(npy_dir, f"{cache_key}.npy")
    arr = embeddings_stacked.detach().cpu().to(torch.float16).numpy()
    np.save(npy_path, arr)  # No compression



def load_embeddings_npy(npy_dir, cache_key, device):
    npy_path = os.path.join(npy_dir, f"{cache_key}.npy")
    arr = np.load(npy_path, mmap_mode=None)  # Fully load into RAM
    return torch.from_numpy(arr).to(device), cache_key


class MultiEncoderFusionn(nn.Module):
    def __init__(self, encoder_class, pretrained_paths, relevance_encoder):
        super().__init__()
        assert len(pretrained_paths) == 45, "You must provide exactly 45 pretrained weight paths."

        self.encoders = nn.ModuleList([
            encoder_class(img_size=256, patch_size=16, num_frames=1, embed_dim=768, pretrained=path)
            for path in pretrained_paths
        ])
        for encoder in self.encoders:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

        self.neck = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=768, output_embed_dim=768, drop_cls_token=True, Hp=16, Wp=16
        )
        self.relevance_encoder = relevance_encoder

        self.embedding_cache_dict = {}  
        self.x = {}
        self.temp_cache = {}
        self.embedding_lru_cache = OrderedDict()
        self.max_cache_size = 100
        self.cache_dir = "/scratch/embs10_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
  


  
    

    def forward(self, image_batches, relevance_vector=None, cache_key=None, cache_key1 = None,check=None):
        embed = None
        assert len(image_batches) == 45, "You must provide exactly 45 batches of input images."
        device = image_batches[0].device

        # --- Lookahead / reuse embedding logic ---
        if cache_key1 is None:
            self.temp_cache.clear()  # Clear temp cache only when explicitly told
        else:
            if cache_key1 not in self.temp_cache:
                logging.info(f"[TempCache] Preloading: {cache_key1}")
                embedding1, _ = load_embeddings_npy(self.cache_dir, cache_key1, device)
                self.temp_cache[cache_key1] = embedding1
                embeddings_stacked = self.temp_cache[cache_key1]
        
        if cache_key is not None and cache_key in self.temp_cache:
            logging.info(f"[TempCache] Using cached embedding for: {cache_key}")
            embeddings_stacked = self.temp_cache[cache_key]
            x1 = ()

        
        
        
        if cache_key is not None and os.path.exists(os.path.join(self.cache_dir, f"{cache_key}.npy")) and cache_key1 is None:
            logging.info("Loading cached embeddings from")
          
            embeddings_stacked , _ = load_embeddings_npy(self.cache_dir, cache_key, device)
           
            x1 = ()
           
        elif cache_key1 is None:
            embeddings = []
            for i in range(45):
                self.encoders[i] = self.encoders[i].to(device)
                with torch.no_grad():
               
                    x = self.encoders[i](image_batches[i])
                    x = self.neck(x)
              
                    if isinstance(x, tuple):
                       fused_feat = x[0]
                    embeddings.append(fused_feat.cpu())
               
                    self.encoders[i] = self.encoders[i].cpu()
                    torch.cuda.empty_cache()
                    x1 = x[1:]
                    del fused_feat  # or any other tensor that's accumulating
                    import gc
                    gc.collect()


            torch.cuda.empty_cache()


            logging.info("Stacking embeddings...")
            
            embeddings_stacked = torch.stack(embeddings, dim=0)
            logging.info("Stacked. Moving to device...")
            embeddings_stacked = embeddings_stacked.to(device)
            logging.info("Moved to device.")

           
            embeddings_stacked = gram_schmidt_optimized(embeddings_stacked)
            logging.info("optimized")
            torch.cuda.empty_cache()
            embeddings_stacked = embeddings_stacked.permute(1, 0, 2, 3, 4)  # (B, 45, C, H, W)
            
            if cache_key is not None:
            
                save_embeddings_npy(embeddings_stacked, self.cache_dir, cache_key)
                
    
            
        #del x
        #gc.collect()
        #logging.info("Stacking embeddings...")
        #embeddings_stacked = torch.stack(embeddings, dim=0)
        #logging.info("Stacked. Moving to device...")
        #embeddings_stacked = embeddings_stacked.to(device)
        #logging.info("Moved to device.")

        #logging.info("stacked")
        # embeddings_stacked = embeddings_stacked.to(dtype=torch.float32, device="cpu")
        #embeddings_stacked = gram_schmidt_optimized(embeddings_stacked)
        #logging.info("optimized")
        #torch.cuda.empty_cache()
        #embeddings_stacked = embeddings_stacked.permute(1, 0, 2, 3, 4)  # (B, 45, C, H, W)
        # === Relevance vector generation ===
        if (relevance_vector is None) or (check == 1):
            with torch.no_grad(), torch.cuda.amp.autocast():
                logging.info("start none emb")
                mu, sigma = self.relevance_encoder(embeddings_stacked)
                logging.info("end none embed")
                epsilon = torch.randn_like(sigma)
                relevance_vector = mu + epsilon * sigma
                relevance_vector = torch.nn.functional.softmax(relevance_vector, dim=1)

        else:
            mu, sigma = None, None

        B = relevance_vector.shape[0]
        relevance_vector1 = relevance_vector.view(B, 45, 1, 1, 1)
        logging.info("start weighted")
        
        
        weighted_embeddings = embeddings_stacked * relevance_vector1
        
       
      
 
        fused_features = weighted_embeddings.sum(dim=1)
        x1 = ()
        fused_features = (fused_features,) + x1
        del embeddings_stacked
        torch.cuda.empty_cache()
   
        return fused_features, relevance_vector, mu, sigma


class head(nn.Module):
    def __init__(self, output_embed_dim = 768, num_classes = 1):
        """
        Takes fused features and processes them through two FCN heads.

        Arguments:
        - output_embed_dim: Number of feature channels in the fused feature map.
        - num_classes: Number of output classes for segmentation.
        """
        super(head, self).__init__()

        # First segmentation head
        self.head = FCNHead(
            in_channels=output_embed_dim,
            channels=256,
            num_classes=num_classes,
            num_convs=1,
            in_index=-1,
            concat_input=False,
            dropout_ratio=0.1,
            align_corners=False
        )

        # Second segmentation head
        self.head2 = FCNHead(
            in_channels=output_embed_dim,
            channels=256,
            num_classes=num_classes,
            num_convs=2,
            in_index=-1,
            concat_input=False,
            dropout_ratio=0.1,
            align_corners=False
        )

    def forward(self, fused_features):
        """
        Processes fused features through both segmentation heads.

        Arguments:
        - fused_features: Tensor of shape (B, output_embed_dim, 256, 256).

        Returns:
        - output1: Output from first segmentation head.
        - output2: Output from second segmentation head.
        """
        output1 = self.head(fused_features)  # Segmentation output from head 1
        output2 = self.head2(fused_features)  # Segmentation output from head 2

        return output1, output2

        
        
        
        
def sample_relevance_vectors(mu, sigma, num_samples=10):
    """
    Generate multiple samples from the relevance distribution.

    Arguments:
    - mu: Mean relevance vector, shape (batch_size, num_channels).
    - sigma: Standard deviation (uncertainty per feature), shape (num_channels,).
    - num_samples: Number of samples to draw.

    Returns:
    - sampled_relevance: A tensor of shape (num_samples, batch_size, num_channels)
                          containing sampled relevance vectors.
    """
    # Sample from standard normal distribution
    epsilon = torch.randn((num_samples, *mu.shape), device=mu.device)  # Shape: (num_samples, batch_size, num_channels)

    # Generate relevance samples: R = mu + epsilon * sigma
    sampled_relevance = mu.unsqueeze(0) + epsilon * sigma.unsqueeze(0)  # Broadcasting sigma
    
   
    sampled_relevance = torch.nn.functional.softmax(sampled_relevance, dim=2)
    


    return sampled_relevance  # Shape: (num_samples, batch_size, num_channels)        
        
        




import csv
import time as time_lib

import torch
import torch.nn.functional as F
import numpy as np
import gc

def active_query_selection(unlabeled_indices, classification_head, timee,total, device):
    future = None
    next_cache_key = None
    future_key = None
    
    """
    Active Learning Query Selection:
    - Computes relevance representations (mu, sigma) for observed (train) and unlabeled samples.
    - Computes average Euclidean distance of each unlabeled relevance vector to all trained ones.
    - Computes selection score using: distance + negative exponential of uncertainty.
    - Returns the most uncertain sample index.

    Arguments:
    - relevance_encoder: The trained relevance encoder (outputs mu, sigma).
    - observed_dataloader: DataLoader for labeled (train) dataset.
    - unlabeled_dataset: The dataset containing unlabeled samples.
    - classification_head: The final classification head for predictions.
    - masked_probs_dict: Dictionary storing model prediction probabilities for each sample.
    - device: Torch device.

    Returns:
    - best_index: Index of the most uncertain sample in `unlabeled_dataset`.
    """
    
    ### Step 1: Compute Relevance Vectors for TRAINED DATASET
    mu_train = []
    core_buffer = foml_trainer.core_buffer
    reservoir_buffer = foml_trainer.reservoir_buffer
    logging.info("both in eval start")
    relevance_encoder.eval()
    segmentation_decoder.eval()
    multiheadfusion.neck.eval()
    logging.info("both in eval end")
    if timee > 5 and core_buffer:
        
        with torch.no_grad():
            logging.info("entered first if")
            for rel_vec, _, _, _,_ ,_,_ in core_buffer:
                mu_train.append(rel_vec.to(device))
            if reservoir_buffer:
                reservoir_sample_count = min(3, len(reservoir_buffer))
            
                sampled_reservoir = random.sample(reservoir_buffer, reservoir_sample_count)
                for rel_vec, _, _, _,_,_,_ in sampled_reservoir:
                    mu_train.append(rel_vec.to(device))
                    
    else:
        with torch.no_grad():
            for batch_group in zip(*observed_dataloaders):
                images_list = [batch['images'].to(device) for batch in batch_group]
                logging.info("entered else before multihead")
                _, _, mu_train1, _ = multiheadfusion(images_list)
                logging.info("end of multihead")
                mu_train.append(mu_train1)

    mu_train = torch.cat(mu_train, dim=0) if mu_train else torch.empty(0)
    
    
 


        
    query_scores = []
    all_rel_samples = [] 
    allmu = []
    allsigma = []
    #relevance_encoder.eval()
    #segmentation_deocder.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
      
      for i, idx in enumerate(unlabeled_indices):
    
            logging.info("start with unlabeled indices train")
            cache_key = f"sample_{idx}"
            images_list = []
            for dataset in observed_datasets:
                sample = dataset[idx]
                image = sample['images'].unsqueeze(0).to(device)
                images_list.append(image)

            labels = sample['label1'].unsqueeze(0).to(device)
            
           
            logging.info("start with getting the mus and sigmas from miltiheadfusion")
            
            

               
                
            fused_feat, rel, mu_unlabeled, sigma = multiheadfusion(images_list, cache_key = cache_key)
          
            allmu.append(mu_unlabeled)
            allsigma.append(sigma)
            all_rel_samples.append(rel)
            query_score_list = []
            qs_list = []
            
    

            # --- Step 1: Distance to all training mu
            logging.info("start with getting the distances")
            distance = torch.cdist(rel, mu_train, p=2).mean().item()
        
            # --- Step 2: Prediction uncertainty
            logging.info("start with getting the pred uncer")
          
            logits,_ = segmentation_decoder(fused_feat)
            #probs = F.softmax(logits, dim=1)
            probs = torch.sigmoid(logits)
            non_2_mask = (labels != 2).expand_as(probs).float()
            probs_masked = probs * non_2_mask
        
           
            valid_pixel_count = non_2_mask.sum().clamp(min=1.0)
            overall_prob = (probs_masked.sum() / valid_pixel_count).item()

          
            
            prediction_uncertainty = abs(0.5 - overall_prob)
        
            query_score = distance * np.exp(-prediction_uncertainty)
           
            final_score = query_score
            query_scores.append((final_score, idx, rel, mu_unlabeled, sigma))
            
            

   
    query_scores.sort(key=lambda x: x[0], reverse=True)
    top10 = query_scores[:10]
    
    if(timee%20 != 0):
        # Check if any unlabeled index still has class 0
        dataset = observed_datasets[0]
        zero_mask_candidate = None
        for idx in unlabeled_indices:
            #for dataset in observed_datasets:
                sample = dataset[idx]
                mask = sample['label1']
                if (mask == 0).any():
                    zero_mask_candidate = idx
                    break
                if zero_mask_candidate is not None:
                    break
        
        
        
        # Check if any of the top 10 already has a 0-mask
        has_zero_mask = False
        for _, idx, _, _, _ in top10:
            #for dataset in observed_datasets:
                sample = observed_datasets[0][idx]
                mask = sample['label1']
                if (mask == 0).any():
                    has_zero_mask = True
                    break
                if has_zero_mask:
                    break
        
        # If not, forcibly include one sample that has 0 (if exists)
        if not has_zero_mask and zero_mask_candidate is not None:
            for i in range(len(query_scores)):
                if query_scores[i][1] == zero_mask_candidate:
                    forced_sample = query_scores[i]
                    break
            # Replace the lowest scoring sample in top10
            top10[-1] = forced_sample
            
        # Ensure only one sample with class 0 exists in top10
        zero_mask_count = sum(1 for _, idx, _, _, _ in top10 if (observed_datasets[0][idx]['label1'] == 0).any())
        if zero_mask_count > 1:
            # If there are multiple 0 samples, remove all but one
            zero_samples = [i for i, (_, idx, _, _, _) in enumerate(top10) if (observed_datasets[0][idx]['label1'] == 0).any()]
            for i in range(1, len(zero_samples)):
                top10.pop(zero_samples[i])
                for j in range(10, len(query_scores)):  # Start from index 10 (lower-ranking)
                    alt_idx = query_scores[j][1]
                    alt_sample = dataset[alt_idx]
                    alt_mask = alt_sample['label1']
                    if not (alt_mask == 0).any():  # Ensure it's class 1
                        top10.append(query_scores[j])  # Add the replacement to top10
                        break
            
    else:
        
        dataset = observed_datasets[0]
        for i in range(len(top10)):
            idx = top10[i][1]
            sample = dataset[idx]
            mask = sample['label1']
            if (mask == 0).any():
                # Found one with class 0, try to replace it
                for j in range(10, len(query_scores)):
                    alt_idx = query_scores[j][1]
                    alt_sample = dataset[alt_idx]
                    alt_mask = alt_sample['label1']
                    if not (alt_mask == 0).any():
                        top10[i] = query_scores[j]
                        break
          #      break  # Only r
    
    
    # Final outputs
    best_indices = [x[1] for x in top10]
    best_rels = [x[2] for x in top10]
    mus = [x[3] for x in top10]
    sigmas = [x[4] for x in top10]
    
    logging.info(f"[QUERY] Selected new sample indices: {best_indices}")
    
    zero_mask_count = 0
    for idx in best_indices:
        sample = observed_datasets[0][idx]  # assuming consistent labels across encoders
        label = sample['label1']
        if (label == 0).any():
            zero_mask_count += 1
  
    
    
    
    
    return best_indices, best_rels, mus, sigmas, zero_mask_count

    
   

          
import re      
 

class CustomLoss1(nn.Module):
    def __init__(self, loss_name='loss_custom1', class_weights=None):
        super(CustomLoss1, self).__init__()
        self._loss_name = loss_name
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights else None

    def generate_noise_mask_path(self, image_path):
        path = re.sub(r"/band_\d+/", "/noise_masks_dirdownaaaa_whole/", image_path)
    	return path.replace("_merged.tif", "_mask.tif")
        
    def load_noise_mask(self, file_path):
        with rasterio.open(file_path) as src:
            noise_mask_array = src.read(1)
            noise_mask_tensor = torch.from_numpy(noise_mask_array).float()
        return noise_mask_tensor
        
    def forward(self, predictions, label_mask, image_paths, **kwargs):
        """
        predictions: [B, 1, H, W] — logits for class 1
        label_mask: [B, H, W] — 0 for class 0, 1 for class 1, 2 for ignore (bg)
        """
        device = predictions.device
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)

        batch_size = predictions.shape[0]
        noise_masks = []

        for i in range(batch_size):
            noise_mask_path = self.generate_noise_mask_path(image_paths[i])
            noise_mask = self.load_noise_mask(noise_mask_path)
            noise_masks.append(noise_mask)

        noise_masks = torch.stack(noise_masks).to(device)  # [B, H, W]

        # Flatten everything for manual masking
        preds_flat = predictions.squeeze(1)  # [B, H, W]
        labels_flat = label_mask  # [B, H, W]

        # Only compute loss for pixels where label is 0 or 1 (ignore 2)
        valid_mask = (labels_flat != 2)

        preds_valid = preds_flat[valid_mask]           # [N]
        labels_valid = labels_flat[valid_mask].float() # [N]
        noise_valid = noise_masks[valid_mask]          # [N]
        
        # ---- Tweak: Modify noise_valid if label 0 exists and noise is 1.0
        if (labels_valid == 0).any():
            # Change all 1.0s in noise_valid to something else, e.g., 0.8
            noise_valid = torch.where(noise_valid == 1.0, torch.tensor(0.031, device=device), noise_valid)

        # BCEWithLogitsLoss (elementwise)
        bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        bce_loss = bce_loss_fn(preds_valid, labels_valid)  # [N]

        pt = torch.exp(-bce_loss)
        # Class weighting: alpha for class 1, (1 - alpha) for class 0
        if self.class_weights is not None:
            alpha = self.class_weights[1]
        else:
            alpha = 0.25  # default if not provided
        
        alpha_t = alpha * labels_valid + (1 - alpha) * (1 - labels_valid)  # [N]
        focal_loss = alpha_t * (1 - pt) ** 2 * bce_loss  # weighted focal loss
        #focal_loss = (1 - pt) ** 2 * bce_loss  # Focal-like weighting

        weighted_loss = focal_loss * noise_valid
        loss = torch.mean(weighted_loss)

        return loss
        


# Example class weights (adjust as needed)
class_weights = [1.0, 0.50]  # Modify based on your dataset's class distribution

# Instantiate the custom loss function
criterion = CustomLoss1(class_weights=class_weights)
    






def select_representatives_intersectionss(thetas_dict, k, epsilon, d):
    n = len(thetas_dict)
    coverage_sets = []

    for dim in range(d):
        sorted_thetas = sorted(thetas_dict.items(), key=lambda item: item[1][dim])
        coverage_list = [[theta[0]] for theta in sorted_thetas]
        coverage_set = [set(coverage_list[0])]

        for i in range(1, n):
            index_i = coverage_list[i][0]
            for j in range(i - 1, -1, -1):
                index_j = coverage_list[j][0]
                if thetas_dict[index_i][dim] <= thetas_dict[index_j][dim] + 2 * epsilon:
                #if abs(thetas_dict[index_i][dim] - thetas_dict[index_j][dim]) <= epsilon:
                    coverage_list[i].append(index_j)
                else:
                    break
            if coverage_set[-1] <= set(coverage_list[i]):
                coverage_set[-1] = set(coverage_list[i])
            else:
                coverage_set.append(set(coverage_list[i]))

        coverage_sets.append(coverage_set)

    # Find the best coverage across dimensions
    max_coverage_size = 0
    best_combo = None

    for combination in product(*coverage_sets):
        merge_set = functools.reduce(lambda x, y: x.intersection(y), combination)
        if max_coverage_size<len(merge_set):
            max_coverage_size=len(merge_set)
        
            covered_points=(combination,merge_set)
    #    if len(merge_set) > max_coverage_size:
     #       max_coverage_size = len(merge_set)
      #      best_combo = combination

    #if best_combo is None:
     #   return np.zeros(d), set()  # fallback if no coverage found

    representative = np.zeros(d)
    for i in range(d):
        r_coordinate= np.average([thetas_dict[theta][i] for theta in covered_points[0][i]])
        representative[i]=r_coordinate
    #print(representative,covered_points[1])


    return representative,covered_points[1]

        #indices = best_combo[i]
        #if indices:
         #   representative[i] = np.mean([thetas_dict[idx][i] for idx in indices])
        #else:
         #   representative[i] = 0  # fallback

    #return representative, functools.reduce(lambda x, y: x.intersection(y), best_combo)



import numpy as np
import functools
from itertools import product

def select_representive_committeee(thetas, k, epsilon, n=100):
    committee = []
    covered_points = []

    # Convert torch tensor or np arrays to list of tuples (for hashability)
    if isinstance(thetas, torch.Tensor):
        thetas = thetas.cpu().numpy()

    thetas = [tuple(t.tolist()) if isinstance(t, np.ndarray) else tuple(t) for t in thetas]
    thetas = list(dict.fromkeys(thetas))  # remove duplicates while preserving order

    # Create a dictionary for indexing
    thetas_dict = {i: thetas[i] for i in range(len(thetas))}
    d = len(thetas[0])  # dimensionality

    for _ in range(k):
        if not thetas_dict:
            break  # All points are covered
        rep, covered = select_representatives_intersectionss(thetas_dict, 1, epsilon, d)
        logging.info(f"Points covered in this iteration: {len(covered)}")
        committee.append(rep)
        covered_points.append(covered)
        for key in covered:
            thetas_dict.pop(key, None)  # safe pop
            
    # Now, assign uncovered points to the nearest representative
    uncovered_points = set(thetas_dict.keys())
    if uncovered_points:
        logging.info(f"Remaining uncovered points: {len(uncovered_points)}")

        # Assign uncovered points to the nearest representative (from the existing 4 clusters)
        for point_idx in uncovered_points:
            # Calculate the distances to each of the 4 representatives
            distances = [np.linalg.norm(np.array(thetas[point_idx]) - np.array(rep)) for rep in committee]
            closest_cluster_idx = np.argmin(distances)  # Find the closest cluster
            covered_points[closest_cluster_idx].add(point_idx)  # Assign to the closest cluster
            thetas_dict.pop(point_idx, None)  # Remove assigned point from dictionary

    # Log the number of points in each cluster at the end
    logging.info("Cluster sizes after assignment:")
    for idx, cluster in enumerate(covered_points):
        logging.info(f"Cluster {idx + 1}: {len(cluster)} points")
    return committee, covered_points      
    # Ensure full coverage: check if there are any uncovered points and cover them
    #uncovered_points = set(thetas_dict.keys())
    #while uncovered_points:
     #   logging.info(f"Additional points to cover: {len(uncovered_points)}")
      #  rep, covered = select_representatives_intersectionss(thetas_dict, 1, epsilon, d)
       # committee.append(rep)
        #covered_points.append(covered)
        #for key in covered:
         #   thetas_dict.pop(key, None)  # safe pop
        #uncovered_points = set(thetas_dict.keys())  # Update uncovered points

    #return committee, covered_points

   


def poly_lr_with_warmup(base_lr, curr_step, max_step, power=1.0,
                        min_lr=0.0, warmup_iters=7, warmup_ratio=0.1):
    if curr_step < warmup_iters:
        warmup_factor = warmup_ratio + (1 - warmup_ratio) * curr_step / warmup_iters
        return base_lr * warmup_factor
    else:
        coeff = (1 - (curr_step - warmup_iters) / (max_step - warmup_iters)) ** power
        return (base_lr - min_lr) * coeff + min_lr


from collections import Counter
from sklearn.cluster import DBSCAN

import pandas as pd

class FOMLTrainer:
    def __init__(self,
                 meta_lr=1e-3, online_lr=1e-3, beta_1=0.01,
                 k=4, epsilon=0.3,
                 core_buffer_size=40, reservoir_buffer_size=60):
        #self.relevance_encoder = relevance_encoder
        #self.segmentation_decoder = segmentation_decoder
        self.class0_usage_counter = Counter()  # Tracks how often each 0-sample (by cache_key) is used
        self.online_optimizer = torch.optim.Adam(
            list(relevance_encoder.parameters()) + list(segmentation_decoder.parameters())+list(multiheadfusion.neck.parameters()),
            lr=online_lr
        )
        self.meta_optimizer = torch.optim.Adam(
            list(relevance_encoder.parameters()) + list(segmentation_decoder.parameters()) +  list(multiheadfusion.neck.parameters()),
            lr=meta_lr
        )
        
        self.initial_online_lr = online_lr
        self.initial_meta_lr = meta_lr
        
        self.current_online_step = 0
        self.current_meta_step = 0
        
        # Estimate total steps or set a large number if unsure
        self.total_steps = 2000  # adjust based on your AL rounds or epochs

        self.core_buffer = []  
        self.reservoir_buffer = [] 
        self.core_buffer_size = core_buffer_size
        self.reservoir_buffer_size = reservoir_buffer_size
        self.total_seen = 0  # For reservoir sampling

        self.k = k
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.clusters = []
        self.cluster_members = []
        self.theta_params = None 


        
    
    def update_clusters(self):
        relevance_encoder.eval()
        segmentation_decoder.eval()
        multiheadfusion.neck.eval()
        mu_values = []
        sigma_values = []
        if len(self.core_buffer) < 1:
            self.clusters = []
            self.cluster_members = []
            return

        # Refresh all relevance vectors in core buffer
        refreshed_core = []
        #logging.info("start of core refresh")
        for _, image_list, mask, paths, cache_key,l,c in self.core_buffer:
            _,_,new_vec,sigma = multiheadfusion(image_list, cache_key = cache_key)
            mu_values.append(new_vec.detach().cpu().numpy())
            sigma_values.append(sigma.detach().cpu().numpy())
            new_vec = new_vec.detach().cpu()
            
            refreshed_core.append((new_vec, image_list, mask, paths, cache_key,l,c))
        self.core_buffer = refreshed_core
        # Reshaping to 2D
        # Convert list to numpy array
        mu_values_np = np.array(mu_values)
        mu_values_reshaped = mu_values_np.squeeze(1) 
        sigma_values_np = np.array(sigma_values)
        sigma_values_reshaped = sigma_values_np.squeeze(1) 
        
   
       

        # Refresh all relevance vectors in reservoir buffer
        refreshed_reservoir = []
        #logging.info("start of reservoir refresh")
        for _, image_list, mask, paths, cache_key,l,c in self.reservoir_buffer:
            _,_,new_vec,_ = multiheadfusion(image_list, cache_key = cache_key)
            new_vec = new_vec.detach().cpu()
            
            #logging.info(f"mu shape: {new_vec.shape}")
            refreshed_reservoir.append((new_vec, image_list, mask, paths, cache_key,l,c))
        self.reservoir_buffer = refreshed_reservoir

        #relevance_matrix = torch.stack([entry[0] for entry in self.core_buffer])
        
        ##logging.info("matrix")
        relevance_matrix = torch.cat([entry[0] for entry in self.core_buffer], dim=0)
        effective_k = min(len(self.core_buffer), self.k)
        logging.info("end of matrix")

        logging.info(f"Relevance matrix shape: {relevance_matrix.shape}")
        logging.info(f"Effective_k: {effective_k}")
        logging.info(f"Epsilon: {self.epsilon}")
        pca = PCA(n_components=7)
        reduced_relevance_matrix = pca.fit_transform(relevance_matrix)
    
        # Log the PCA transformation details
        logging.info(f"Relevance matrix shape after PCA: {reduced_relevance_matrix.shape}")
        logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        committee, covered_points = select_representive_committeee(
            reduced_relevance_matrix, k=effective_k, epsilon=self.epsilon)
        
        #logging.info("end of committee")
        self.clusters = committee
        self.cluster_members = covered_points
        


    def update_online(self, image_list, mask, image_paths, relevance_vector,mu,sigma, counter, cache_key = None):
        logging.info("update_online: Start")
        relevance_encoder.train()
        segmentation_decoder.train()
        multiheadfusion.neck.train()
        self.online_optimizer.zero_grad()

        # Compute relevance vector
        #relevance_vector = self.relevance_encoder(image_list).detach().cpu()

        
        
        # Increment lifetime for all existing core and reservoir samples
        for i in range(len(self.core_buffer)):
            self.core_buffer[i] = self.core_buffer[i][:-2] + (self.core_buffer[i][-2] + 1, self.core_buffer[i][-1])  # increment lifetime
        for i in range(len(self.reservoir_buffer)):
            self.reservoir_buffer[i] = self.reservoir_buffer[i][:-2] + (self.reservoir_buffer[i][-2] + 1, self.reservoir_buffer[i][-1])
        
        new_sample = (mu, [img.detach() for img in image_list], mask.detach(), image_paths, cache_key, 1, 1)  # lifetime=1, counter=1
        self.core_buffer.append(new_sample)
        
        # Discard sample with lifetime >= 40
        discard_idx = next((i for i, s in enumerate(self.core_buffer) if s[-2] >= 40), None)
        if discard_idx is not None:
            discarded_sample = self.core_buffer.pop(discard_idx)
       
        else:
            discarded_sample = None
      

        #logging.info("end core buffer update")
        device = self.core_buffer[0][0].device  # use device of the first tensor
        #mu_vals = torch.cat([entry[0].to(device) for entry in self.core_buffer], dim=0)

        #logging.info(f"mu range: {mu_vals.min().item():.4f} to {mu_vals.max().item():.4f}")

        # --- Reservoir buffer update (randomized) ---
        if discarded_sample:
            self.total_seen += 1
            if len(self.reservoir_buffer) < self.reservoir_buffer_size:
                self.reservoir_buffer.append(discarded_sample)
            else:
                oldest_idx = max(range(len(self.reservoir_buffer)), key=lambda i: self.reservoir_buffer[i][5])
                self.reservoir_buffer.pop(oldest_idx)
                self.reservoir_buffer.append(discarded_sample)
                #replace_idx = random.randint(0, self.total_seen - 1)
                #if replace_idx < self.reservoir_buffer_size:
                 #   self.reservoir_buffer[replace_idx] = discarded_sample
     

        
        
         
        
        
        logging.info(f"[ONLINE] Updating with sample: {cache_key}")
        

    def update_meta(self, oimage_list, omask, oimage_paths, orelevance_vector,omu,osigma, ocounter, ocache_key = None ):
        self.update_clusters()
        logging.info(f"core bufer len: {len(self.core_buffer)}")
        logging.info(f"res bufer len: {len(self.reservoir_buffer)}")
        logging.info("start of meta")
        relevance_encoder.train()
        segmentation_decoder.train()
        multiheadfusion.neck.train()
        #if not self.core_buffer or not self.cluster_members:
         #   return
        if not self.core_buffer:
             return
        
        if (len(self.core_buffer) >= 30) and (ocounter%10==0):
            n_meta_steps = 5  # Or any number of meta update steps
        else:
            n_meta_steps = 1
    
        
        #selected_images, selected_masks = [], []
        
        selected_images, selected, selected_masks, selected_paths, selected_mu, selected_sigma = [], [], [],[],[],[]
        logging.info("start mem")
        # Keep track of how many samples we have selected
        num_selected = 0
        
        # While we have less than 5 selected samples
        logging.info(f"cluster size: {len(self.clusters)}")
        

        

        selected_cache_keys = set()  # Set to track selected cache keys
        remaining_points_needed = -10

         # Step 1: Check number of clusters
        num_clusters = len(self.clusters)
        
        # Ensure we select at least 4 points from clusters
        points_selected_from_clusters = 0
        
        if num_clusters < 4:
            # If less than 4 clusters, cycle through all and select one point per cluster until we get 4 points
            logging.info(f"Less than 4 clusters, selecting at least 4 points from clusters.")
            
            while points_selected_from_clusters < 4:
                for member_indices in self.cluster_members:
                    sorted_indices = sorted(member_indices, key=lambda idx: self.core_buffer[idx][-2] / self.core_buffer[idx][-1], reverse=True)
                    
                    for idx in sorted_indices:
                        cache_key = self.core_buffer[idx][4]  # Get the cache key
                        
                        if cache_key in selected_cache_keys:
                            continue  # Skip if already selected
        
                        # Retrieve the data associated with chosen_idx
                        mu, image_list, mask, paths, cache_keys, lifetime, counter = self.core_buffer[idx]
                        
                        # Perform fusion
                        fused, _, mu, sigma = multiheadfusion(torch.stack(image_list, dim=0), cache_key=cache_keys)
                        
                        # Append the selected point's data
                        selected_images.append(fused)
                        selected_mu.append(mu)
                        selected_sigma.append(sigma)
                        selected_masks.append(mask.squeeze() if mask.dim() == 3 else mask)
                        selected_paths.extend(paths)
                        
                        # Add the sample to selected
                        selected.append((mu, image_list, mask, paths, cache_keys, lifetime, counter + 1))
                        
                        # Mark the cache key as selected
                        selected_cache_keys.add(cache_key)
                        
                        # Increment the count of selected samples
                        points_selected_from_clusters += 1
                        
                        logging.info(f"Selected {mu} from cluster_members, total selected: {points_selected_from_clusters}")
                        
                        
                        break
                
                # If we have selected 4 points, stop the loop
                if points_selected_from_clusters >= 4:
                    break
        
        else:
            # If more than 4 clusters, select one point from each cluster
            logging.info(f"More than 4 clusters, selecting 1 point from each cluster.")
            
            for member_indices in self.cluster_members:
                sorted_indices = sorted(member_indices, key=lambda idx: self.core_buffer[idx][-2] / self.core_buffer[idx][-1], reverse=True)
                
                for idx in sorted_indices:
                    cache_key = self.core_buffer[idx][4]  # Get the cache key
                    
                    if cache_key in selected_cache_keys:
                        continue  # Skip if already selected
        
                    # Retrieve the data associated with chosen_idx
                    mu, image_list, mask, paths, cache_keys, lifetime, counter = self.core_buffer[idx]
                    
                    # Perform fusion
                    fused, _, mu, sigma = multiheadfusion(torch.stack(image_list, dim=0), cache_key=cache_keys)
                    
                    # Append the selected point's data
                    selected_images.append(fused)
                    selected_mu.append(mu)
                    selected_sigma.append(sigma)
                    selected_masks.append(mask.squeeze() if mask.dim() == 3 else mask)
                    selected_paths.extend(paths)
                    
                    # Add the sample to selected
                    selected.append((mu, image_list, mask, paths, cache_keys, lifetime, counter + 1))
                    
                    # Mark the cache key as selected
                    selected_cache_keys.add(cache_key)
                    
                    logging.info(f"Selected {mu} from cluster_members, total selected: {len(selected)}")
                    
                    # Only select 1 from each cluster, stop once we have 1 point per cluster
                    break  # Stop at the first point of each cluster
        
        # Step 2: Now that we've selected from the clusters, select the top 3 points from the reservoir buffer
        logging.info("Selecting top 3 points from the reservoir buffer based on lifetime/(counter+1)")
        
        if len(self.reservoir_buffer) > 0:
            # Sort reservoir buffer by highest lifetime/(counter+1) ratio
            sorted_reservoir = sorted(range(len(self.reservoir_buffer)),
                                      key=lambda idx: self.reservoir_buffer[idx][-2] / (self.reservoir_buffer[idx][-1] + 1),
                                      reverse=True)
            
            # Limit to the top 3 highest lifetime/(counter+1) points
            sorted_reservoir = sorted_reservoir[:3]
        
            #remaining_points_needed = 3 - len(sorted_reservoir)  # Calculate how many more we need from the reservoir
            #logging.info(f"Remaining points needed from reservoir: {remaining_points_needed}")
        
            selected_from_reservoir = 0
            for idx in sorted_reservoir:
              
        
                cache_key = self.reservoir_buffer[idx][4]  # Get the cache key
                
                # Skip if the cache key has already been selected
                if cache_key in selected_cache_keys:
                    continue  # Skip if already selected
        
                # Retrieve data for the point
                mu, image_list, mask, paths, cache_keys, lifetime, counter = self.reservoir_buffer[idx]
        
                # Perform fusion
                fused, _, mu, sigma = multiheadfusion(torch.stack(image_list, dim=0), cache_key=cache_keys)
        
                # Append the selected point's data
                selected_images.append(fused)
                selected_mu.append(mu)
                selected_sigma.append(sigma)
                selected_masks.append(mask.squeeze() if mask.dim() == 3 else mask)
                selected_paths.extend(paths)
        
                # Add the sample to selected
                selected.append((mu, image_list, mask, paths, cache_keys, lifetime, counter + 1))
        
                # Mark the cache key as selected
                selected_cache_keys.add(cache_key)
        
                logging.info(f"Selected {mu} from reservoir with cache key {cache_key}, total selected: {len(selected)}")
                selected_from_reservoir += 1
        
        # Step 3: If the reservoir does not provide enough points, select the remaining points from clusters
        while (len(selected)<7):
            logging.info(f"Reservoir buffer exhausted, selecting from cluster_members to reach 7 points.")
            
            
            for member_indices in self.cluster_members:
                #if remaining_points_needed!=-10:
                if len(selected)>=7:
                        break  # If we have enough points, stop
        
                sorted_indices = sorted(member_indices, key=lambda idx: self.core_buffer[idx][-2] / self.core_buffer[idx][-1], reverse=True)
                
                for idx in sorted_indices:
                    cache_key = self.core_buffer[idx][4]  # Get the cache key
                    
                    # Skip if the cache key has already been selected
                    if cache_key in selected_cache_keys:
                        continue  # Skip if already selected
        
                    # Retrieve the data associated with chosen_idx
                    mu, image_list, mask, paths, cache_keys, lifetime, counter = self.core_buffer[idx]
                    
                    # Perform fusion
                    fused, _, mu, sigma = multiheadfusion(torch.stack(image_list, dim=0), cache_key=cache_keys)
                    
                    # Append the selected point's data
                    selected_images.append(fused)
                    selected_mu.append(mu)
                    selected_sigma.append(sigma)
                    selected_masks.append(mask.squeeze() if mask.dim() == 3 else mask)
                    selected_paths.extend(paths)
                    
                    # Add the sample to selected
                    selected.append((mu, image_list, mask, paths, cache_keys, lifetime, counter + 1))
                    
                    # Mark the cache key as selected
                    selected_cache_keys.add(cache_key)
                    
                    logging.info(f"Selected {mu} from cluster_members, total selected: {len(selected)}")
                    
                    # Decrease remaining points needed
                    #if remaining_points_needed!=-10:
                     #   remaining_points_needed -= 1
                    break
        
        logging.info(f"Selection completed with {len(selected)} points.")

        
                
                        
               
        
        
        zero_sample = None
        z = None
        z2=None
        zero_sample2=None
        z3=None
        zero_sample3=None
        
        # Gather all 0-mask samples from both buffers
        zero_class_candidates = []
        for buffer in [self.core_buffer, self.reservoir_buffer]:
            for mu, image_list, mask, paths, cache_keysss, life, count in buffer:
                if (mask == 0).any():
                    zero_class_candidates.append((mu, image_list, mask, paths, cache_keysss, life, count))
        
        # Sort candidates by how often they've been used in meta update (from tracker)
        zero_class_candidates.sort(key=lambda x: self.class0_usage_counter[x[4]])  # x[4] = cache_key
        
        # Pick least used one
        if zero_class_candidates:
            mu, image_list, mask, paths, cache_keyssss, life, count = zero_class_candidates[0]
            fused, _, mu, sigma = multiheadfusion(torch.stack(image_list, dim=0), cache_key=cache_keyssss)
            zero_sample = (fused, mu, sigma, mask.squeeze() if mask.dim() == 3 else mask, paths)
            z = (mu, image_list, mask, paths, cache_keyssss, life, count + 1)
            if len(zero_class_candidates) > 1:
                mu1, image_list2, mask3, paths4, cache_keyssss5, life6, count7 = zero_class_candidates[1]
                fused1, _, mu2, sigma3 = multiheadfusion(torch.stack(image_list2, dim=0), cache_key=cache_keyssss5)
                zero_sample2 = (fused1, mu2, sigma3, mask3.squeeze() if mask3.dim() == 3 else mask3, paths4)
                z2 = (mu1, image_list2, mask3, paths4, cache_keyssss5, life6, count7 + 1)
            if len(zero_class_candidates) > 2:
                mu1, image_list2, mask3, paths4, cache_keyssss5, life6, count7 = zero_class_candidates[2]
                fused1, _, mu2, sigma3 = multiheadfusion(torch.stack(image_list2, dim=0), cache_key=cache_keyssss5)
                zero_sample3 = (fused1, mu2, sigma3, mask3.squeeze() if mask3.dim() == 3 else mask3, paths4)
                z3 = (mu1, image_list2, mask3, paths4, cache_keyssss5, life6, count7 + 1)
                
        #all_samples = self.core_buffer + self.reservoir_buffer
        # Compute ranking scores
        #ranked = sorted(all_samples, key=lambda s: s[-2] / s[-1], reverse=True)  # lifetime / counter
        #selected = ranked[:7]
        

      

        
        
        

        # Define a helper set of existing selected keys
        selected_keys = set(s[4] for s in selected)
        
        if zero_sample and z[4] not in selected_keys:
            fused, mu, sigma, mask, paths = zero_sample
            selected_images.append(fused)
            selected_mu.append(mu)
            selected_sigma.append(sigma)
            selected_masks.append(mask)
            selected_paths.extend(paths)
            selected.append(z)
            logging.info(f"injected a 0 to meta from key {z[4]}")
            self.class0_usage_counter[z[4]] += 1
        
        if zero_sample2 and z2[4] not in selected_keys:
            fused, mu, sigma, mask, paths = zero_sample2
            selected_images.append(fused)
            selected_mu.append(mu)
            selected_sigma.append(sigma)
            selected_masks.append(mask)
            selected_paths.extend(paths)
            selected.append(z2)
            logging.info(f"injected a 0 to meta from key {z2[4]}")
            self.class0_usage_counter[z2[4]] += 1
        
        if zero_sample3 and z3[4] not in selected_keys:
            fused, mu, sigma, mask, paths = zero_sample3
            selected_images.append(fused)
            selected_mu.append(mu)
            selected_sigma.append(sigma)
            selected_masks.append(mask)
            selected_paths.extend(paths)
            selected.append(z3)
            logging.info(f"injected a 0 to meta from key {z3[4]}")
            self.class0_usage_counter[z3[4]] += 1
        
        
        
        
        updated_by_key = {s[4]: s for s in selected}
        
        # Overwrite matching entries in core_buffer
        self.core_buffer = [
            updated_by_key[s[4]] if s[4] in updated_by_key else s
            for s in self.core_buffer
        ]
        
        # Same for reservoir_buffer
        self.reservoir_buffer = [
            updated_by_key[s[4]] if s[4] in updated_by_key else s
            for s in self.reservoir_buffer
        ]
        
        # --- Now log all final selections ---
        for i, s in enumerate(selected):
            score = s[-2] / s[-1]
            logging.info(f"[META RANKING] Selected #{i+1}: key={s[4]}, lifetime={s[-2]}, counter={s[-1]}, score={score:.4f}")
            



        for step in range(n_meta_steps):
            self.meta_optimizer.zero_grad()
            predicted_mask_list = []
            op2_list = []
            p = []
            op1 = []
            #logging.info("decoder")
            for fused, mask in zip(selected_images, selected_masks):
               pm, op = segmentation_decoder(fused)
               
               probs1 = torch.sigmoid(op)  # [B, 1, H, W]
               preds1 = (probs1 > 0.54).squeeze(1)  # [B, H, W]
               # Sigmoid + threshold
               probs = torch.sigmoid(pm)  # [B, 1, H, W]
               preds = (probs > 0.54).squeeze(1)  # [B, H, W]
               p.append(pm)
               op1.append(op)
               op2_list.append(preds1)
               predicted_mask_list.append(preds)
            #logging.info("decoder end")
            # Stack predictions and masks
            predicted_mask = torch.cat(predicted_mask_list, dim=0)
            op2 = torch.cat(op2_list, dim=0)
            p1 = torch.cat(p, dim=0)
            op11 = torch.cat(op1, dim=0)
            masks_val = torch.stack(selected_masks, dim=0).to(next(segmentation_decoder.parameters()).device)
            mu_all = torch.cat(selected_mu, dim=0)
            sigma_all = torch.cat(selected_sigma, dim=0)
            
            # KL divergence: encourage posterior to be close to prior
            kl_divergence = 0.5 * torch.sum(1 + torch.log(sigma_all**2) - sigma_all**2 - mu_all**2, dim=1).mean()
    
    
            #masks_val = torch.cat(selected_masks, dim=0).to(next(segmentation_decoder.parameters()).device)
            print("predicted_mask shape:", predicted_mask.shape)
            print("masks_val shape:", masks_val.shape)
    
    
    
    
    

            
            segmentation_loss = criterion(p1, masks_val , selected_paths)
            segmentation_loss1 = criterion(op11, masks_val , selected_paths)
            meta_loss1 = segmentation_loss + segmentation_loss1
            meta_loss = meta_loss1 + (0.01*kl_divergence)
            lr = poly_lr_with_warmup(
                base_lr=self.initial_meta_lr,
                curr_step=self.current_meta_step,
                max_step=97,
                power=2.0, min_lr=0.0,
                warmup_iters=10,
                warmup_ratio=0.1
            )
            for g in self.meta_optimizer.param_groups:
                g['lr'] = lr
            
            self.current_meta_step += 1
            logging.info(f"[META] Step {self.current_meta_step}, LR = {lr:.6f}")
    
            meta_loss.backward()
            self.meta_optimizer.step()
            #scheduler.step()
            self.theta_params = torch.cat([p.clone().detach().view(-1) for p in segmentation_decoder.parameters()]+[p.clone().detach().view(-1) for p in multiheadfusion.neck.parameters()] 
            + [p.clone().detach().view(-1) for p in relevance_encoder.parameters()])
            num_zeroes = (masks_val == 0).sum().item()
            num_ones = (masks_val == 1).sum().item()
            predicted_mask_labels = predicted_mask
            metrics = evaluate_segmentation(predicted_mask_labels, masks_val)
            
            logging.info(f"Evaluation Metrics: {metrics}")
            logging.info(f"num total actual zeroes: {num_zeroes}")
            logging.info(f"num total actual ones: {num_ones}")  
            # Convert predicted logits to labels
            pred_labels = predicted_mask.detach().cpu().numpy()
            true_labels = masks_val.detach().cpu().numpy()
            
            # Flatten all pixels into 1D arrays
            pred_flat = pred_labels.flatten()
            true_flat = true_labels.flatten()
            
    
            mask = true_flat != 2
            pred_flat = pred_flat[mask]
            true_flat = true_flat[mask]
            
            # Compute confusion matrix
            conf_mat = confusion_matrix(true_flat, pred_flat, labels=[0, 1])
            
            # Log it nicely
            logging.info("Confusion Matrix:")
            logging.info("\n" + str(conf_mat))
            #print("meta loss", meta_loss1.item())
            logging.info(f"meta loss: {meta_loss1.item()}")
            
            #return meta_loss.item()
            
            

        
        

pretrained_paths = [
    '/nfs/turbo/train_combined_nsra/model_weights_config_1_script1.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_2_script1.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_3_script1.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_4_script1.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_5_script1.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_6_script1.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_7_script1.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_8_script1.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_9_script1.pt',

    '/nfs/turbo/train_combined_nsra/model_weights_config_1_script2.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_2_script2.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_3_script2.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_4_script2.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_5_script2.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_6_script2.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_7_script2.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_8_script2.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_9_script2.pt',

    '/nfs/turbo/train_combined_nsra/model_weights_config_1_script3.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_2_script3.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_3_script3.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_4_script3.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_5_script3.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_6_script3.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_7_script3.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_8_script3.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_9_script3.pt',

    '/nfs/turbo/train_combined_nsra/model_weights_config_1_script4.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_2_script4.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_3_script4.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_4_script4.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_5_script4.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_6_script4.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_7_script4.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_8_script4.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_9_script4.pt',

    '/nfs/turbo/train_combined_nsra/model_weights_config_1_script5.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_2_script5.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_3_script5.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_4_script5.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_5_script5.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_6_script5.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_7_script5.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_8_script5.pt',
    '/nfs/turbo/train_combined_nsra/model_weights_config_9_script5.pt'
]



# Initialize Model Components
relevance_encoder = RelevanceEncoder2().to(device)
segmentation_decoder = head().to(device)  # Replace with actual FCN

multiheadfusion = MultiEncoderFusionn(TemporalViTEncoder, pretrained_paths,relevance_encoder).to(device)



# Fully Online Meta-Learner
foml_trainer = FOMLTrainer()



observed_dataloaders = [
    observed_loader1, observed_loader2, observed_loader3, observed_loader4, observed_loader5,
    observed_loader6, observed_loader7, observed_loader8, observed_loader9, observed_loader10,
    observed_loader11, observed_loader12, observed_loader13, observed_loader14, observed_loader15,
    observed_loader16, observed_loader17, observed_loader18, observed_loader19, observed_loader20,
    observed_loader21, observed_loader22, observed_loader23, observed_loader24, observed_loader25,
    observed_loader26, observed_loader27, observed_loader28, observed_loader29, observed_loader30,
    observed_loader31, observed_loader32, observed_loader33, observed_loader34, observed_loader35,
    observed_loader36, observed_loader37, observed_loader38, observed_loader39, observed_loader40,
    observed_loader41, observed_loader42, observed_loader43, observed_loader44, observed_loader45
]

timee = 0
total = 150
total1=150
#len(unlabeled_indices)
observed_indices_per_dataset = [[] for _ in range(45)]
counter=0
num=0
#while total1 > 0:
for round_num in range(15):
#for _ in range(total1):  # Replace 2 with however many AL rounds you want
    counter += 1
    logging.info(f"\n[DEBUG] LOOP {counter} START → time = {timee}, total1 = {total1}, len(unlabeled_indices) = {len(unlabeled_indices)}")
    timee += 10
    total1=total1-10
    logging.info(f"[DEBUG] LOOP {counter} AFTER UPDATE → time = {timee}, total1 = {total1}")
    logging.info("start active query train")
    selected_indices, rels, mus, sigmas,zero = active_query_selection(
        unlabeled_indices, segmentation_decoder, timee, total, device
    )
    
    logging.info(f"[ACTIVE QUERY] {zero} out of {len(selected_indices)} selected samples have at least one pixel == 0.")
    
    for selected_idx, rel, mu, sigma in zip(selected_indices, rels, mus, sigmas):
        num+=1
        # === Add selected index to observed indices
        for i in range(45):
            observed_indices_per_dataset[i].append(selected_idx)
    
        # === Build new sample for online update ===
        new_images_list = []
        for dataset in observed_datasets:
            sample = dataset[selected_idx]
            image = sample['images'].unsqueeze(0).to(device)
            new_images_list.append(image)
        label = sample['label'].unsqueeze(0).to(device)
        labels1 = sample['label1'].unsqueeze(0).to(device)
        image_paths = [sample['image_paths']]
        cache_key = f"sample_{selected_idx}"
    
        # === FOML Online update: Only on the newly selected sample
        foml_trainer.update_online(
            new_images_list, label, image_paths, rel, mu, sigma, num, cache_key=cache_key
        )
    
        
    
        # === Remove selected point from unlabeled indices
        #unlabeled_indices = np.delete(unlabeled_indices, selected_idx)
        #unlabeled_indices = np.setdiff1d(unlabeled_indices, [selected_idx])
    # Remove all selected indices at once AFTER loop
    # === FOML Meta update: Uses buffer
    new_images_list=None
    label=None
    image_paths=None
    rel=None
    mu=None
    sigma=None
    ocache_key=None
    foml_trainer.update_meta(new_images_list, label, image_paths, rel, mu, sigma, num, ocache_key=cache_key)

    logging.info(f"Round {time}, Meta Loss")
    unlabeled_indices = np.setdiff1d(unlabeled_indices, selected_indices)
    











def active_query_selection_inf(unlabeled_indices_test, classification_head, time, total, device):
    """
    Active Learning Query Selection:
    - Computes relevance representations (mu, sigma) for observed (train) and unlabeled samples.
    - Computes average Euclidean distance of each unlabeled relevance vector to all trained ones.
    - Computes selection score using: distance + negative exponential of uncertainty.
    - Returns the most uncertain sample index.

    Arguments:
    - relevance_encoder: The trained relevance encoder (outputs mu, sigma).
    - observed_dataloader: DataLoader for labeled (train) dataset.
    - unlabeled_dataset: The dataset containing unlabeled samples.
    - classification_head: The final classification head for predictions.
    - masked_probs_dict: Dictionary storing model prediction probabilities for each sample.
    - device: Torch device.

    Returns:
    - best_index: Index of the most uncertain sample in `unlabeled_dataset`.
    """


    
    ### Step 1: Compute Relevance Vectors for TRAINED DATASET
    mu_train = []
    core_buffer = foml_trainer.core_buffer
    reservoir_buffer = foml_trainer.reservoir_buffer
    relevance_encoder.eval()
    segmentation_decoder.eval()
    multiheadfusion.neck.eval()
    
                    
    if core_buffer is not None:
        with torch.no_grad():
            for rel_vec, _, _ ,_,_,_,_ in core_buffer:
                mu_train.append(rel_vec.to(device))

            if reservoir_buffer:
                reservoir_sample_count = min(3, len(reservoir_buffer))
            
                sampled_reservoir = random.sample(reservoir_buffer, reservoir_sample_count)
                for rel_vec, _, _, _,_,_,_ in sampled_reservoir:
                    mu_train.append(rel_vec.to(device))
                    
    else:
        logging.info("returned")

    mu_train = torch.cat(mu_train, dim=0) if mu_train else torch.empty(0)


    ### ** Step 2: Compute Relevance Vectors for Each UNLABELED IMAGE**
    query_scores = []
    all_rel_samples = [] 
    allmu = []
    allsigma = []
 
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx in unlabeled_indices_test:
            cache_key = f"samplet_{idx}"
            images_list = []
            #for dataloader in observed_dataloaders:
             #   image, _, label,_ = get_sample_from_dataloader(dataloader, idx)
              #  images_list.append(image.to(device))  # shape: (1, C, H, W)
            #labels = label.to(device)  # shape: (1, H, W)
            for dataset in observed_test_datasets:
              sample = dataset[idx]
              image = sample['images'].unsqueeze(0).to(device)
              images_list.append(image)

            labels = sample['label1'].unsqueeze(0).to(device)
            #x = unlabeled_dataset[idx]
            #x['images'] = x['images'].unsqueeze(0).to(device)

            # Get relevance vector for this sample
            #mu_unlabeled = multiheadfusion(x['images'])  # Shape: (1, num_channels)
            _,_,mu_unlabeled, sigma = multiheadfusion(images_list, cache_key = cache_key)
            rel = sample_relevance_vectors(mu_unlabeled, sigma, num_samples=1)
            allmu.append(mu_unlabeled)
            allsigma.append(sigma)
            all_rel_samples.append(rel)
            query_score_list = []
            qs_list = []

            for i in range(1):
                rel_vec = rel[i]  # shape: (1, num_channels)

                # --- Step 1: Distance to all training mu
                distance = torch.cdist(rel_vec, mu_train, p=2).mean().item()

                # --- Step 2: Prediction uncertainty
                fused_feat,_,_,_ = multiheadfusion(images_list, rel_vec, cache_key = cache_key)  # Assumes fusion works with rel_vec
                logits,_ = segmentation_decoder(fused_feat)
                #probs = F.softmax(logits, dim=1)
                probs = torch.sigmoid(logits)
                non_2_mask = (labels != 2).expand_as(probs).float()
                probs_masked = probs * non_2_mask

                valid_pixel_count = non_2_mask.sum().clamp(min=1.0)
                overall_prob = (probs_masked.sum() / valid_pixel_count).item()
                
                prediction_uncertainty = abs(0.5 - overall_prob)

                avg_prob_1 = probs_masked.sum() / non_2_mask.sum()
                avg_prob_0 = ((1 - probs) * non_2_mask).sum() / non_2_mask.sum()

                w0 = 1 - (time / total)
                w1 = 1
                final_prob1 = w0 * avg_prob_0 + w1 * avg_prob_1

               

                query_score = distance * np.exp(-prediction_uncertainty)
                qs = final_prob1.item() * np.exp(-distance)

                query_score_list.append(query_score)
                qs_list.append(qs)

            avg_query_score = np.mean(query_score_list)
            avg_qs = np.mean(qs_list)

            k = (total - time) / (total + time)
            final_score = k * avg_query_score + (1 - k) * avg_qs
            #query_scores.append(final_score)
            query_scores.append((final_score, idx, rel, mu_unlabeled, sigma))


    
    # Sort and get top 5 by score
    query_scores.sort(key=lambda x: x[0], reverse=True)
    top5 = query_scores[:5]
    
    best_indices = [x[1] for x in top5]
    best_rels = [x[2] for x in top5]
    mus = [x[3] for x in top5]
    sigmas = [x[4] for x in top5]
    

   
    
    return best_indices, best_rels, mus, sigmas

    



test_indices = np.arange(len(test_dataset1))  # All test indices
observed_indices_test = np.random.choice(test_indices, 0, replace=False)  # Initial observed (labeled) data
unlabeled_indices_test = np.setdiff1d(test_indices, observed_indices_test)  # Remaining unlabeled data


counter1=0
reward = 0  # Initialize reward  
total = 50
total1=50
#len(unlabeled_indices_test)
time = 0
num = 0
all_preds=[]
all_labels=[]
#while len(unlabeled_indices_test) > 0:
#for _ in range(total1):  
#while total1 > 0:  
for round_num in range(10):
 
    counter1 = counter1+1
    
  
   
    logging.info(f"\n[DEBUG] LOOP {counter1} START → time = {time}, total1 = {total1}, len(unlabeled_indices_test) = {len(unlabeled_indices_test)}") 
    time+=5
    
    total1=total1-5
    logging.info(f"[DEBUG] LOOP {counter1} AFTER UPDATE → time = {time}, total1 = {total1}")
    
    indexx, rel_bestt, muu, sigmaa = active_query_selection_inf(unlabeled_indices_test, segmentation_decoder, time, total, device)
    all_predsl=[]
    all_labelsl=[]
    rewardl = 0
    save_all_rel_mu_sigma(ut, observed_test_datasets, multiheadfusion, device)
    for index, rel_best, mu, sigma in zip(indexx, rel_bestt, muu, sigmaa):
    
        # index = the specific test point you're trying to observe
        cache_key = f"samplet_{index}"
        images_list = []
        labels_list = []
        
        
        
        
        
    
        for dataset in observed_test_datasets:
          sample = dataset[index]
          image = sample['images'].unsqueeze(0).to(device)
          images_list.append(image)
    
        labels = sample['label'].unsqueeze(0).to(device)
        labels1 = sample['label1'].unsqueeze(0).to(device)
        image_paths = [sample['image_paths']]
    
        # --- Perform Inference on Selected Image and Update Reward ---
        
        relevance_encoder.eval()
        segmentation_decoder.eval()
        multiheadfusion.neck.eval()
        with torch.no_grad():
            predictions = []  # Store each of the 10 predictions
        
            for i in range(1):
                rel_vec = rel_best[i]  # Shape: (1, D)
        
                # Fuse and forward pass
                fused_feat,_,_,_ = multiheadfusion(images_list, rel_vec, cache_key = cache_key, check=1)  # Shape: (1, C, H, W)
                output, _ = segmentation_decoder(fused_feat)  # Shape: (1, num_classes, H, W)
                probs1 = torch.sigmoid(output)  # [B, 1, H, W]
                preds1 = (probs1 > 0.54).squeeze(1)  # [B, H, W]
                for i in range(probs1.shape[0]):  # batch dimension
                    logging.info(f"Sample {i} valid probs and labels:")
                    
                    mask = labels1[i] != 2
                    valid_probs = probs1[i, 0][mask]
                    valid_labels = labels1[i][mask]
                
                    for p, l in zip(valid_probs, valid_labels):
                        logging.info(f"Prob: {p.item():.4f}  |  Label: {l.item()}")


                # Sigmoid + threshold
                
                #pred_label = torch.argmax(output, dim=1)  # Shape: (1, H, W)
                predictions.append(preds1)
        
            # === Stack all predictions: shape (10, 1, H, W) -> squeeze to (10, H, W)
            #predictions = torch.cat(predictions, dim=0).squeeze(1)
        
            # === Majority Voting along the 0th dim (over the 10 votes)
            # Resulting shape: (H, W)
            #final_pred = torch.mode(predictions, dim=0).values.unsqueeze(0)
            final_pred = predictions[0]
            all_preds.append(final_pred)
            all_labels.append(labels1)
            all_predsl.append(final_pred)
            all_labelsl.append(labels1)
        
            # === Compare to ground truth, excluding label==2
            selected_label = labels1  # shape: (H, W)
            mask = (selected_label != 2)
        
            correct_mask = (final_pred == selected_label) & mask & (selected_label == 1)
            
            correct = correct_mask.sum().item()
        
            # Update reward
            reward += correct
            rewardl+=correct
            
        num+=1
        # === Pass list of 10 tensors to the trainer ===
        foml_trainer.update_online(images_list, labels, image_paths, rel_best, mu, sigma, num, cache_key = cache_key)
       
        
        unlabeled_indices_test = np.setdiff1d(unlabeled_indices_test, [index])
      
       
    
        
        logging.info(f"Remaining unlabeled samples: {len(unlabeled_indices_test)}")
        
    
    all_predsl1 = torch.cat(all_predsl, dim=0)
    all_labelsl1 = torch.cat(all_labelsl, dim=0)
    num_zeroes = (all_labelsl1 == 0).sum().item()
    num_ones = (all_labelsl1 == 1).sum().item()
    metricsl = evaluate_segmentation(all_predsl1, all_labelsl1)
    logging.info(f"Reward local rate: {rewardl/(num_ones+1e-6)}")
    logging.info(f"Evaluation local  Metrics: {metricsl}")
    logging.info(f"num total actual zeroes: {num_zeroes}")
    logging.info(f"num total actual ones: {num_ones}")  
    # Convert predicted logits to labels
    pred_labels = all_predsl1.detach().cpu().numpy()
    true_labels = all_labelsl1.detach().cpu().numpy()
    
    # Flatten all pixels into 1D arrays
    pred_flat = pred_labels.flatten()
    true_flat = true_labels.flatten()
    
    # Optional: If you want to ignore class 2 in confusion matrix
    mask = true_flat != 2
    pred_flat = pred_flat[mask]
    true_flat = true_flat[mask]
    
    # Compute confusion matrix
    conf_mat = confusion_matrix(true_flat, pred_flat, labels=[0, 1])
    
    # Log it nicely
    logging.info("Confusion Matrix:")
    logging.info("\n" + str(conf_mat))
    # === FOML Meta update: Uses buffer
    new_images_list=None
    label=None
    image_paths=None
    rel=None
    mu=None
    sigma=None
    ocache_key=None
    foml_trainer.update_meta(new_images_list, label, image_paths, rel, mu, sigma, num, ocache_key=cache_key)
    logging.info(f"Inference Round: {time}")
    logging.info("Meta Loss")
    
all_preds1 = torch.cat(all_preds, dim=0)
all_labels1 = torch.cat(all_labels, dim=0)
num_zeroes = (all_labels1 == 0).sum().item()
num_ones = (all_labels1 == 1).sum().item()
metrics = evaluate_segmentation(all_preds1, all_labels1)
logging.info(f"Reward rate: {reward/50}")
logging.info(f"Evaluation Metrics: {metrics}")
logging.info(f"num total actual zeroes: {num_zeroes}")
logging.info(f"num total actual ones: {num_ones}")  
# Convert predicted logits to labels
pred_labels = all_preds1.detach().cpu().numpy()
true_labels = all_labels1.detach().cpu().numpy()

# Flatten all pixels into 1D arrays
pred_flat = pred_labels.flatten()
true_flat = true_labels.flatten()

# Optional: If you want to ignore class 2 in confusion matrix
mask = true_flat != 2
pred_flat = pred_flat[mask]
true_flat = true_flat[mask]

# Compute confusion matrix
conf_mat = confusion_matrix(true_flat, pred_flat, labels=[0, 1])

# Log it nicely
logging.info("Confusion Matrix:")
logging.info("\n" + str(conf_mat))
    
            
torch.save({
    'relevance_encoder': relevance_encoder.state_dict(),
    'segmentation_decoder': segmentation_decoder.state_dict(),
    'neck': multiheadfusion.neck.state_dict()
}, '/scratch/trained_model_weights1.pth')

    
    
    
    
    