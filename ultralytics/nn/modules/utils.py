# YOLO11-PAS-seg - https://github.com/Anawaert/YOLO11-PAS-seg

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

__all__ = "multi_scale_deformable_attn_pytorch", "inverse_sigmoid"


def _get_clones(module, n):
    """
    Create a list of cloned modules from the given module.

    创建一个给定模块的克隆模块列表。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """
    Initialize conv/fc bias value according to a given probability value.

    根据给定的概率值初始化 conv/fc 偏置值。
    """
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init - 返回 bias_init


def linear_init(module):
    """
    Initialize the weights and biases of a linear module.

    初始化线性模块的权重和偏置。
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """
    Calculate the inverse sigmoid function for a tensor.

    计算张量的反 sigmoid 函数。
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Implement multi-scale deformable attention in PyTorch.

    This function performs deformable attention across multiple feature map scales, allowing the model to attend to
    different spatial locations with learned offsets.

    在 PyTorch 中实现多尺度可变形注意力。

    该函数在多个特征图尺度上执行可变形注意力，允许模型使用学习到的偏移量关注不同的空间位置。

    Args:
        value (torch.Tensor): The value tensor with shape (bs, num_keys, num_heads, embed_dims). 具有形状 (bs, num_keys, num_heads, embed_dims) 的值张量。
        value_spatial_shapes (torch.Tensor): Spatial shapes of the value tensor with shape (num_levels, 2). 具有形状 (num_levels, 2) 的值张量的空间形状。
        sampling_locations (torch.Tensor): The sampling locations with shape
            (bs, num_queries, num_heads, num_levels, num_points, 2). 具有形状 (bs, num_queries, num_heads, num_levels, num_points, 2) 的采样位置。
        attention_weights (torch.Tensor): The attention weights with shape
            (bs, num_queries, num_heads, num_levels, num_points). 具有形状 (bs, num_queries, num_heads, num_levels, num_points) 的注意力权重。

    Returns:
        (torch.Tensor): The output tensor with shape (bs, num_queries, embed_dims). 具有形状 (bs, num_queries, embed_dims) 的输出张量。

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()
