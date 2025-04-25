# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):
    """
    A module implementing the HungarianMatcher, which is a differentiable module to solve the assignment problem in an
    end-to-end fashion.

    HungarianMatcher performs optimal assignment over the predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally, mask predictions.

    实现 HungarianMatcher 的模块，这是一个可微分模块，用于以端到端的方式解决分配问题。

    HungarianMatcher 使用考虑分类分数、边界框坐标和可选的掩码预测的成本函数，在预测和真实边界框之间执行最优分配。

    Attributes:
        cost_gain (Dict): Dictionary of cost coefficients: 'class', 'bbox', 'giou', 'mask', and 'dice'. 成本系数字典：'class'，'bbox'，'giou'，'mask' 和 'dice'。
        use_fl (bool): Indicates whether to use Focal Loss for the classification cost calculation. 是否使用 Focal Loss 进行分类成本计算。
        with_mask (bool): Indicates whether the model makes mask predictions. 指示模型是否进行掩码预测。
        num_sample_points (int): The number of sample points used in mask cost calculation. 掩码成本计算中使用的样本点数。
        alpha (float): The alpha factor in Focal Loss calculation. Focal Loss 计算中的 alpha 因子。
        gamma (float): The gamma factor in Focal Loss calculation. Focal Loss 计算中的 gamma 因子。

    Methods:
        forward: Computes the assignment between predictions and ground truths for a batch. 计算批次的预测和真实值之间的分配。
        _cost_mask: Computes the mask cost and dice cost if masks are predicted. 如果预测了掩码，则计算掩码成本和 dice 成本。
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """
        Initialize a HungarianMatcher module for optimal assignment of predicted and ground truth bounding boxes.

        初始化 HungarianMatcher 模块，用于预测和真实边界框的最优分配。
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        """
        Forward pass for HungarianMatcher. Computes costs based on prediction and ground truth and finds the optimal
        matching between predictions and ground truth based on these costs.

        HungarianMatcher 的前向传递。基于预测和真实值计算成本，并根据这些成本找到预测和真实值之间的最佳匹配。

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 4). 预测的边界框。
            pred_scores (torch.Tensor): Predicted scores with shape (batch_size, num_queries, num_classes). 预测的分数。
            gt_cls (torch.Tensor): Ground truth classes with shape (num_gts, ). 真实类别。
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (num_gts, 4). 真实边界框。
            gt_groups (List[int]): List of length equal to batch size, containing the number of ground truths for
                each image. 长度等于批次大小的列表，包含每个图像的真实值数量。
            masks (torch.Tensor, optional): Predicted masks with shape (batch_size, num_queries, height, width).
                预测的掩码 (batch_size, num_queries, height, width)。
            gt_mask (List[torch.Tensor], optional): List of ground truth masks, each with shape (num_masks, Height, Width).
                真实掩码列表，每个掩码的形状为 (num_masks, Height, Width)。

        Returns:
            (List[Tuple[torch.Tensor, torch.Tensor]]): A list of size batch_size, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

            大小为 batch_size 的列表，每个元素是一个元组 (index_i, index_j)，其中:
                - index_i 是选定预测的索引张量（按顺序）
                - index_j 是相应选定的真实目标的索引张量（按顺序）
            对于每个批次元素，满足:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # 我们展平以批量计算成本矩阵
        # (batch_size * num_queries, num_classes)
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        # (batch_size * num_queries, 4)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        # Compute the classification cost
        # 计算分类成本
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores**self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # Compute the L1 cost between boxes
        # 计算边界框之间的 L1 成本
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt)

        # Compute the GIoU cost between boxes, (bs*num_queries, num_gt)
        # 计算边界框之间的 GIoU 成本
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)

        # Final cost matrix
        # 最终成本矩阵
        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_giou
        )
        # Compute the mask cost and dice cost
        # 计算掩码成本和 dice 成本
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # Set invalid values (NaNs and infinities) to 0 (fixes ValueError: matrix contains invalid numeric entries)
        # 将无效值（NaN 和无穷大）设置为 0（修复 ValueError: 矩阵包含无效的数值条目）
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # (idx for queries, idx for gt)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
            for k, (i, j) in enumerate(indices)
        ]

    # This function is for future RT-DETR Segment models
    # 该函数是为未来的 RT-DETR Segment 模型准备的
    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
    #     # all masks share the same set of points for efficient matching
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
    #     sample_points = 2.0 * sample_points - 1.0
    #
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)
    #
    #     tgt_mask = torch.cat(gt_mask).unsqueeze(1)
    #     sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    #
    #     with torch.amp.autocast("cuda", enabled=False):
    #         # binary cross entropy cost
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
    #         cost_mask /= self.num_sample_points
    #
    #         # dice cost
    #         out_mask = F.sigmoid(out_mask)
    #         numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
    #         denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
    #         cost_dice = 1 - (numerator + 1) / (denominator + 1)
    #
    #         C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    #     return C


def get_cdn_group(
    batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False
):
    """
    Get contrastive denoising training group with positive and negative samples from ground truths.

    从真实值中获取具有正样本和负样本的对比去噪训练组。

    Args:
        batch (Dict): A dict that includes 'gt_cls' (torch.Tensor with shape (num_gts, )), 'gt_bboxes'
            (torch.Tensor with shape (num_gts, 4)), 'gt_groups' (List[int]) which is a list of batch size length
            indicating the number of gts of each image.
            包含 'gt_cls'（形状为 (num_gts, ) 的 torch.Tensor）、'gt_bboxes'
            （形状为 (num_gts, 4) 的 torch.Tensor）、'gt_groups'（长度为批次大小的列表，指示每个图像的真实值数量）的字典。
        num_classes (int): Number of classes. 类别数。
        num_queries (int): Number of queries. 查询数。
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space. 嵌入权重，将类别标签映射到嵌入空间。
        num_dn (int, optional): Number of denoising queries. 去噪查询数。
        cls_noise_ratio (float, optional): Noise ratio for class labels. 类别标签的噪声比率。
        box_noise_scale (float, optional): Noise scale for bounding box coordinates. 边界框坐标的噪声比例。
        training (bool, optional): If it's in training mode. 是否处于训练模式。

    Returns:
        padding_cls (Optional[torch.Tensor]): The modified class embeddings for denoising. 用于去噪的修改后的类别嵌入。
        padding_bbox (Optional[torch.Tensor]): The modified bounding boxes for denoising. 用于去噪的修改后的边界框。
        attn_mask (Optional[torch.Tensor]): The attention mask for denoising. 用于去噪的注意力掩码。
        dn_meta (Optional[Dict]): Meta information for denoising. 用于去噪的元信息。
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    # Pad gt to max_num of a batch
    # 将真实值填充到批次的最大数量
    bs = len(gt_groups)
    gt_cls = batch["cls"]  # (bs*num, )
    gt_bbox = batch["bboxes"]  # bs*num, 4
    b_idx = batch["batch_idx"]

    # Each group has positive and negative queries.
    # 每个组有正样本和负样本查询。
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # Positive and negative mask
    # 正负掩码
    # (bs*num*num_group, ), the second total_num*num_group part as negative samples
    # (bs*num*num_group, )，第二部分 total_num*num_group 作为负样本
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        # Half of bbox prob
        # 一半的边界框概率
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # Randomly put a new one here
        # 随机放一个新的
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)

        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4

        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid - 逆 sigmoid

    num_dn = int(max_nums * 2 * num_group)  # total denoising queries - 总去噪查询
    # class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])
    dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)

    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # Match query cannot see the reconstruct
    # 匹配查询不能看到重建
    attn_mask[num_dn:, :num_dn] = True
    # Reconstruct cannot see each other
    # 重建不能相互看到
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )
