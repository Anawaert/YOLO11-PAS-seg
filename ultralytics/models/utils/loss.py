# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the
    DETR object detection model.

    DETR（DEtection TRansformer）损失类，用于计算各种损失组件。

    该类计算分类损失、边界框损失、GIoU 损失，并可选地计算 DETR 目标检测模型的辅助损失。

    Attributes:
        nc (int): Number of classes. 不同损失组件的系数。
        loss_gain (Dict): Coefficients for different loss components. 不同损失组件的系数。
        aux_loss (bool): Whether to compute auxiliary losses. 是否计算辅助损失。
        use_fl (bool): Whether to use FocalLoss. 是否使用 FocalLoss。
        use_vfl (bool): Whether to use VarifocalLoss. 是否使用 VarifocalLoss。
        use_uni_match (bool): Whether to use a fixed layer for auxiliary branch label assignment. 是否使用固定层进行辅助分支标签分配。
        uni_match_ind (int): Index of fixed layer to use if use_uni_match is True. 如果 use_uni_match 为 True，则使用固定层的索引。
        matcher (HungarianMatcher): Object to compute matching cost and indices. 用于计算匹配成本和索引的对象。
        fl (FocalLoss | None): Focal Loss object if use_fl is True, otherwise None. 如果 use_fl 为 True，则为 Focal Loss 对象，否则为 None。
        vfl (VarifocalLoss | None): Varifocal Loss object if use_vfl is True, otherwise None. 如果 use_vfl 为 True，则为 Varifocal Loss 对象，否则为 None。
        device (torch.device): Device on which tensors are stored. 存储张量的设备。
    """

    def __init__(
        self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=False, uni_match_ind=0
    ):
        """
        Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        使用默认 loss_gain（如果未提供）初始化 DETR 损失函数。

        使用预设成本增益初始化 HungarianMatcher。支持辅助损失和各种损失类型。

        Args:
            nc (int): Number of classes. 类别数。
            loss_gain (Dict): Coefficients for different loss components. 不同损失组件的系数。
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer. 是否使用每个解码器层的辅助损失。
            use_fl (bool): Whether to use FocalLoss. 是否使用 FocalLoss。
            use_vfl (bool): Whether to use VarifocalLoss. 是否使用 VarifocalLoss。
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment. 是否使用固定层进行辅助分支标签分配。
            uni_match_ind (int): Index of fixed layer for uni_match. uni_match 的固定层索引。
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """
        Compute classification loss based on predictions, target values, and ground truth scores.

        根据预测、目标值和真实分数计算分类损失。
        """
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        """
        Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        计算预测和真实边界框的边界框和 GIoU 损失。
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # 此函数用于未来的 RT-DETR Segment 模型
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        match_indices=None,
        postfix="",
        masks=None,
        gt_mask=None,
    ):
        """
        Get auxiliary losses for intermediate decoder layers.

        获取中间解码器层的辅助损失。

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers. 辅助层的预测边界框。
            pred_scores (torch.Tensor): Predicted scores from auxiliary layers. 辅助层的预测分数。
            gt_bboxes (torch.Tensor): Ground truth bounding boxes. 真实边界框。
            gt_cls (torch.Tensor): Ground truth classes. 真实类别。
            gt_groups (List[int]): Number of ground truths per image. 每张图像的真实数量。
            match_indices (List[tuple], optional): Pre-computed matching indices. 预先计算的匹配索引。
            postfix (str): String to append to loss names. 附加到损失名称的字符串。
            masks (torch.Tensor, optional): Predicted masks if using segmentation. 如果使用分割，则为预测的掩码。
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation. 如果使用分割，则为真实掩码。

        Returns:
            (Dict): Dictionary of auxiliary losses. 辅助损失的字典。
        """
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices):
        """
        Extract batch indices, source indices, and destination indices from match indices.

        从匹配索引中提取批次索引、源索引和目标索引。

        Args:
            match_indices (List[tuple]): List of tuples containing matched indices. 包含匹配索引的元组列表。

        Returns:
            (tuple): Tuple containing (batch_idx, src_idx) and dst_idx. 包含 (batch_idx, src_idx) 和 dst_idx 的元组。
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        """
        Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

        根据匹配索引将预测边界框分配给真实边界框。

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes. 预测边界框。
            gt_bboxes (torch.Tensor): Ground truth bounding boxes. 真实边界框。
            match_indices (List[tuple]): List of tuples containing matched indices. 包含匹配索引的元组列表。

        Returns:
            (tuple): Tuple containing assigned predictions and ground truths. 包含分配的预测和真实值的元组。
        """
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        masks=None,
        gt_mask=None,
        postfix="",
        match_indices=None,
    ):
        """
        Calculate losses for a single prediction layer.

        计算单个预测层的损失。

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes. 预测边界框。
            pred_scores (torch.Tensor): Predicted class scores. 预测类别分数。
            gt_bboxes (torch.Tensor): Ground truth bounding boxes. 真实边界框。
            gt_cls (torch.Tensor): Ground truth classes. 真实类别。
            gt_groups (List[int]): Number of ground truths per image. 每张图像的真实数量。
            masks (torch.Tensor, optional): Predicted masks if using segmentation. 如果使用分割，则为预测的掩码。
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation. 如果使用分割，则为真实掩码。
            postfix (str): String to append to loss names. 附加到损失名称的字符串。
            match_indices (List[tuple], optional): Pre-computed matching indices. 预先计算的匹配索引。

        Returns:
            (Dict): Dictionary of losses. 损失的字典。
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(self, pred_bboxes, pred_scores, batch, postfix="", **kwargs):
        """
        Calculate loss for predicted bounding boxes and scores.

        计算预测边界框和分数的损失。

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape [l, b, query, 4]. 预测边界框，形状 [l, b, query, 4]。
            pred_scores (torch.Tensor): Predicted class scores, shape [l, b, query, num_classes]. 预测类别分数，形状 [l, b, query, num_classes]。
            batch (Dict): Batch information containing: cls, bboxes, gt_groups. 包含 cls、bboxes、gt_groups 的批次信息。
                cls (torch.Tensor): Ground truth classes, shape [num_gts]. 真实类别，形状 [num_gts]。
                bboxes (torch.Tensor): Ground truth bounding boxes, shape [num_gts, 4]. 真实边界框，形状 [num_gts, 4]。
                gt_groups (List[int]): Number of ground truths for each image in the batch. 批次中每张图像的真实数量。
            postfix (str): Postfix for loss names. 损失名称的后缀。
            **kwargs (Any): Additional arguments, may include 'match_indices'. 其他参数，可能包括 'match_indices'。

        Returns:
            (Dict): Computed losses, including main and auxiliary (if enabled). 计算的损失，包括主要和辅助（如果启用）。

        Notes:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True. 如果 self.aux_loss 为 True，则使用 pred_bboxes 和 pred_scores 的最后一个元素进行主要损失，
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.

    扩展 DETRLoss 的实时 DeepTracker（RT-DETR）检测损失类。

    该类计算 RT-DETR 模型的检测损失，其中包括标准检测损失以及在提供去噪元数据时的额外去噪训练损失。
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        Forward pass to compute detection loss with optional denoising loss.

        前向传递以计算检测损失和可选的去噪损失。

        Args:
            preds (tuple): Tuple containing predicted bounding boxes and scores. 包含预测边界框和分数的元组。
            batch (Dict): Batch data containing ground truth information. 包含真实信息的批次数据。
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. 去噪边界框。
            dn_scores (torch.Tensor, optional): Denoising scores. 去噪分数。
            dn_meta (Dict, optional): Metadata for denoising. 去噪的元数据。

        Returns:
            (Dict): Dictionary containing total loss and denoising loss if applicable. 包含总损失和去噪损失（如果适用）的字典。
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        # 检查去噪元数据以计算去噪训练损失
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            # 获取去噪的匹配索引
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            # 计算去噪训练损失
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            # 如果未提供去噪元数据，则将去噪损失设置为零
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get match indices for denoising.

        获取去噪的匹配索引。

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising. 包含去噪正索引的张量列表。
            dn_num_group (int): Number of denoising groups. 去噪组的数量。
            gt_groups (List[int]): List of integers representing number of ground truths per image. 表示每张图像的真实数量的整数列表。

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising. 包含去噪匹配索引的元组列表。
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), "Expected the same length, "
                f"but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
