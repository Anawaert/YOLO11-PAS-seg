# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions
    to compute metrics such as mAP for both detection and segmentation tasks.

    一个扩展了 DetectionValidator 类的类，用于基于分割模型进行验证。

    该验证器处理分割模型的评估，处理边界框和掩码预测以计算检测和分割任务的 mAP 等指标。

    Attributes:
        plot_masks (List): List to store masks for plotting. 用于存储绘图掩码的列表。
        process (callable): Function to process masks based on save_json and save_txt flags. 根据 save_json 和 save_txt 标志处理掩码的函数。
        args (namespace): Arguments for the validator. 用于验证器的参数。
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks. 用于分割任务的指标计算器。
        stats (Dict): Dictionary to store statistics during validation. 用于在验证期间存储统计信息的字典。

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        初始化 SegmentationValidator 并将任务设置为 'segment'，将指标设置为 SegmentMetrics。

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation. 用于验证的数据加载器。
            save_dir (Path, optional): Directory to save results. 保存结果的目录。
            pbar (Any, optional): Progress bar for displaying progress. 用于显示进度的进度条。
            args (namespace, optional): Arguments for the validator. 用于验证器的参数。
            _callbacks (List, optional): List of callback functions. 回调函数列表。
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics(save_dir=self.save_dir)

    def preprocess(self, batch):
        """
        Preprocess batch by converting masks to float and sending to device.

        通过将掩码转换为浮点数并发送到设备来预处理批次。
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """
        Initialize metrics and select mask processing function based on save_json flag.

        初始化指标并根据 save_json 标志选择掩码处理函数。

        Args:
            model (torch.nn.Module): Model to validate. 要验证的模型。
        """
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        # more accurate vs faster
        # 更准确 vs 更快
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """
        Return a formatted description of evaluation metrics.

        返回评估指标的格式化描述。
        """
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """
        Post-process YOLO predictions and return output detections with proto.

        后处理 YOLO 预测并返回带有 proto 的输出检测。

        Args:
            preds (List): Raw predictions from the model. 模型的原始预测。

        Returns:
            p (torch.Tensor): Processed detection predictions. 处理后的检测预测。
            proto (torch.Tensor): Prototype masks for segmentation. 用于分割的原型掩码。
        """
        p = super().postprocess(preds[0])
        # second output is len 3 if pt, but only 1 if exported
        # 如果 pt，则第二个输出长度为 3，但如果导出，则只有 1
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
        return p, proto

    def _prepare_batch(self, si, batch):
        """
        Prepare a batch for training or inference by processing images and targets.

        通过处理图像和目标来为训练或推理准备批次。

        Args:
            si (int): Batch index. 批次索引。
            batch (Dict): Batch data containing images and targets. 包含图像和目标的批次数据。

        Returns:
            (Dict): Prepared batch with processed images and targets. 处理后的批次，其中包含处理后的图像和目标。
        """
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """
        Prepare predictions for evaluation by processing bounding boxes and masks.

        通过处理边界框和掩码来准备预测以进行评估。

        Args:
            pred (torch.Tensor): Raw predictions from the model. 模型的原始预测。
            pbatch (Dict): Prepared batch data. 准备好的批次数据。
            proto (torch.Tensor): Prototype masks for segmentation. 用于分割的原型掩码。

        Returns:
            predn (torch.Tensor): Processed bounding box predictions. 处理后的边界框预测。
            pred_masks (torch.Tensor): Processed mask predictions. 处理后的掩码预测。
        """
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])
        return predn, pred_masks

    def update_metrics(self, preds, batch):
        """
        Update metrics with the current batch predictions and targets.

        使用当前批次的预测和目标更新指标。

        Args:
            preds (List): Predictions from the model. 模型的预测。
            batch (Dict): Batch data containing images and targets. 包含图像和目标的批次数据。
        """
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Masks
            # 掩膜
            gt_masks = pbatch.pop("masks")
            # Predictions
            # 预测
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            # 评估
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_m"] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                )
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save
            # 保存
            if self.args.save_json:
                self.pred_to_json(
                    predn,
                    batch["im_file"][si],
                    ops.scale_image(
                        pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        pbatch["ori_shape"],
                        ratio_pad=batch["ratio_pad"][si],
                    ),
                )
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_masks,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def finalize_metrics(self, *args, **kwargs):
        """
        Set speed and confusion matrix for evaluation metrics.

        为评估指标设置速度和混淆矩阵。
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        根据边界框和可选掩码计算批次的正确预测矩阵。

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detected bounding boxes and
                associated confidence scores and class indices. Each row is of the format [x1, y1, x2, y2, conf, class].
                表示检测到的边界框及其置信度分数和类别索引的张量，形状为 (N, 6)。每行的格式为 [x1, y1, x2, y2, conf, class]。
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground truth bounding box coordinates.
                Each row is of the format [x1, y1, x2, y2].
                表示地面实况边界框坐标的张量，形状为 (M, 4)。每行的格式为 [x1, y1, x2, y2]。
            gt_cls (torch.Tensor): Tensor of shape (M,) representing ground truth class indices. 表示真实类别索引的张量，形状为 (M,)。
            pred_masks (torch.Tensor, optional): Tensor representing predicted masks, if available. The shape should
                match the ground truth masks.
                表示预测掩码的张量，如果可用。形状应与地面实况掩码匹配。
            gt_masks (torch.Tensor, optional): Tensor of shape (M, H, W) representing ground truth masks, if available.
                表示真实掩码的张量，形状为 (M, H, W)（如果有的话）。
            overlap (bool): Flag indicating if overlapping masks should be considered. 指示是否应考虑重叠掩码的标志。
            masks (bool): Flag indicating if the batch contains mask data. 指示批次是否包含掩码数据的标志。

        Returns:
            (torch.Tensor): A correct prediction matrix of shape (N, 10), where 10 represents different IoU levels.
                形状为 (N, 10) 的正确预测矩阵，其中 10 表示不同的 IoU 水平。

        Note:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
              如果 `masks` 为 True，则函数计算预测和真实掩码之间的 IoU。

            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.
              如果 `overlap` 为 True 并且 `masks` 为 True，则在计算 IoU 时会考虑重叠掩码。

        Examples:
            >>> detections = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            >>> gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            >>> gt_cls = torch.tensor([1, 0])
            >>> correct_preds = validator._process_batch(detections, gt_bboxes, gt_cls)
        """
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """
        Plot validation samples with bounding box labels and masks.

        绘制带有边界框标签和掩码的验证样本。

        Args:
            batch (Dict): Batch data containing images and targets. 包含图像和目标的批次数据。
            ni (int): Batch index. 批次索引。
        """
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        Plot batch predictions with masks and bounding boxes.

        绘制带有掩码和边界框的批次预测。

        Args:
            batch (Dict): Batch data containing images. 包含图像的批次数据。
            preds (List): Predictions from the model. 模型的预测。
            ni (int): Batch index. 批次索引。
        """
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=15),  # not set to self.args.max_det due to slow plotting speed - 由于绘图速度慢，未设置为 self.args.max_det
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred - 预测
        self.plot_masks.clear()

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        将 YOLO 检测保存到特定格式的 txt 文件中，以规范化坐标表示。

        Args:
            predn (torch.Tensor): Predictions in the format [x1, y1, x2, y2, conf, cls]. 格式为 [x1, y1, x2, y2, conf, cls] 的预测。
            pred_masks (torch.Tensor): Predicted masks. 预测掩码。
            save_conf (bool): Whether to save confidence scores. 是否保存置信度分数。
            shape (Tuple): Original image shape. 原始图像形状。
            file (Path): File path to save the detections. 保存检测结果的文件路径。
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            masks=pred_masks,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result for COCO evaluation.

        为 COCO 评估保存一个 JSON 结果。

        Args:
            predn (torch.Tensor): Predictions in the format [x1, y1, x2, y2, conf, cls]. 格式为 [x1, y1, x2, y2, conf, cls] 的预测。
            filename (str): Image filename. 图像文件名。
            pred_masks (numpy.ndarray): Predicted masks. 预测掩码。

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa - pycocotools

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner - xy 中心到左上角
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def eval_json(self, stats):
        """
        Return COCO-style object detection evaluation metrics.

        返回 COCO 风格的对象检测评估指标。
        """
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations - 注释
            pred_json = self.save_dir / "predictions.json"  # predictions - 预测
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api - 初始化注释 API
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path) - 初始化预测 API（必须传递字符串，而不是 Path）
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval - 评估的 im
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50 - 更新 mAP50-95 和 mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
