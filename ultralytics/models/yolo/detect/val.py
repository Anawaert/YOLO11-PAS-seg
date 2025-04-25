# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    一个扩展了 BaseValidator 类的类，用于基于检测模型进行验证。

    该类实现了特定于目标检测任务的验证功能，包括指标计算、预测处理和结果可视化。

    Attributes:
        nt_per_class (np.ndarray): Number of targets per class. 每个类别的目标数量。
        nt_per_image (np.ndarray): Number of targets per image. 每张图片的目标数量。
        is_coco (bool): Whether the dataset is COCO. 数据集是否为 COCO。
        is_lvis (bool): Whether the dataset is LVIS. 数据集是否为 LVIS。
        class_map (List): Mapping from model class indices to dataset class indices. 从模型类别索引到数据集类别索引的映射。
        metrics (DetMetrics): Object detection metrics calculator. 目标检测指标计算器。
        iouv (torch.Tensor): IoU thresholds for mAP calculation. 用于 mAP 计算的 IoU 阈值。
        niou (int): Number of IoU thresholds. IoU 阈值的数量。
        lb (List): List for storing ground truth labels for hybrid saving. 用于混合保存的存储真实标签的列表。
        jdict (List): List for storing JSON detection results. 用于存储 JSON 检测结果的列表。
        stats (Dict): Dictionary for storing statistics during validation. 用于存储验证过程中的统计信息的字典。

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize detection validator with necessary variables and settings.

        初始化检测验证器，设置必要的变量和参数。

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation. 验证使用的数据加载器。
            save_dir (Path, optional): Directory to save results. 保存结果的目录。
            pbar (Any, optional): Progress bar for displaying progress. 用于显示进度的进度条。
            args (Dict, optional): Arguments for the validator. 验证器的参数。
            _callbacks (List, optional): List of callback functions. 回调函数列表。
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95 - 用于 mAP@0.5:0.95 的 IoU 向量
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling - 用于自动标注
        if self.args.save_hybrid and self.args.task == "detect":
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def preprocess(self, batch):
        """
        Preprocess batch of images for YOLO validation.

        预处理 YOLO 验证的图像批次。

        Args:
            batch (Dict): Batch containing images and annotations. 包含图像和注释的批次。

        Returns:
            (Dict): Preprocessed batch. 预处理后的批次。
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid and self.args.task == "detect":
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO detection validation.

        初始化 YOLO 检测验证的评估指标。

        Args:
            model (torch.nn.Module): Model to validate. 要验证的模型。
        """
        val = self.data.get(self.args.split, "")  # validation path - 验证路径
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val - 运行最终验证
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """
        Return a formatted string summarizing class metrics of YOLO model.

        返回总结 YOLO 模型类别指标的格式化字符串。
        """
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.

        对预测输出应用非最大抑制。

        Args:
            preds (torch.Tensor): Raw predictions from the model. 模型的原始预测。

        Returns:
            (List[torch.Tensor]): Processed predictions after NMS. NMS 后的处理预测。
        """
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )

    def _prepare_batch(self, si, batch):
        """
        Prepare a batch of images and annotations for validation.

        准备一批图像和注释以进行验证。

        Args:
            si (int): Batch index. 批次索引。
            batch (Dict): Batch data containing images and annotations. 包含图像和注释的批次数据。

        Returns:
            (Dict): Prepared batch with processed annotations. 处理后的注释的准备批次。
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """
        Prepare predictions for evaluation against ground truth.

        准备预测以与真实标签进行评估。

        Args:
            pred (torch.Tensor): Model predictions. 模型预测。
            pbatch (Dict): Prepared batch information. 准备的批次信息。

        Returns:
            (torch.Tensor): Prepared predictions in native space. 本地空间中的准备预测。
        """
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred - 本地空间预测
        return predn

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth.

        使用新的预测和真实标签更新指标。

        Args:
            preds (List[torch.Tensor]): List of predictions from the model. 模型的预测列表。
            batch (Dict): Batch data containing ground truth. 包含真实标签的批次数据。
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
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

            # Predictions - 预测
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate - 评估
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save - 保存
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def finalize_metrics(self, *args, **kwargs):
        """
        Set final values for metrics speed and confusion matrix.

        设置指标速度和混淆矩阵的最终值。

        Args:
            *args (Any): Variable length argument list. 可变长度参数列表。
            **kwargs (Any): Arbitrary keyword arguments. 任意关键字参数。
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """
        Calculate and return metrics statistics.

        计算并返回指标统计信息。

        Returns:
            (Dict): Dictionary containing metrics results. 包含指标结果的字典。
        """
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats):
            self.metrics.process(**stats, on_plot=self.on_plot)
        return self.metrics.results_dict

    def print_results(self):
        """
        Print training/validation set metrics per class.

        打印每个类别的训练/验证集指标。
        """
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format - 打印格式
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class - 按类别打印结果
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        返回正确的预测矩阵。

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
                表示检测结果的张量，形状为 (N, 6)。每行的格式为 [x1, y1, x2, y2, conf, class]。
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
                表示真实边界框坐标的张量，形状为 (M, 4)。每行的格式为 [x1, y1, x2, y2]。
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.
                表示目标类别索引的张量，形状为 (M,)。

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels. Each row represents the
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        构建 YOLO 数据集。

        Args:
            img_path (str): Path to the folder containing images. 包含图像的文件夹路径。
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode. `train` 模式或 `val` 模式，用户可以为每种模式自定义不同的增强。
            batch (int, optional): Size of batches, this is for `rect`. 批次大小，用于 `rect`。

        Returns:
            (Dataset): YOLO dataset. YOLO 数据集。
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """
        Construct and return dataloader.

        构建并返回数据加载器。

        Args:
            dataset_path (str): Path to the dataset. 数据集路径。
            batch_size (int): Size of each batch. 每批次的图像数量。

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation. 用于验证的数据加载器。
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader - 返回数据加载器

    def plot_val_samples(self, batch, ni):
        """
        Plot validation image samples.

        绘制验证图像样本。

        Args:
            batch (Dict): Batch containing images and annotations. 包含图像和注释的批次。
            ni (int): Batch index. 批次索引。
        """
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        Plot predicted bounding boxes on input images and save the result.

        在输入图像上绘制预测的边界框并保存结果。

        Args:
            batch (Dict): Batch containing images and annotations. 包含图像和注释的批次。
            preds (List[torch.Tensor]): List of predictions from the model. 模型的预测列表。
            ni (int): Batch index. 批次索引。
        """
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        将 YOLO 检测结果保存到特定格式的 txt 文件中，以归一化坐标表示。

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class). 预测结果的格式为 (x1, y1, x2, y2, conf, class)。
            save_conf (bool): Whether to save confidence scores. 是否保存置信度分数。
            shape (tuple): Shape of the original image. 原始图像的形状。
            file (Path): File path to save the detections. 保存检测结果的文件路径。
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """
        Serialize YOLO predictions to COCO json format.

        将 YOLO 预测序列化为 COCO JSON 格式。

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class). 预测结果的格式为 (x1, y1, x2, y2, conf, class)。
            filename (str): Image filename. 图像文件名。
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner - xy 中心到左上角
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """
        Evaluate YOLO output in JSON format and return performance statistics.

        评估 JSON 格式的 YOLO 输出并返回性能统计信息。

        Args:
            stats (Dict): Current statistics dictionary. 当前统计信息字典。

        Returns:
            (Dict): Updated statistics dictionary with COCO/LVIS evaluation results. 带有 COCO/LVIS 评估结果的更新统计信息字典。
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions - 预测
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations - 注释
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api - 初始化注释 API
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path) - 初始化预测 API（必须传递字符串，而不是 Path）
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api - 初始化注释 API
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path) - 初始化预测 API（必须传递字符串，而不是 Path）
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval - 要评估的图像
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results - 显式调用 print_results
                # update mAP50-95 and mAP50 - 更新 mAP50-95 和 mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
