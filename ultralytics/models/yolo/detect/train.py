# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models
    for object detection.

    一个扩展了 BaseTrainer 类的类，用于基于检测模型进行训练。

    该训练器专门处理目标检测任务，处理训练 YOLO 模型进行目标检测的特定要求。

    Attributes:
        model (DetectionModel): The YOLO detection model being trained. 正在训练的 YOLO 检测模型。
        data (Dict): Dictionary containing dataset information including class names and number of classes. 包含数据集信息的字典，包括类名和类别数量。
        loss_names (Tuple[str]): Names of the loss components used in training (box_loss, cls_loss, dfl_loss). 用于训练的损失组件的名称（box_loss、cls_loss、dfl_loss）。

    Methods:
        build_dataset: Build YOLO dataset for training or validation. 构建用于训练或验证的 YOLO 数据集。
        get_dataloader: Construct and return dataloader for the specified mode. 构建并返回指定模式的数据加载器。
        preprocess_batch: Preprocess a batch of images by scaling and converting to float. 通过缩放和转换为浮点数对图像批次进行预处理。
        set_model_attributes: Set model attributes based on dataset information. 根据数据集信息设置模型属性。
        get_model: Return a YOLO detection model. 返回一个 YOLO 检测模型。
        get_validator: Return a validator for model evaluation. 返回用于模型评估的验证器。
        label_loss_items: Return a loss dictionary with labeled training loss items. 返回带有标记的训练损失项的损失字典。
        progress_string: Return a formatted string of training progress. 返回训练进度的格式化字符串。
        plot_training_samples: Plot training samples with their annotations. 绘制带有注释的训练样本。
        plot_metrics: Plot metrics from a CSV file. 从 CSV 文件绘制指标。
        plot_training_labels: Create a labeled training plot of the YOLO model. 创建 YOLO 模型的标记训练图。
        auto_batch: Calculate optimal batch size based on model memory requirements. 根据模型内存需求计算最佳批次大小。

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        构建用于训练或验证的 YOLO 数据集。

        Args:
            img_path (str): Path to the folder containing images. 图像文件夹的路径。
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode. `train` 模式或 `val` 模式，用户可以为每种模式定制不同的增强。
            batch (int, optional): Size of batches, this is for `rect`. 批次大小，这是为了 `rect`。

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode. 配置为指定模式的 YOLO 数据集对象。
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader for the specified mode.

        构建并返回指定模式的数据加载器。

        Args:
            dataset_path (str): Path to the dataset. 数据集的路径。
            batch_size (int): Number of images per batch. 每批图像的数量。
            rank (int): Process rank for distributed training. 分布式训练的进程等级。
            mode (str): 'train' for training dataloader, 'val' for validation dataloader. 'train' 用于训练数据加载器，'val' 用于验证数据加载器。

        Returns:
            (DataLoader): PyTorch dataloader object. PyTorch 数据加载器对象。
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP - 如果是 DDP，则仅初始化数据集 *.cache 一次
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader - 返回数据加载器

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images by scaling and converting to float.

        通过缩放和转换为浮点数对图像批次进行预处理。

        Args:
            batch (Dict): Dictionary containing batch data with 'img' tensor. 包含 'img' 张量的批数据字典。

        Returns:
            (Dict): Preprocessed batch with normalized images. 具有归一化图像的预处理批次。
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor - 缩放因子
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple) - 新形状（拉伸到 gs 的倍数）
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """
        Set model attributes based on dataset information.

        根据数据集信息设置模型属性。
        """
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps) - 检测层的数量（用于缩放超参数）
        # self.args.box *= 3 / nl  # scale to layers - 缩放到层
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers - 缩放到类别和层
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers - 缩放到图像大小和层
        self.model.nc = self.data["nc"]  # attach number of classes to model - 将类别数量附加到模型
        self.model.names = self.data["names"]  # attach class names to model - 将类名附加到模型
        self.model.args = self.args  # attach hyperparameters to model - 将超参数附加到模型
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLO detection model.

        返回一个 YOLO 检测模型。

        Args:
            cfg (str, optional): Path to model configuration file. 模型配置文件的路径。
            weights (str, optional): Path to model weights. 模型权重的路径。
            verbose (bool): Whether to display model information. 是否显示模型信息。

        Returns:
            (DetectionModel): YOLO detection model. YOLO 检测模型。
        """
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """

        Return a DetectionValidator for YOLO model validation.

        返回用于 YOLO 模型验证的 DetectionValidator。
        """
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labeled training loss items tensor.

        返回带有标记的训练损失项张量的损失字典。

        Args:
            loss_items (List[float], optional): List of loss values. 损失值列表。
            prefix (str): Prefix for keys in the returned dictionary. 返回的字典中键的前缀。

        Returns:
            (Dict | List): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys. 如果提供了 loss_items，则返回带有标记的损失项字典，否则返回键列表。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats - 将张量转换为 5 位小数浮点数
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """
        Return a formatted string of training progress with epoch, GPU memory, loss, instances and size.

        返回带有 epoch、GPU 内存、损失、实例和大小的训练进度的格式化字符串。
        """
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """
        Plot training samples with their annotations.

        绘制带有注释的训练样本。

        Args:
            batch (Dict): Dictionary containing batch data. 包含批数据的字典。
            ni (int): Number of iterations. 迭代次数。
        """
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """
        Plot metrics from a CSV file.

        从 CSV 文件绘制指标。
        """
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png - 保存 results.png

    def plot_training_labels(self):
        """
        Create a labeled training plot of the YOLO model.

        创建 YOLO 模型的标记训练图。
        """
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        通过计算模型的内存占用来获取最佳批次大小。

        Returns:
            (int): Optimal batch size. 最佳批次大小。
        """
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation - 4 用于马赛克增强
        return super().auto_batch(max_num_obj)
