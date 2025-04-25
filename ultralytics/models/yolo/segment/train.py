# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    This trainer specializes in handling segmentation tasks, extending the detection trainer with segmentation-specific
    functionality including model initialization, validation, and visualization.

    一个扩展了 DetectionTrainer 类的类，用于基于分割模型进行训练。

    该训练器专门处理分割任务，扩展了检测训练器，具有分割特定的功能，包括模型初始化、验证和可视化。

    Attributes:
        loss_names (Tuple[str]): Names of the loss components used during training. 训练期间使用的损失组件的名称。

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize a SegmentationTrainer object with given arguments.
        使用给定的参数初始化 SegmentationTrainer 对象。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return SegmentationModel initialized with specified config and weights.
        返回使用指定配置和权重初始化的 SegmentationModel。
        """
        model = SegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """
        Return an instance of SegmentationValidator for validation of YOLO model.
        返回 SegmentationValidator 的实例，用于验证 YOLO 模型。
        """
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """
        Creates a plot of training sample images with labels and box coordinates.
        创建带有标签和框坐标的训练样本图的绘图。
        """
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """
        Plots training/val metrics.
        绘制训练/验证指标。
        """
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
