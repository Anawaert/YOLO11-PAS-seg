# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    一个扩展了 BasePredictor 类的类，用于基于检测模型进行预测。

    该预测器专门处理目标检测任务，将模型输出处理为有意义的检测结果，包括边界框和类别预测。

    Attributes:
        args (namespace): Configuration arguments for the predictor. 预测器的配置参数。
        model (nn.Module): The detection model used for inference. 用于推理的检测模型。
        batch (List): Batch of images and metadata for processing. 用于处理的图像和元数据批次。

    Methods:
        postprocess: Process raw model predictions into detection results. 将原始模型预测处理为检测结果。
        construct_results: Build Results objects from processed predictions. 从处理后的预测中构建 Results 对象。
        construct_result: Create a single Result object from a prediction. 从预测中创建单个 Result 对象。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo11n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-processes predictions and returns a list of Results objects.

        后处理预测结果并返回 Results 对象列表。
        """
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list - 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        return self.construct_results(preds, img, orig_imgs, **kwargs)

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

        从模型预测中构建 Results 对象列表。

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores for each image. 每个图像的预测边界框和分数列表。
            img (torch.Tensor): Batch of preprocessed images used for inference. 用于推理的预处理图像批次。
            orig_imgs (List[np.ndarray]): List of original images before preprocessing. 预处理前的原始图像列表。

        Returns:
            (List[Results]): List of Results objects containing detection information for each image. 包含每个图像的检测信息的 Results 对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction.

        从一个图像预测中构建单个 Results 对象。

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections. 具有形状 (N, 6) 的预测边界框和分数，其中 N 是检测数量。
            img (torch.Tensor): Preprocessed image tensor used for inference. 用于推理的预处理图像张量。
            orig_img (np.ndarray): Original image before preprocessing. 预处理前的原始图像。
            img_path (str): Path to the original image file. 原始图像文件的路径。

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes. 包含原始图像、图像路径、类别名称和缩放边界框的 Results 对象。
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
