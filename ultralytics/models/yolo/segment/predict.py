# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    一个扩展了 DetectionPredictor 类的类，用于基于分割模型进行预测。

    该类专门处理分割模型的输出，处理预测结果中的边界框和掩码。

    Attributes:
        args (Dict): Configuration arguments for the predictor. 预测器的配置参数。
        model (torch.nn.Module): The loaded YOLO segmentation model. 加载的 YOLO 分割模型。
        batch (List): Current batch of images being processed. 正在处理的当前图像批次。

    Methods:
        postprocess: Applies non-max suppression and processes detections. 应用非最大抑制并处理检测结果。
        construct_results: Constructs a list of result objects from predictions. 从预测中构造结果对象列表。
        construct_result: Constructs a single result object from a prediction. 从预测中构造单个结果对象。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo11n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the SegmentationPredictor with configuration, overrides, and callbacks.
        使用配置、覆盖和回调初始化 SegmentationPredictor。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """
        Apply non-max suppression and process detections for each image in the input batch.
        对输入批次中的每个图像应用非极大值抑制并处理检测结果。
        """
        # Extract protos - tuple if PyTorch model or array if exported
        # 提取原型 - 如果是 PyTorch 模型则为元组，如果是导出的则为数组
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """
        Construct a list of result objects from the predictions.
        从预测中构造结果对象列表。

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes, scores, and masks. 预测的边界框、分数和掩码列表。
            img (torch.Tensor): The image after preprocessing. 预处理后的图像。
            orig_imgs (List[np.ndarray]): List of original images before preprocessing. 预处理前的原始图像列表。
            protos (List[torch.Tensor]): List of prototype masks. 原型掩码列表。

        Returns:
            (List[Results]): List of result objects containing the original images, image paths, class names,
                bounding boxes, and masks. 包含原始图像、图像路径、类名、边界框和掩码的结果对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """
        Construct a single result object from the prediction.
        从预测中构造单个结果对象。

        Args:
            pred (np.ndarray): The predicted bounding boxes, scores, and masks. 预测的边界框、分数和掩码。
            img (torch.Tensor): The image after preprocessing. 预处理后的图像。
            orig_img (np.ndarray): The original image before preprocessing. 预处理前的原始图像。
            img_path (str): The path to the original image. 原始图像的路径。
            proto (torch.Tensor): The prototype masks. 原型掩码。

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
                包含原始图像、图像路径、类名、边界框和掩码的结果对象。
        """
        if not len(pred):  # save empty boxes - 保存空框
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks - 只保留有掩码的预测
            pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
