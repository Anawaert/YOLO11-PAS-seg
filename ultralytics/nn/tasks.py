# YOLO11-PAS-seg - https://github.com/Anawaert/YOLO11-PAS-seg

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect,
    BiFPNCat2,
    BiFPNCat3,
    CCAM,
    C3k2Ghost,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None  # conda support without 'ultralytics-thop' installed - conda 支持，无需安装'ultralytics-thop'


class BaseModel(torch.nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.

    基类 BaseModel 用于所有 Ultralytics YOLO 系列模型。
    """

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        执行模型的前向传递，用于训练或推理。

        如果 x 是一个字典，则计算并返回训练的损失。否则，返回推理的预测结果。

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training. 用于推理的输入张量，或包含图像张量和标签的字典用于训练。
            *args (Any): Variable length argument list. 可变长度参数列表。
            **kwargs (Any): Arbitrary keyword arguments. 任意关键字参数。

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference). 如果 x 是一个字典（训练），则返回损失；否则返回网络预测结果。
        """
        if isinstance(x, dict):  # for cases of training and validating while training. - 用于训练和验证的情况。
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        执行网络的前向传递。

        Args:
            x (torch.Tensor): The input tensor to the model. 模型的输入张量。
            profile (bool): Print the computation time of each layer if True. 如果为 True，则打印每一层的计算时间。
            visualize (bool): Save the feature maps of the model if True. 如果为 True，则保存模型的特征图。
            augment (bool): Augment image during prediction. 在预测期间增强图像。
            embed (List, optional): A list of feature vectors/embeddings to return. 要返回的特征向量/嵌入的列表。

        Returns:
            (torch.Tensor): The last output of the model. 模型的最后输出。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        执行网络的前向传递。

        Args:
            x (torch.Tensor): The input tensor to the model. 模型的输入张量。
            profile (bool): Print the computation time of each layer if True. 如果为 True，则打印每一层的计算时间。
            visualize (bool): Save the feature maps of the model if True. 如果为 True，则保存模型的特征图。
            embed (List, optional): A list of feature vectors/embeddings to return. 要返回的特征向量/嵌入的列表。

        Returns:
            (torch.Tensor): The last output of the model. 模型的最后输出。
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer - 如果不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers - 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output - 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten - 展平
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference.

        对输入图像 x 执行增强，并返回增强的推理结果。
        """
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.

        在给定输入上对模型的单个层的计算时间和 FLOPs 进行分析。

        Args:
            m (torch.nn.Module): The layer to be profiled. 要分析的层。
            x (torch.Tensor): The input data to the layer. 层的输入数据。
            dt (List): A list to store the computation time of the layer. 用于存储层的计算时间的列表。
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix - 是最后一层列表，将输入复制为就地修复
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        将模型的 `Conv2d()` 和 `BatchNorm2d()` 层融合为单个层，以提高计算效率。

        Returns:
            (torch.nn.Module): The fused model is returned. 返回融合后的模型。
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv - 更新卷积
                    delattr(m, "bn")  # remove batchnorm - 删除批量归一化
                    m.forward = m.forward_fuse  # update forward - 更新前向传递
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm - 删除批量归一化
                    m.forward = m.forward_fuse  # update forward - 更新前向传递
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward - 更新前向传递
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        检查模型是否具有少于一定阈值的 BatchNorm 层。

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. BatchNorm 层的阈值数量。

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
                如果模型中的 BatchNorm 层的数量小于阈值，则返回 True；否则返回 False。
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d() - 标凃化层，例如 BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model - 如果模型中的 BatchNorm 层小于 'thresh'，则返回 True

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model. 如果为 True，则打印有关模型的详细信息。
            verbose (bool): If True, prints out the model information. 如果为 True，则打印模型信息。
            imgsz (int): The size of the image that the model will be trained on. 模型将在其上训练的图像的大小。
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Apply a function to all tensors in the model that are not parameters or registered buffers.

        将函数应用于模型中不是参数或注册缓冲区的所有张量。

        Args:
            fn (function): The function to apply to the model. 要应用于模型的函数。

        Returns:
            (BaseModel): An updated BaseModel object. 更新后的 BaseModel 对象。
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect - 包括所有 Detect 子类，如 Segment、Pose、OBB、WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load weights into the model.

        将权重加载到模型中。

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded. 要加载的预训练权重。
            verbose (bool, optional): Whether to log the transfer progress. 是否记录传输进度。
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts - torchvision 模型不是字典
        csd = model.float().state_dict()  # checkpoint state_dict as FP32 - 将检查点 state_dict 转换为 FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect - 交叉
        self.load_state_dict(csd, strict=False)  # load - 加载
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        计算损失。

        Args:
            batch (dict): Batch to compute loss on. 用于计算损失的批次。
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions. 预测。
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """
        Initialize the loss criterion for the BaseModel.

        初始化 BaseModel 的损失标准。
        """
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """
    YOLO detection model.

    YOLO 检测模型。
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes - 模型、输入通道、类别数
        """
        Initialize the YOLO detection model with the given config and parameters.

        使用给定的配置和参数初始化 YOLO 检测模型。

        Args:
            cfg (str | dict): Model configuration file path or dictionary. 模型配置文件路径或字典。
            ch (int): Number of input channels. 输入通道数。
            nc (int, optional): Number of classes. 类别数。
            verbose (bool): Whether to display model information. 是否显示模型信息。
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict - 配置字典
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels - 输入通道
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value - 覆盖 YAML 值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist - 模型、保存列表
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict - 默认名称字典
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect - 包括所有 Detect 子类，如 Segment、Pose、OBB、WorldDetect
            s = 256  # 2x min stride - 2x 最小步长
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference and train outputs.

        对输入图像 x 执行增强，并返回增强的推理和训练输出。

        Args:
            x (torch.Tensor): Input image tensor. 输入图像张量。

        Returns:
            (torch.Tensor): Augmented inference output. 增强的推理输出。
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width - 高度、宽度
        s = [1, 0.83, 0.67]  # scales - 比例
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward - 前向传递
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails - 剪辑增强尾部
        return torch.cat(y, -1), None  # augmented inference, train - 增强推理、训练

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        De-scale predictions following augmented inference (inverse operation).

        在增强推理后对预测进行反缩放（逆操作）。

        Args:
            p (torch.Tensor): Predictions tensor. 预测张量。
            flips (int): Flip type (0=none, 2=ud, 3=lr). 翻转类型（0=无，2=ud，3=lr）。
            scale (float): Scale factor. 缩放因子。
            img_size (tuple): Original image size (height, width). 原始图像大小（高度、宽度）。
            dim (int): Dimension to split at. 分割的维度。

        Returns:
            (torch.Tensor): De-scaled predictions. 反缩放的预测。
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud - 反翻转上下
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr - 反翻转左右
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """
        Clip YOLO augmented inference tails.

        剪辑 YOLO 增强推理尾部。

        Args:
            y (List[torch.Tensor]): List of detection tensors. 检测张量列表。

        Returns:
            (List[torch.Tensor]): Clipped detection tensors. 剪辑后的检测张量。
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5) - 检测层的数量（P3-P5）
        g = sum(4**x for x in range(nl))  # grid points - 网格点
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices - 索引
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """
        Initialize the loss criterion for the DetectionModel.

        初始化 DetectionModel 的损失标准。
        """
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """
    YOLO Oriented Bounding Box (OBB) model.

    YOLO 方向边界框（OBB）模型。
    """

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLO OBB model with given config and parameters.

        使用给定的配置和参数初始化 YOLO OBB 模型。

        Args:
            cfg (str | dict): Model configuration file path or dictionary. 模型配置文件路径或字典。
            ch (int): Number of input channels. 输入通道数。
            nc (int, optional): Number of classes. 类别数。
            verbose (bool): Whether to display model information. 是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """
        Initialize the loss criterion for the model.

        初始化模型的损失标准。
        """
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """
    YOLO segmentation model.

    YOLO 分割模型。
    """

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOv8 segmentation model with given config and parameters.

        使用给定的配置和参数初始化 YOLOv8 分割模型。

        Args:
            cfg (str | dict): Model configuration file path or dictionary. 模型配置文件路径或字典。
            ch (int): Number of input channels. 输入通道数。
            nc (int, optional): Number of classes. 类别数。
            verbose (bool): Whether to display model information. 是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """
        Initialize the loss criterion for the SegmentationModel.

        初始化 SegmentationModel 的损失标准。
        """
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLO pose model."""

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """
        Initialize YOLOv8 Pose model.

        初始化 YOLOv8 Pose 模型。

        Args:
            cfg (str | dict): Model configuration file path or dictionary. 模型配置文件路径或字典。
            ch (int): Number of input channels. 输入通道数。
            nc (int, optional): Number of classes. 类别数。
            data_kpt_shape (tuple): Shape of keypoints data. 关键点数据的形状。
            verbose (bool): Whether to display model information. 是否显示模型信息。
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML - 加载模型 YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """
        Initialize the loss criterion for the PoseModel.

        初始化 PoseModel 的损失标准。
        """
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """
    YOLO classification model.

    YOLO 分类模型。
    """

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        使用 YAML、通道、类别数、详细标志初始化 ClassificationModel。

        Args:
            cfg (str | dict): Model configuration file path or dictionary. 模型配置文件路径或字典。
            ch (int): Number of input channels. 输入通道数。
            nc (int, optional): Number of classes. 类别数。
            verbose (bool): Whether to display model information. 是否显示模型信息。
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """
        Set YOLOv8 model configurations and define the model architecture.

        设置 YOLOv8 模型配置并定义模型架构。

        Args:
            cfg (str | dict): Model configuration file path or dictionary. 模型配置文件路径或字典。
            ch (int): Number of input channels. 输入通道数。
            nc (int, optional): Number of classes. 类别数。
            verbose (bool): Whether to display model information. 是否显示模型信息。
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict - 配置字典

        # Define model
        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels - 输入通道
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value - 覆盖 YAML 值
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist - 模型、保存列表
        self.stride = torch.Tensor([1])  # no stride constraints - 没有步长约束
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict - 默认名称字典
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """
        Update a TorchVision classification model to class count 'n' if required.

        如果需要，将 TorchVision 分类模型更新为类计数 'n'。

        Args:
            model (torch.nn.Module): Model to update. 要更新的模型。
            nc (int): New number of classes. 新的类别数。
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module - 最后一个模块
        if isinstance(m, Classify):  # YOLO Classify() head - YOLO Classify() 头
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet - ResNet、EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index - 最后一个 torch.nn.Linear 索引
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index - 最后一个 torch.nn.Conv2d 索引
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """
        Initialize the loss criterion for the ClassificationModel.

        初始化 ClassificationModel 的损失标准。
        """
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    RTDETR（使用 Transformer 实时检测和跟踪）检测模型类。

    该类负责构建 RTDETR 架构、定义损失函数，并促进训练和推理过程。RTDETR 是一个扩展自 DetectionModel 基类的对象检测和跟踪模型。

    Methods:
        init_criterion: Initializes the criterion used for loss calculation. 初始化用于计算损失的标准。
        loss: Computes and returns the loss during training. 计算并返回训练期间的损失。
        predict: Performs a forward pass through the network and returns the output. 执行网络的前向传递并返回输出。
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        初始化 RTDETRDetectionModel。

        Args:
            cfg (str | dict): Configuration file name or path. 配置文件名或路径。
            ch (int): Number of input channels. 输入通道数。
            nc (int, optional): Number of classes. 类别数。
            verbose (bool): Print additional information during initialization. 在初始化期间打印额外信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """
        Initialize the loss criterion for the RTDETRDetectionModel.

        初始化 RTDETRDetectionModel 的损失标准。
        """
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        计算给定数据批次的损失。

        Args:
            batch (dict): Dictionary containing image and label data. 包含图像和标签数据的字典。
            preds (torch.Tensor, optional): Precomputed model predictions. 预先计算的模型预测。

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor. 包含总损失和主要三个损失的张量的元组。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        # 备注：将 gt_bbox 和 gt_labels 预处理为列表。
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        # 备注：RTDETR 中有大约 12 个损失，使用所有损失进行反向传播，但只显示主要三个损失。
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        执行模型的前向传递。

        Args:
            x (torch.Tensor): The input tensor. 输入张量。
            profile (bool): If True, profile the computation time for each layer. 如果为 True，则分析每一层的计算时间。
            visualize (bool): If True, save feature maps for visualization. 如果为 True，则保存特征图以进行可视化。
            batch (dict, optional): Ground truth data for evaluation. 用于评估的真实数据。
            augment (bool): If True, perform data augmentation during inference. 如果为 True，则在推理期间执行数据增强。
            embed (List, optional): A list of feature vectors/embeddings to return. 要返回的特征向量/嵌入的列表。

        Returns:
            (torch.Tensor): Model's output tensor. 模型的输出张量。
        """
        y, dt, embeddings = [], [], []  # outputs - 输出
        for m in self.model[:-1]:  # except the head part - 除了头部部分
            if m.f != -1:  # if not from previous layer - 如果不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers - 来自较早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run - 运行
            y.append(x if m.i in self.save else None)  # save output - 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten - 展平
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference - 头部推理
        return x


class WorldModel(DetectionModel):
    """
    YOLOv8 World Model.

    YOLOv8 World 模型。
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOv8 world model with given config and parameters.

        使用给定的配置和参数初始化 YOLOv8 World 模型。

        Args:
            cfg (str | dict): Model configuration file path or dictionary. 模型配置文件路径或字典。
            ch (int): Number of input channels. 输入通道数。
            nc (int, optional): Number of classes. 类别数。
            verbose (bool): Whether to display model information. 是否显示模型信息。
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder - 特征占位符
        self.clip_model = None  # CLIP model placeholder - CLIP 模型占位符
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        提前设置类别，以便模型可以在没有 clip 模型的情况下进行离线推理。

        Args:
            text (List[str]): List of class names. 类名列表。
            batch (int): Batch size for processing text tokens. 用于处理文本标记的批处理大小。
            cache_clip_model (bool): Whether to cache the CLIP model. 是否缓存 CLIP 模型。
        """
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute - 用于缺少 clip_model 属性的模型的向后兼容性
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        执行模型的前向传递。

        Args:
            x (torch.Tensor): The input tensor. 输入张量。
            profile (bool): If True, profile the computation time for each layer. 如果为 True，则分析每一层的计算时间。
            visualize (bool): If True, save feature maps for visualization. 如果为 True，则保存特征图以进行可视化。
            txt_feats (torch.Tensor, optional): The text features, use it if it's given. 文本特征，如果给定则使用。
            augment (bool): If True, perform data augmentation during inference. 如果为 True，则在推理期间执行数据增强。
            embed (List, optional): A list of feature vectors/embeddings to return. 要返回的特征向量/嵌入的列表。

        Returns:
            (torch.Tensor): Model's output tensor. 模型的输出张量。
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x) or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs - 输出
        for m in self.model:  # except the head part - 除了头部部分
            if m.f != -1:  # if not from previous layer - 如果不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers - 来自较早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run - 运行

            y.append(x if m.i in self.save else None)  # save output - 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten - 展平
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        计算损失。

        Args:
            batch (dict): Batch to compute loss on. 用于计算损失的批次。
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions. 预测。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """
    Ensemble of models.

    模型集合。
    """

    def __init__(self):
        """
        Initialize an ensemble of models.

        初始化模型集合。
        """
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Generate the YOLO network's final layer.

        生成 YOLO 网络的最终层。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。
            augment (bool): Whether to augment the input. 是否对输入进行增强。
            profile (bool): Whether to profile the model. 是否对模型进行分析。
            visualize (bool): Whether to visualize the features. 是否可视化特征。

        Returns:
            (tuple): Tuple containing the concatenated predictions and None. 包含连接的预测和 None 的元组。
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble - 最大集合
        # y = torch.stack(y).mean(0)  # mean ensemble - 均值集合
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C) - nms 集合，y 形状（B，HW，C）
        return y, None  # inference, train output - 推理、训练输出


# Functions ------------------------------------------------------------------------------------------------------------
# 函数 -----------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    用于临时添加或修改 Python 模块缓存（`sys.modules`）中的模块的上下文管理器。

    此函数可用于在运行时更改模块路径。在重构代码时很有用，当您将模块从一个位置移动到另一个位置时，但仍希望支持旧的导入路径以实现向后兼容性。

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths. 一个将旧模块路径映射到新模块路径的字典。
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes. 一个将旧模块属性映射到新模块属性的字典。

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.

        更改仅在上下文管理器内生效，并在上下文管理器退出时撤消。
        请注意，直接操作 `sys.modules` 可能会导致不可预测的结果，特别是在较大的应用程序或库中。请谨慎使用此函数。
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        # 在旧名称下设置 sys.modules 中的属性
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        # 在旧名称下设置 sys.modules 中的模块
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        # 删除临时模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """
    A placeholder class to replace unknown classes during unpickling.

    用于在反序列化期间替换未知类的占位符类。
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize SafeClass instance, ignoring all arguments.

        初始化 SafeClass 实例，忽略所有参数。
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Run SafeClass instance, ignoring all arguments.

        运行 SafeClass 实例，忽略所有参数。
        """
        pass


class SafeUnpickler(pickle.Unpickler):
    """
    Custom Unpickler that replaces unknown classes with SafeClass.

    将未知类替换为 SafeClass 的自定义 Unpickler。
    """

    def find_class(self, module, name):
        """
        Attempt to find a class, returning SafeClass if not among safe modules.

        尝试查找类，如果不在安全模块中，则返回 SafeClass。

        Args:
            module (str): Module name. 模块名称。
            name (str): Class name. 类名。

        Returns:
            (type): Found class or SafeClass. 找到的类或 SafeClass。
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe - 添加其他被认为安全的模块
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    尝试使用 torch.load() 函数加载 PyTorch 模型。如果引发 ModuleNotFoundError 错误，则捕获该错误、记录警告消息，并尝试通过 check_requirements() 函数安装缺少的模块。
    安装后，该函数再次尝试使用 torch.load() 加载模型。

    Args:
        weight (str): The file path of the PyTorch model. PyTorch 模型的文件路径。
        safe_only (bool): If True, replace unknown classes with SafeClass during loading. 如果为 True，则在加载期间将未知类替换为 SafeClass。

    Returns:
        ckpt (dict): The loaded model checkpoint. 加载的模型检查点。
        file (str): The loaded filename. 加载的文件名。

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally - 如果本地缺失，则在线搜索
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                # 通过自定义 pickle 模块加载
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name - e.name 是缺少的模块名称
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module - 安装缺少的模块
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        # 文件可能是使用 torch.save(model, "saved_model.pt") 保存的 YOLO 实例
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """
    Load an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a.

    加载模型权重的集合 weights=[a,b,c] 或单个模型权重 weights=[a] 或 weights=a。

    Args:
        weights (str | List[str]): Model weights path(s). 模型权重路径。
        device (torch.device, optional): Device to load model to. 加载模型的设备。
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model. 是否融合模型。

    Returns:
        (torch.nn.Module): Loaded model. 加载的模型。
    """
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args - 组合参数
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model - FP32 模型

        # Model compatibility updates
        # 模型兼容性更新
        model.args = args  # attach args to model - 将参数附加到模型
        model.pt_path = w  # attach *.pt file path to model - 将 *.pt 文件路径附加到模型
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        # 追加
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode - 模型处于评估模式

    # Module updates
    # 模块更新
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility - torch 1.11.0 兼容性

    # Return model
    # 返回模型
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    # 返回集合
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """
    Load a single model weights.

    加载单个模型权重。

    Args:
        weight (str): Model weight path. 模型权重路径。
        device (torch.device, optional): Device to load model to. 加载模型的设备。
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model. 是否融合模型。

    Returns:
        (tuple): Tuple containing the model and checkpoint. 包含模型和检查点的元组。
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args - 结合模型和默认参数，优先使用模型参数
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model - FP32 模型

    # Model compatibility updates
    # 模型兼容性更新
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model - 将参数附加到模型
    model.pt_path = weight  # attach *.pt file path to model - 将 *.pt 文件路径附加到模型
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode - 模型处于评估模式

    # Module updates
    # 模块更新
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility - torch 1.11.0 兼容性

    # Return model and ckpt
    # 返回模型和 ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3) - 模型字典，输入通道数（3）
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    将 YOLO model.yaml 字典解析为 PyTorch 模型。

    Args:
        d (dict): Model dictionary. 模型字典。
        ch (int): Input channels. 输入通道数。
        verbose (bool): Whether to print model details. 是否打印模型细节。

    Returns:
        (tuple): Tuple containing the PyTorch model and sorted list of output layers. 包含 PyTorch 模型和排序后的输出层列表的元组。
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models - v3/v5/v8/v9 模型的向后兼容性
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU() - 重新定义默认激活函数，例如 Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print - 打印

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out - 层、保存列表、输出通道数
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C3k2Ghost,  # C3k2 with GhostConv - 由 GhostConv 构成的 C3k2
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
            CCAM,  # Attention Module consisted of CAM and CA - 由 CAM 和 CA 组成的注意力模块
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments - 具有 'repeat' 参数的模块
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C3k2Ghost,  # C3k2 with GhostConv - 由 GhostConv 构成的 C3k2
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args - 来自、数量、模块、参数
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain - 深度增益
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output) - 如果 c2 不等于类别数（例如 Classify() 输出）
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads - 设置 1）嵌入通道和 2）头数
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats - 重复次数
                n = 1
            if m is C3k2:  # for M/L/X sizes - 用于 M/L/X 尺寸
                legacy = False
                if scale in "mlx":
                    args[3] = True
            # If using C3k2Ghost, default args[3] = True
            # 若使用 C3k2Ghost，在模型尺度为 M/L/X 时，默认 args[3] = True
            if m is C3k2Ghost:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes - 用于 L/X 尺寸
                    args.extend((True, 1.2))
        elif m is AIFI:
            args = [ch[f], *args]

        # Added BiFPN concatenate modules, which behave like Concat
        # 添加 BiFPN 连接模块，其表现类似于 Concat
        elif m is BiFPNCat2:
            c2 = sum(ch[x] for x in f)
        elif m is BiFPNCat3:
            c2 = sum(ch[x] for x in f)

        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats - 重复次数
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset({Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1 - 特殊情况，通道参数必须在索引 1 中传递
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module - 模块
        t = str(m)[8:-2].replace("__main__.", "")  # module type - 模块类型
        m_.np = sum(x.numel() for x in m_.parameters())  # number params - 参数数量
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type - 附加索引、'from' 索引、类型
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print - 打印
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist - 追加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """
    Load a YOLOv8 model from a YAML file.

    从 YAML 文件加载 YOLOv8 模型。

    Args:
        path (str | Path): Path to the YAML file. YAML 文件的路径。

    Returns:
        (dict): Model dictionary. 模型字典。
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml - 例如 yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict - 模型字典
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Extract the size character n, s, m, l, or x of the model's scale from the model path.

    从模型路径中提取模型规模的大小字符 n、s、m、l 或 x。

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file. YOLO 模型的 YAML 文件的路径。

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x). 模型规模的大小字符（n、s、m、l 或 x）。
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # returns n, s, m, l, or x - 返回 n、s、m、l 或 x
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    从模型的架构或配置中猜测 PyTorch 模型的任务。

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format. PyTorch 模型或 YAML 格式的模型配置。

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb'). 模型的任务（'detect'、'segment'、'classify'、'pose'、'obb'）。
    """

    def cfg2task(cfg):
        """
        Guess from YAML dictionary.

        从 YAML 字典猜测。
        """
        m = cfg["head"][-1][-2].lower()  # output module name - 输出模块名称
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg - 从模型 cfg 猜测
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model - PyTorch 模型
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    # 从模型文件名猜测
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    # 无法从模型确定任务
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect - 假设检测
