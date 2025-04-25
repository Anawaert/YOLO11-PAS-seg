# YOLO11-PAS-seg - https://github.com/Anawaert/YOLO11-PAS-seg
"""
Convolution modules.

卷积模块
"""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "BiFPNCat2",  # Channel-wise concatenation of two feature maps - 两个特征图的通道连接
    "BiFPNCat3",  # Channel-wise concatenation of three feature maps - 三个特征图的通道连接
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation - 卷积核、填充、膨胀
    """
    Pad to 'same' shape outputs.

    自动填充到相同形状输出。
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size - 实际卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad - 自动填充
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    标准卷积模块，带有批量归一化和激活。

    Attributes:
        conv (nn.Conv2d): Convolutional layer. 卷积层
        bn (nn.BatchNorm2d): Batch normalization layer. 批量归一化层
        act (nn.Module): Activation function layer. 激活函数层
        default_act (nn.Module): Default activation function (SiLU). 默认激活函数（SiLU）
    """

    default_act = nn.SiLU()  # default activation - 默认激活

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        使用给定参数初始化 Conv 层。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size. 卷积核大小
            s (int): Stride. 步长
            p (int, optional): Padding. 填充
            g (int): Groups. 组
            d (int): Dilation. 膨胀
            act (bool | nn.Module): Activation function. 激活函数
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        将卷积、批量归一化和激活应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        应用卷积和激活，不使用批量归一化。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    简化的 RepConv 模块，带有 Conv 融合。

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer. 主3x3卷积层
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer. 附加1x1卷积层
        bn (nn.BatchNorm2d): Batch normalization layer. 批量归一化层
        act (nn.Module): Activation function layer. 激活函数层
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        使用给定参数初始化 Conv2 层。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size. 卷积核大小
            s (int): Stride. 步长
            p (int, optional): Padding. 填充
            g (int): Groups. 组
            d (int): Dilation. 膨胀
            act (bool | nn.Module): Activation function. 激活函数
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        将卷积、批量归一化和激活应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        将融合卷积、批量归一化和激活应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """
        Fuse parallel convolutions.

        融合并行卷积。
        """
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    1x1 和深度卷积的轻量卷积模块。

    此实现基于 PaddleDetection HGNetV2 骨干网络。

    Attributes:
        conv1 (Conv): 1x1 convolution layer. 1x1 卷积层
        conv2 (DWConv): Depthwise convolution layer. 深度卷积层
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        使用给定参数初始化 LightConv 层。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size for depthwise convolution. 深度卷积的卷积核大小
            act (nn.Module): Activation function. 激活函数
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        将 2 个卷积应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """
    Depth-wise convolution module.

    深度卷积模块。
    """

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        使用给定参数初始化深度卷积。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size. 卷积核大小
            s (int): Stride. 步长
            d (int): Dilation. 膨胀
            act (bool | nn.Module): Activation function. 激活函数
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """
    Depth-wise transpose convolution module.

    深度转置卷积模块。
    """

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        使用给定参数初始化深度转置卷积。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size. 卷积核大小
            s (int): Stride. 步长
            p1 (int): Padding. 填充
            p2 (int): Output padding. 输出填充
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    具有可选批量归一化和激活的卷积转置模块。

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer. 转置卷积层
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer. 批量归一化层
        act (nn.Module): Activation function layer. 激活函数层
        default_act (nn.Module): Default activation function (SiLU). 默认激活函数（SiLU）
    """

    default_act = nn.SiLU()  # default activation - 默认激活

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        使用给定参数初始化 ConvTranspose 层。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size. 卷积核大小
            s (int): Stride. 步长
            p (int): Padding. 填充
            bn (bool): Use batch normalization. 使用批量归一化
            act (bool | nn.Module): Activation function. 激活函数
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        将转置卷积、批量归一化和激活应用于输入。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        将激活和转置卷积操作应用于输入。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    用于集中特征信息的 Focus 模块。

    将输入张量切片为 4 部分，并在通道维度上将它们连接起来。

    Attributes:
        conv (Conv): Convolution layer. 卷积层
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        使用给定参数初始化 Focus 模块。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size. 卷积核大小
            s (int): Stride. 步长
            p (int, optional): Padding. 填充
            g (int): Groups. 组
            act (bool | nn.Module): Activation function. 激活函数
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).

        将 Focus 操作和卷积应用于输入张量。

        输入形状为 (b,c,w,h)，输出形状为 (b,4c,w/2,h/2)。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Ghost Convolution 模块。

    通过使用廉价操作，使用更少的参数生成更多特征的 Ghost Convolution 模块。


    Attributes:
        cv1 (Conv): Primary convolution. 主要卷积
        cv2 (Conv): Cheap operation convolution. 廉价操作卷积

    References:
        https://github.com/huawei-noah/ghostnet
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        使用给定参数初始化 Ghost Convolution 模块。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size. 卷积核大小
            s (int): Stride. 步长
            g (int): Groups. 组
            act (bool | nn.Module): Activation function. 激活函数
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        将 Ghost Convolution 应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor with concatenated features. 具有连接特征的输出张量
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    具有训练和部署模式的 RepConv 模块。

    此模块用于 RT-DETR，并且可以在推理期间融合卷积以提高效率。

    Attributes:
        conv1 (Conv): 3x3 convolution. 3x3 卷积
        conv2 (Conv): 1x1 convolution. 1x1 卷积
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch. 用于身份分支的批量归一化
        act (nn.Module): Activation function. 激活函数
        default_act (nn.Module): Default activation function (SiLU). 默认激活函数（SiLU）

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation - 默认激活

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        使用给定参数初始化 RepConv 模块。

        Args:
            c1 (int): Number of input channels. 输入通道数
            c2 (int): Number of output channels. 输出通道数
            k (int): Kernel size. 卷积核大小
            s (int): Stride. 步长
            p (int): Padding. 填充
            g (int): Groups. 组
            d (int): Dilation. 膨胀
            act (bool | nn.Module): Activation function. 激活函数
            bn (bool): Use batch normalization for identity branch. 用于身份分支的批量归一化
            deploy (bool): Deploy mode for inference. 推理的部署模式
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        部署模式的前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        训练模式的前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Output tensor. 输出张量
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        通过融合卷积计算等效卷积核和偏置。

        Returns:
            (tuple): Tuple containing 元组包含:

                - Equivalent kernel (torch.Tensor) 等效卷积核
                - Equivalent bias (torch.Tensor) 等效偏置
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        将 1x1 卷积核填充到 3x3 大小。

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel. 1x1 卷积核

        Returns:
            (torch.Tensor): Padded 3x3 kernel. 填充的 3x3 卷积核
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        将批量归一化与卷积权重融合。

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse. 要融合的分支

        Returns:
            (tuple): Tuple containing 元组包含:

                - Fused kernel (torch.Tensor) 融合的卷积核
                - Fused bias (torch.Tensor) 融合的偏置
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """
        Fuse convolutions for inference by creating a single equivalent convolution.

        通过创建单个等效卷积来融合卷积以进行推理。
        """
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    用于特征重校准的 Channel-attention 模块。

    根据全局平均池化对通道应用注意力权重。

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling. 全局平均池化
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution. 作为 1x1 卷积实现的全连接层
        act (nn.Sigmoid): Sigmoid activation for attention weights. 用于注意力权重的 Sigmoid 激活

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        初始化 Channel-attention 模块。

        Args:
            channels (int): Number of input channels. 输入通道数
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        将通道注意力应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Channel-attended output tensor. 通道注意力输出张量
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    用于特征重校准的 Spatial-attention 模块。

    根据通道统计信息对空间维度应用注意力权重。

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention. 空间注意力的卷积层
        act (nn.Sigmoid): Sigmoid activation for attention weights. 用于注意力权重的 Sigmoid 激活
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        初始化 Spatial-attention 模块。

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7). 卷积核大小（3 或 7）
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        将空间注意力应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Spatial-attended output tensor. 空间注意力输出张量
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    注意力卷积块模块。

    组合通道和空间注意力机制，进行全面的特征细化。

    Attributes:
        channel_attention (ChannelAttention): Channel attention module. 通道注意力模块
        spatial_attention (SpatialAttention): Spatial attention module. 空间注意力模块
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        使用给定参数初始化 CBAM。

        Args:
            c1 (int): Number of input channels. 输入通道数
            kernel_size (int): Size of the convolutional kernel for spatial attention. 空间注意力的卷积核大小
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        将通道和空间注意力依次应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量

        Returns:
            (torch.Tensor): Attended output tensor. 注意力输出张量
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    沿指定维度连接张量列表。

    Attributes:
        d (int): Dimension along which to concatenate tensors. 沿着哪个维度连接张量
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        初始化 Concat 模块。

        Args:
            dimension (int): Dimension along which to concatenate tensors. 沿着哪个维度连接张量
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension.

        沿指定维度连接输入张量。

        Args:
            x (List[torch.Tensor]): List of input tensors. 输入张量列表

        Returns:
            (torch.Tensor): Concatenated tensor. 连接的张量
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    返回输入的特定索引。

    Attributes:
        index (int): Index to select from input. 从输入中选择的索引
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        初始化 Index 模块。

        Args:
            index (int): Index to select from input. 从输入中选择的索引
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Select and return a particular index from input.

        从输入中选择并返回特定索引。

        Args:
            x (List[torch.Tensor]): List of input tensors. 输入张量列表

        Returns:
            (torch.Tensor): Selected tensor. 选择的张量
        """
        return x[self.index]


class BiFPNCat2(nn.Module):
    """
    BiFPNCat2 module for feature fusion in neural networks, used for two-scale feature extraction.

    This module implements a feature pyramid network (FPN) architecture that combines features from different
    resolutions using a top-down and bottom-up approach. It allows for efficient multi-scale feature extraction
    and fusion, enhancing the model's ability to detect objects at various scales.

    BiFPNCat2 模块用于神经网络中的特征融合，用于两种尺度的特征提取。

    此模块实现了一种特征金字塔网络 (FPN) 架构，使用自上而下和自下而上的方法结合来自不同分辨率的特征。它允许高效的多尺度特征提取和融合，从而增强模型在各种尺度下检测对象的能力。
    """

    def __init__(self, cat_dimension=1):
        """
        Initialize BiFPNCat2 module.

        初始化 BiFPNCat2 模块。

        Args:
            cat_dimension: Dimension along which to concatenate features, amd default is 1. 特征连接的维度，默认为 1。
        """
        super(BiFPNCat2, self).__init__()
        # Initialize the module with the specified concatenation dimension.
        # 使用指定的连接维度初始化模块。
        self.d = cat_dimension
        # Initialize the learnable weights for the concatenation operation.
        # 初始化连接操作的可学习权重。
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # Set the eps value for numerical stability in the softmax operation.
        # 设置软最大值操作的数值稳定性 eps 值。
        self.eps = 0.001

    def forward(self, x):
        """
        Forward pass through the BiFPNCat2 module.

        通过 BiFPNCat2 模块执行前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after concatenation. 连接后的输出张量。
        """
        weight = self.w
        normalised_weight = weight / (torch.sum(weight, dim=0) + self.eps)
        y = [normalised_weight[0] * x[0], normalised_weight[1] * x[1]]
        return torch.cat(y, dim=self.d)  # Concatenate the features along the specified dimension - 沿指定维度连接特征


class BiFPNCat3(nn.Module):
    """
    BiFPNCat3 module for feature fusion in neural networks, used for three-scale feature extraction.

    This module implements a feature pyramid network (FPN) architecture that combines features from different
    resolutions using a top-down and bottom-up approach. It allows for efficient multi-scale feature extraction
    and fusion, enhancing the model's ability to detect objects at various scales.

    BiFPNCat3 模块用于神经网络中的特征融合，用于三种尺度的特征提取。

    此模块实现了一种特征金字塔网络 (FPN) 架构，使用自上而下和自下而上的方法结合来自不同分辨率的特征。它允许高效的多尺度特征提取和融合，从而增强模型在各种尺度下检测对象的能力。
    """

    def __init__(self, cat_dimension=1):
        """
        Initialize BiFPNCat3 module.

        初始化 BiFPNCat3 模块。

        Args:
            cat_dimension: Dimension along which to concatenate features, amd default is 1. 特征连接的维度，默认为 1。
        """
        super(BiFPNCat3, self).__init__()
        # Initialize the module with the specified concatenation dimension.
        # 使用指定的连接维度初始化模块。
        self.d = cat_dimension
        # Initialize the learnable weights for the concatenation operation.
        # 初始化连接操作的可学习权重。
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        # Set the eps value for numerical stability in the softmax operation.
        # 设置软最大值操作的数值稳定性 eps 值。
        self.eps = 0.001

    def forward(self, x):
        """
        Forward pass through the BiFPNCat3 module.

        通过 BiFPNCat3 模块执行前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after concatenation. 连接后的输出张量。
        """
        weight = self.w
        normalised_weight = weight / (torch.sum(weight, dim=0) + self.eps)
        y = [normalised_weight[0] * x[0], normalised_weight[1] * x[1], normalised_weight[2] * x[2]]
        return torch.cat(y, dim=self.d)  # Concatenate the features along the specified dimension - 沿指定维度连接特征

