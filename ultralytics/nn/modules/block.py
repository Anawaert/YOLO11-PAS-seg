# YOLO11-PAS-seg - https://github.com/Anawaert/YOLO11-PAS-seg
"""
Block modules.

块模块，包含各类常用的卷积块的定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.ma.core import identity

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "CCAM",  # Convolutional Coordinate Attention Module - 通道-坐标注意力模块
    "C3k2Ghost",  # C3k2 module with GhostBottleneck - 具有 GhostBottleneck 的 C3k2 模块
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391

    分布焦点损失 (DFL) 的积分模块。

    提出于 Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """
        Initialize a convolutional layer with a given number of input channels.

        初始化具有给定输入通道数的卷积层。
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """
        Apply the DFL module to input tensor and return transformed output.

        将 DFL 模块应用于输入张量并返回转换后的输出。
        """
        b, _, a = x.shape  # batch, channels, anchors - 批次，通道，锚点
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """
    YOLOv8 mask Proto module for segmentation models.

    YOLOv8 mask Proto 模块，用于分割模型。
    """

    def __init__(self, c1, c_=256, c2=32):
        """
        Initialize the YOLOv8 mask Proto module with specified number of protos and masks.

        使用指定数量的原型和掩码初始化 YOLOv8 mask Proto 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c_ (int): Intermediate channels. 中间通道数。
            c2 (int): Output channels (number of protos). 输出通道数（原型数量）。
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest') - 上采样
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """
        Perform a forward pass through layers using an upsampled input image.

        通过使用上采样的输入图像对层进行前向传递。
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

    PPHGNetV2 的 StemBlock，包含 5 个卷积和一个 maxpool2d。

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """
        Initialize the StemBlock of PPHGNetV2.

        初始化 PPHGNetV2 的 StemBlock。

        Args:
            c1 (int): Input channels. 输入通道数。
            cm (int): Middle channels. 中间通道数。
            c2 (int): Output channels. 输出通道数。
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """
        Forward pass of a PPHGNetV2 backbone layer.

        PPHGNetV2 骨干层的前向传递。
        """
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

    PPHGNetV2 的 HG_Block，包含 2 个卷积和 LightConv。

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """
        Initialize HGBlock with specified parameters.

        使用指定参数初始化 HGBlock。

        Args:
            c1 (int): Input channels. 输入通道数。
            cm (int): Middle channels. 中间通道数。
            c2 (int): Output channels. 输出通道数。
            k (int): Kernel size. 核大小。
            n (int): Number of LightConv or Conv blocks. LightConv 或 Conv 块的数量。
            lightconv (bool): Whether to use LightConv. 是否使用 LightConv。
            shortcut (bool): Whether to use shortcut connection. 是否使用快捷连接。
            act (nn.Module): Activation function. 激活函数。
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv - 挤压卷积
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv - 激励卷积
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of a PPHGNetV2 backbone layer.

        PPHGNetV2 骨干层的前向传递。
        """
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729.

    空间金字塔池化（SPP）层 https://arxiv.org/abs/1406.4729。
    """

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        使用输入/输出通道和池化核大小初始化 SPP 层。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            k (Tuple[int, int, int]): Kernel sizes for max pooling. 最大池化的核大小。
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """
        Forward pass of the SPP layer, performing spatial pyramid pooling.

        SPP 层的前向传递，执行空间金字塔池化。
        """
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.

    用于 YOLOv5 的空间金字塔池化 - 快速（SPPF）层，由 Glenn Jocher 实现。
    """

    def __init__(self, c1, c2, k=5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        使用给定的输入/输出通道和核大小初始化 SPPF 层。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            k (int): Kernel size. 核大小。

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)). It is faster and more memory efficient.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        Apply sequential pooling operations to input and return concatenated feature maps.

        对输入应用顺序池化操作并返回连接的特征图。
        """
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """
    CSP Bottleneck with 1 convolution.

    具有 1 个卷积的 CSP 瓶颈。
    """

    def __init__(self, c1, c2, n=1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        使用 1 个卷积初始化 CSP 瓶颈。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of convolutions. 卷积数量。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """
        Apply convolution and residual connection to input tensor.

        对输入张量应用卷积和残差连接。
        """
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """
    CSP Bottleneck with 2 convolutions.

    具有 2 个卷积的 CSP 瓶颈。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        使用 2 个卷积初始化 CSP 瓶颈。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Bottleneck blocks. 瓶颈块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2) - 可选激活函数
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention() - 通道注意力或空间注意力
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Forward pass through the CSP bottleneck with 2 convolutions.

        通过具有 2 个卷积的 CSP 瓶颈的前向传递。
        """
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions.

    具有 2 个卷积的 CSP 瓶颈的更快实现。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        使用 2 个卷积初始化 CSP 瓶颈。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Bottleneck blocks. 瓶颈块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2) - 可选激活函数
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """
        Forward pass through C2f layer.

        通过 C2f 层的前向传递。
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """
        Forward pass using split() instead of chunk().

        使用 split() 而不是 chunk() 的前向传递。
        """
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """
    CSP Bottleneck with 3 convolutions.

    具有 3 个卷积的 CSP 瓶颈。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        使用 3 个卷积初始化 CSP 瓶颈。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Bottleneck blocks. 瓶颈块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)- 可选激活函数
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Forward pass through the CSP bottleneck with 3 convolutions.

        通过具有 3 个卷积的 CSP 瓶颈的前向传递。
        """
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """
    C3 module with cross-convolutions.

    具有交叉卷积的 C3 模块。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with cross-convolutions.

        使用交叉卷积初始化 C3 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Bottleneck blocks. 瓶颈块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """
    Rep C3.

    重复的 C3。
    """

    def __init__(self, c1, c2, n=3, e=1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        使用单个卷积初始化 CSP 瓶颈。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of RepConv blocks. RepConv 块数量。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """
        Forward pass of RepC3 module.

        RepC3 模块的前向传递。
        """
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """
    C3 module with TransformerBlock().

    具有 TransformerBlock() 的 C3 模块。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with TransformerBlock.

        使用 TransformerBlock 初始化 C3 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Transformer blocks. Transformer 块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """
    C3 module with GhostBottleneck().

    具有 GhostBottleneck() 的 C3 模块。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with GhostBottleneck.

        使用 GhostBottleneck 初始化 C3 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Ghost bottleneck blocks. Ghost 瓶颈块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck https://github.com/huawei-noah/ghostnet.

    Ghost 颈部 https://github.com/huawei-noah/ghostnet.
    """

    def __init__(self, c1, c2, k=3, s=1):
        """
        Initialize Ghost Bottleneck module.

        初始化 Ghost 颈部模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            k (int): Kernel size. 核大小。
            s (int): Stride. 步长。
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """
        Apply skip connection and concatenation to input tensor.

        对输入张量应用跳过连接和连接。
        """
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """
    Standard bottleneck.

    标准颈部。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize a standard bottleneck module.

        初始化标准颈部模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            shortcut (bool): Whether to use shortcut connection. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            k (Tuple[int, int]): Kernel sizes for convolutions. 卷积的核大小。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Apply bottleneck with optional shortcut connection.

        应用具有可选快捷连接的瓶颈。
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks.

    CSP 颈 https://github.com/WongKinYiu/CrossStagePartialNetworks.
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize CSP Bottleneck.

        初始化 CSP 颈。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Bottleneck blocks. 瓶颈块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3) - 应用于 cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Apply CSP bottleneck with 3 convolutions.

        应用具有 3 个卷积的 CSP 颈。
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """
    ResNet block with standard convolution layers.

    具有标准卷积层的 ResNet 块。
    """

    def __init__(self, c1, c2, s=1, e=4):
        """
        Initialize ResNet block.

        初始化 ResNet 块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            s (int): Stride. 步长。
            e (int): Expansion ratio. 扩张率。
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the ResNet block.

        通过 ResNet 块的前向传递。
        """
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """
    ResNet layer with multiple ResNet blocks.

    具有多个 ResNet 块的 ResNet 层。
    """

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """
        Initialize ResNet layer.

        初始化 ResNet 层。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            s (int): Stride. 步长。
            is_first (bool): Whether this is the first layer. 是否为第一层。
            n (int): Number of ResNet blocks. ResNet 块数量。
            e (int): Expansion ratio. 扩张率。
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Forward pass through the ResNet layer.

        通过 ResNet 层的前向传递。
        """
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """
    Max Sigmoid attention block.

    最大 Sigmoid 注意力块。
    """

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """
        Initialize MaxSigmoidAttnBlock.

        初始化 MaxSigmoidAttnBlock。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            nh (int): Number of heads. 头数。
            ec (int): Embedding channels. 嵌入通道数。
            gc (int): Guide channels. 指南通道数。
            scale (bool): Whether to use learnable scale parameter. 是否使用可学习的比例参数。
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """
        Forward pass of MaxSigmoidAttnBlock.

        MaxSigmoidAttnBlock 的前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。
            guide (torch.Tensor): Guide tensor. 指南张量。

        Returns:
            (torch.Tensor): Output tensor after attention. 在注意力后的输出张量。
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """
    C2f module with an additional attn module.

    具有额外 attn 模块的 C2f 模块。
    """

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """
        Initialize C2f module with attention mechanism.

        使用注意机制初始化 C2f 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Bottleneck blocks. 瓶颈块数量。
            ec (int): Embedding channels for attention. 注意力的嵌入通道数。
            nh (int): Number of heads for attention. 注意力的头数。
            gc (int): Guide channels for attention. 注意力的指南通道数。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2) - 可选激活函数
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """
        Forward pass through C2f layer with attention.

        通过具有注意力的 C2f 层的前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。
            guide (torch.Tensor): Guide tensor for attention. 注意力的指南张量。

        Returns:
            (torch.Tensor): Output tensor after processing. 处理后的输出张量。
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """
        Forward pass using split() instead of chunk().

        使用 split() 而不是 chunk() 的前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。
            guide (torch.Tensor): Guide tensor for attention. 注意力的指南张量。

        Returns:
            (torch.Tensor): Output tensor after processing. 处理后的输出张量。
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """
    ImagePoolingAttn: Enhance the text embeddings with image-aware information.

    ImagePoolingAttn：使用感知图像信息增强文本嵌入。
    """

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """
        Initialize ImagePoolingAttn module.

        初始化 ImagePoolingAttn 模块。

        Args:
            ec (int): Embedding channels. 嵌入通道数。
            ch (Tuple): Channel dimensions for feature maps. 特征图的通道维度。
            ct (int): Channel dimension for text embeddings. 文本嵌入的通道维度。
            nh (int): Number of attention heads. 注意力头数。
            k (int): Kernel size for pooling. 池化的核大小。
            scale (bool): Whether to use learnable scale parameter. 是否使用可学习的比例参数。
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """
        Forward pass of ImagePoolingAttn.

        ImagePoolingAttn 的前向传递。

        Args:
            x (List[torch.Tensor]): List of input feature maps. 输入特征图的列表。
            text (torch.Tensor): Text embeddings. 文本嵌入。

        Returns:
            (torch.Tensor): Enhanced text embeddings. 增强的文本嵌入。
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """
    Implements contrastive learning head for region-text similarity in vision-language models.

    为视觉-语言模型中的区域-文本相似性实现对比学习头。
    """

    def __init__(self):
        """
        Initialize ContrastiveHead with region-text similarity parameters.

        使用区域-文本相似性参数初始化 ContrastiveHead。
        """
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        # 备注：使用 -10.0 保持初始 cls 损失与其他损失的一致性
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """
        Forward function of contrastive learning.

        对比学习的前向函数。

        Args:
            x (torch.Tensor): Image features. 图像特征。
            w (torch.Tensor): Text features. 文本特征。

        Returns:
            (torch.Tensor): Similarity scores. 相似性分数。
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization..

    使用批量归一化而不是 l2-归一化的 YOLO-World 的批量归一化对比头。

    Args:
        embed_dims (int): Embed dimensions of text and image features. 文本和图像特征的嵌入维度。
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        初始化 BNContrastiveHead。

        Args:
            embed_dims (int): Embedding dimensions for features. 特征的嵌入维度。
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        # 备注：使用 -10.0 保持初始 cls 损失与其他
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        # 使用 -1.0 更稳定
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """
        Forward function of contrastive learning with batch normalization.

        使用批量归一化的对比学习的前向函数。

        Args:
            x (torch.Tensor): Image features. 图像特征。
            w (torch.Tensor): Text features. 文本特征。

        Returns:
            (torch.Tensor): Similarity scores. 相似性分数。
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """
    Rep bottleneck.

    重复的瓶颈。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize RepBottleneck.

        初始化 RepBottleneck。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            shortcut (bool): Whether to use shortcut connection. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            k (Tuple[int, int]): Kernel sizes for convolutions. 卷积的核大小。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """
    Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction.

    用于高效特征提取的可重复交叉阶段部分网络 (RepCSP) 模块。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize RepCSP layer.

        初始化 RepCSP 层。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of RepBottleneck blocks. RepBottleneck 块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """
    CSP-ELAN.
    """

    def __init__(self, c1, c2, c3, c4, n=1):
        """
        Initialize CSP-ELAN layer.

        初始化 CSP-ELAN 层。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            c3 (int): Intermediate channels. 中间通道数。
            c4 (int): Intermediate channels for RepCSP. RepCSP 的中间通道数。
            n (int): Number of RepCSP blocks. RepCSP 块数量。
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """
        Forward pass through RepNCSPELAN4 layer.

        通过 RepNCSPELAN4 层的前向传递。
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """
        Forward pass using split() instead of chunk().

        使用 split() 而不是 chunk() 的前向传递。
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """
    ELAN1 module with 4 convolutions.

    具有 4 个卷积的 ELAN1 模块。
    """

    def __init__(self, c1, c2, c3, c4):
        """
        Initialize ELAN1 layer.

        初始化 ELAN1 层。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            c3 (int): Intermediate channels. 中间通道数。
            c4 (int): Intermediate channels for convolutions. 卷积的中间通道数。
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """
    AConv.
    """

    def __init__(self, c1, c2):
        """
        Initialize AConv module.

        初始化 AConv 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """
        Forward pass through AConv layer.

        通过 AConv 层的前向传递。
        """
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """
        Initialize ADown module.

        初始化 ADown 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """
        Forward pass through ADown layer.

        通过 ADown 层的前向传递。
        """
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """
        Initialize SPP-ELAN block.

        初始化 SPP-ELAN 块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            c3 (int): Intermediate channels. 中间通道数。
            k (int): Kernel size for max pooling. 最大池化的核大小。
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """
        Forward pass through SPPELAN layer.

        通过 SPPELAN 层的前向传递。
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """
        Initialize CBLinear module.

        初始化 CBLinear 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2s (List[int]): List of output channel sizes. 输出通道大小的列表。
            k (int): Kernel size. 核大小。
            s (int): Stride. 步长。
            p (int | None): Padding. 填充。
            g (int): Groups. 组数。
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """
        Forward pass through CBLinear layer.

        通过 CBLinear 层的前向传递。
        """
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """
        Initialize CBFuse module.

        初始化 CBFuse 模块。

        Args:
            idx (List[int]): Indices for feature selection. 特征选择的索引。
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """
        Forward pass through CBFuse layer.

        通过 CBFuse 层的前向传递。

        Args:
            xs (List[torch.Tensor]): List of input tensors. 输入张量的列表。

        Returns:
            (torch.Tensor): Fused output tensor. 融合后的输出张量。
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions.

    具有 2 个卷积的 CSP 瓶颈的更快实现。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        使用两个卷积初始化 CSP 颈层。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Bottleneck blocks. 瓶颈块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2) - 可选激活函数
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """
        Forward pass through C3f layer.

        通过 C3f 层的前向传递。
        """
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions.

    具有 2 个卷积的 CSP 瓶颈的更快实现。
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module.

        初始化 C3k2 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of blocks. 块数量。
            c3k (bool): Whether to use C3k blocks. 是否使用 C3k 块。
            e (float): Expansion ratio. 扩张率。
            g (int): Groups for convolutions. 卷积的组数。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """
    C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks.

    C3k 是具有可自定义核大小的 CSP 瓶颈模块，用于神经网络中的特征提取。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        初始化 C3k 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of Bottleneck blocks. 瓶颈块数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
            k (int): Kernel size. 核大小。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels -
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """
    RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture.

    RepVGGDW 是 RepVGG 架构中表示深度可分离卷积块的类。
    """

    def __init__(self, ed) -> None:
        """
        Initialize RepVGGDW module.

        初始化 RepVGGDW 模块。

        Args:
            ed (int): Input and output channels. 输入和输出通道数。
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Perform a forward pass of the RepVGGDW block.

        执行 RepVGGDW 块的前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution. 应用深度可分禀卷积后的输出张量。
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        执行 RepVGGDW 块的前向传递，而不融合卷积。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution. 应用深度可分禀卷积后的输出张量。
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.

        融合 RepVGGDW 块中的卷积层。

        此方法融合卷积层并相应地更新权重和偏置。
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    条件身份块 (CIB) 模块。

    Args:
        c1 (int): Number of input channels. 输入通道数。
        c2 (int): Number of output channels. 输出通道数。
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True. 是否添加快捷连接。
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5. 隐藏通道的缩放因子。
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False. 是否使用 RepVGGDW 作为第三个卷积层。
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """
        Initialize the CIB module.

        初始化 CIB 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            shortcut (bool): Whether to use shortcut connection. 是否使用快捷连接。
            e (float): Expansion ratio. 扩张率。
            lk (bool): Whether to use RepVGGDW. 是否使用 RepVGGDW。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        CIB 模块的前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor. 输出张量。
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    C2fCIB 类表示具有 C2f 和 CIB 模块的卷积块。

    Args:
        c1 (int): Number of input channels. 输入通道数。
        c2 (int): Number of output channels. 输出通道数。
        n (int, optional): Number of CIB modules to stack. Defaults to 1. 要堆叠的 CIB 模块数量。
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False. 是否使用快捷连接。
        lk (bool, optional): Whether to use local key connection. Defaults to False. 是否使用本地键连接。
        g (int, optional): Number of groups for grouped convolution. Defaults to 1. 分组卷积的组数。
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5. CIB 模块的扩张率。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """
        Initialize C2fCIB module.

        初始化 C2fCIB 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of CIB modules. CIB 模块数量。
            shortcut (bool): Whether to use shortcut connection. 是否使用快捷连接。
            lk (bool): Whether to use local key connection. 是否使用本地键连接。
            g (int): Groups for convolutions. 卷积的组数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    执行输入张量的自注意力的注意力模块。

    Args:
        dim (int): The input tensor dimension. 输入张量的维度。
        num_heads (int): The number of attention heads. 注意力头的数量。
        attn_ratio (float): The ratio of the attention key dimension to the head dimension. 注意键维度与头维度的比率。

    Attributes:
        num_heads (int): The number of attention heads. 注意力头的数量。
        head_dim (int): The dimension of each attention head. 每个注意力头的维度。
        key_dim (int): The dimension of the attention key. 注意键的维度。
        scale (float): The scaling factor for the attention scores. 注意力分数的缩放因子。
        qkv (Conv): Convolutional layer for computing the query, key, and value. 用于计算查询、键和值的卷积层。
        proj (Conv): Convolutional layer for projecting the attended values. 用于投影关注值的卷积层。
        pe (Conv): Convolutional layer for positional encoding. 用于位置编码的卷积层。
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """
        Initialize multi-head attention module.

        初始化多头注意力模块。

        Args:
            dim (int): Input dimension. 输入维度。
            num_heads (int): Number of attention heads. 注意头的数量。
            attn_ratio (float): Attention ratio for key dimension. 注意键维度的注意比率。
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        注意力模块的前向传递。

        Args:
            x (torch.Tensor): The input tensor. 输入张量。

        Returns:
            (torch.Tensor): The output tensor after self-attention. 自注意力后的输出张量。
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    PSABlock 类实现了神经网络的位置敏感注意力块。

    此类封装了应用多头注意力和前馈神经网络层的功能，可选快捷连接。

    Attributes:
        attn (Attention): Multi-head attention module. 多头注意力模块。
        ffn (nn.Sequential): Feed-forward neural network module. 前馈神经网络模块。
        add (bool): Flag indicating whether to add shortcut connections. 指示是否添加快捷连接的标志。

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers. 通过 PSABlock 执行前向传递，应用注意力和前馈层。

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """
        Initialize the PSABlock.

        初始化 PSABlock。

        Args:
            c (int): Input and output channels. 输入和输出通道数。
            attn_ratio (float): Attention ratio for key dimension. 注意键维度的注意比率。
            num_heads (int): Number of attention heads. 注意头的数量。
            shortcut (bool): Whether to use shortcut connections. 是否使用快捷连接。
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """
        Execute a forward pass through PSABlock.

        通过 PSABlock 执行前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing. 经过注意力和前馈处理后的输出张量。
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    PSA 类用于在神经网络中实现位置敏感注意力。

    此类封装了将位置敏感注意力和前馈网络应用于输入张量的功能，增强了特征提取和处理能力。

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution. 应用初始卷积后的隐藏通道数。
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c. 1x1 卷积层，将输入通道数减少到 2*c。
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c. 1x1 卷积层，将输出通道数减少到 c。
        attn (Attention): Attention module for position-sensitive attention. 用于位置敏感注意力的注意力模块。
        ffn (nn.Sequential): Feed-forward network for further processing. 用于进一步处理的前馈网络。

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor. 将位置敏感注意力和前馈网络应用于输入张量。

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """
        Initialize PSA module.

        初始化 PSA 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Execute forward pass in PSA module.

        在 PSA 模块中执行前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing. 经过注意力和前馈处理后的输出张量。
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    具有注意机制的 C2PSA 模块，用于增强特征提取和处理。

    此模块实现了具有注意机制的卷积块，以增强特征提取和处理能力。它包括一系列 PSABlock 模块，用于自注意力和前馈操作。

    Attributes:
        c (int): Number of hidden channels. 隐藏通道数。
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c. 1x1 卷积层，将输入通道数减少到 2*c。
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c. 1x1 卷积层，将输出通道数减少到 c。
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations. 用于注意力和前馈操作的 PSABlock 模块的顺序容器。

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
            通过 C2PSA 模块执行前向传递，应用注意力和前馈操作。

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
            这个模块本质上与 PSA 模块相同，但重构以允许堆叠更多的 PSABlock 模块。

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2PSA module.

        初始化 C2PSA 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of PSABlock modules. PSABlock 模块数量。
            e (float): Expansion ratio. 扩张率。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        Process the input tensor through a series of PSA blocks.

        通过一系列 PSA 块处理输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after processing. 处理后的输出张量。
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    使用 PSA 块增强特征提取的 C2fPSA 模块。

    此类通过结合 PSA 块扩展了 C2f 模块，以实现更好的注意机制和特征提取。

    Attributes:
        c (int): Number of hidden channels. 隐藏通道数。
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c. 1x1 卷积层，将输入通道数减少到 2*c。
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c. 1x1 卷积层，将输出通道数减少到 c。
        m (nn.ModuleList): List of PSA blocks for feature extraction. 用于特征提取的 PSA 块列表。

    Methods:
        forward: Performs a forward pass through the C2fPSA module. 通过 C2fPSA 模块执行前向传递。
        forward_split: Performs a forward pass using split() instead of chunk(). 使用 split() 而不是 chunk() 执行前向传递。

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2fPSA module.

        初始化 C2fPSA 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of PSABlock modules. PSABlock 模块数量。
            e (float): Expansion ratio. 扩张率。
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    使用可分离卷积进行下采样的 SCDown 模块。

    该模块使用点卷积和深度卷积的组合进行下采样，有助于有效地减少输入张量的空间维度，同时保持通道信息。

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels. 减少通道数的点卷积层。
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling. 执行空间下采样的深度卷积层。

    Methods:
        forward: Applies the SCDown module to the input tensor. 将 SCDown 模块应用于输入张量。

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """
        Initialize SCDown module.

        初始化 SCDown 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            k (int): Kernel size. 核大小。
            s (int): Stride. 步幅。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Apply convolution and downsampling to the input tensor.

        将卷积和下采样应用于输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Downsampled output tensor. 下采样的输出张量。
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    通过 TorchVision 模块，可以加载任何 torchvision 模型。

    此类提供了一种从 torchvision 库加载模型、选择性加载预训练权重以及通过截断或展开层来自定义模型的方法。

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped. 加载的 torchvision 模型，可能被截断和展开。

    Args:
        model (str): Name of the torchvision model to load. 要加载的 torchvision 模型的名称。
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT". 要加载的预训练权重。默认为 "DEFAULT"。
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
            如果为 True，则将模型展开为一个序列，其中包含除最后 `truncate` 层之外的所有层。默认为 True。
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
            如果 `unwrap` 为 True，则从末尾截断的层数。默认为 2。
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
            将中间子模块的输出作为列表返回。默认为 False。
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """
        Load the model and weights from torchvision.

        从 torchvision 加载模型和权重。

        Args:
            model (str): Name of the torchvision model to load. 要加载的 torchvision 模型的名称。
            weights (str): Pre-trained weights to load. Default is "DEFAULT". 要加载的预训练权重。默认为 "DEFAULT"。
            unwrap (bool): Whether to unwrap the model. Default is True. 是否展开模型。默认为 True。
            truncate (int): Number of layers to truncate. Default is 2. 要截断的层数。默认为 2。
            split (bool): Whether to split the output. Default is False. 是否拆分输出。默认为 False。
        """
        import torchvision  # scope for faster 'import ultralytics' - 在更快的 'import ultralytics' 的范围内

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin - 一些模型的第二级，如 EfficientNet、Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the model.

        通过模型执行前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor | List[torch.Tensor]): Output tensor or list of tensors. 输出张量或张量列表。
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    为 YOLO 模型提供高效的注意机制的区域注意力模块。

    该模块实现了一种基于区域的注意力机制，以空间感知方式处理输入特征，特别适用于目标检测任务。

    Attributes:
        area (int): Number of areas the feature map is divided. 被划分的特征图的区域数量。
        num_heads (int): Number of heads into which the attention mechanism is divided. 注意机制被划分的头数。
        head_dim (int): Dimension of each attention head. 每个注意力头的维度。
        qkv (Conv): Convolution layer for computing query, key and value tensors. 用于计算查询、键和值张量的卷积层。
        proj (Conv): Projection convolution layer. 投影卷积层。
        pe (Conv): Position encoding convolution layer. 位置编码卷积层。

    Methods:
        forward: Applies area-attention to input tensor. 将区域注意力应用于输入张量。

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initialize an Area-attention module for YOLO models.

        为 YOLO 模型初始化区域注意力模块。

        Args:
            dim (int): Number of hidden channels. 隐藏通道数。
            num_heads (int): Number of heads into which the attention mechanism is divided. 注意机制被划分的头数。
            area (int): Number of areas the feature map is divided, default is 1. 特征图被划分的区域数量，默认为 1。
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """
        Process the input tensor through the area-attention.

        通过区域注意力处理输入张量。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after area-attention. 区域注意力后的输出张量。
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    用于 YOLO 模型中高效特征提取的区域注意力块模块。

    该模块实现了结合前馈网络的区域注意力机制，用于处理特征图。它使用了一种新颖的基于区域的注意力方法，比传统的自注意力更有效，同时保持了有效性。

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features. 用于处理空间特征的区域注意力模块。
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation. 用于特征转换的多层感知器。

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution. 使用截断正态分布初始化模块权重。
        forward: Applies area-attention and feed-forward processing to input tensor. 将区域注意力和前馈处理应用于输入张量。

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initialize an Area-attention block module.

        初始化区域注意力块模块。

        Args:
            dim (int): Number of input channels. 输入通道数。
            num_heads (int): Number of heads into which the attention mechanism is divided. 注意机制被划分的头数。
            mlp_ratio (float): Expansion ratio for MLP hidden dimension. MLP 隐藏维度的扩张率。
            area (int): Number of areas the feature map is divided. 特征图被划分的区域数量。
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        使用截断正态分布初始化权重。

        Args:
            m (nn.Module): Module to initialize. 要初始化的模块。
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through ABlock.

        通过 ABlock 执行前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing. 经过区域注意力和前馈处理后的输出张量。
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    用于通过基于区域的注意力机制增强特征提取的区域注意力 C2f 模块。

    该模块通过结合区域注意力和 ABlock 层扩展了 C2f 架构，以改进特征处理。它支持区域注意力和标准卷积模式。

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels. 将输入通道数减少到隐藏通道数的初始 1x1 卷积层。
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features. 处理连接特征的最终 1x1 卷积层。
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention. 当使用区域注意力时，用于残差缩放的可学习参数。
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing. 用于特征处理的 ABlock 或 C3k 模块列表。

    Methods:
        forward: Processes input through area-attention or standard convolution pathway. 通过区域注意力或标准卷积路径处理输入。

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Initialize Area-Attention C2f module.

        初始化区域注意力 C2f 模块。

        Args:
            c1 (int): Number of input channels. 输入通道数。
            c2 (int): Number of output channels. 输出通道数。
            n (int): Number of ABlock or C3k modules to stack. 要堆叠的 ABlock 或 C3k 模块数量。
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead. 是否使用区域注意力块。如果为 False，则使用 C3k 块。
            area (int): Number of areas the feature map is divided. 特征图被划分的区域数量。
            residual (bool): Whether to use residual connections with learnable gamma parameter. 是否使用具有可学习 gamma 参数的残差连接。
            mlp_ratio (float): Expansion ratio for MLP hidden dimension. MLP 隐藏维度的扩张率。
            e (float): Channel expansion ratio for hidden channels. 隐藏通道的通道扩张率。
            g (int): Number of groups for grouped convolutions. 分组卷积的组数。
            shortcut (bool): Whether to use shortcut connections in C3k blocks. 是否在 C3k 块中使用快捷连接。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels - 隐藏通道
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """
        Forward pass through A2C2f layer.

        通过 A2C2f 层执行前向传递。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after processing. 处理后的输出张量。
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class C3k2Ghost(C2f):
    def __init__(self, c1, c2, n=1, c3_ghost=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2Ghost module.

        初始化 C3k2Ghost 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            c2 (int): Output channels. 输出通道数。
            n (int): Number of C3k modules to stack. 要堆叠的 C3k 模块数量。
            c3_ghost (bool): Whether to use GhostNet-like convolutions. 是否使用 GhostNet 样式的卷积。
            e (float): Expansion ratio for hidden channels. 隐藏通道的扩张率。
            g (int): Number of groups for grouped convolutions. 分组卷积的组数。
            shortcut (bool): Whether to use shortcut connections in C3k blocks. 是否在 C3k 块中使用快捷连接。
        """
        super().__init__(c1, c2, n=n, e=e, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(
            C3Ghost(self.c, self.c, 2, shortcut, g, e) if c3_ghost else GhostBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class FastSigmoid(nn.Module):
    def __init__(self):
        super(FastSigmoid, self).__init__()
        self.ReLU6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        # Fast Sigmoid approximation, using ReLU6 to avoid overflow
        # 快速 Sigmoid 近似，使用 ReLU6 避免溢出
        return self.ReLU6(x + 3) / 6


class FastSwish(nn.Module):
    def __init__(self):
        super(FastSwish, self).__init__()
        self.f_sigmoid = FastSigmoid()

    def forward(self, x):
        # Fast Swish approximation
        # 快速 Swish 近似
        return x * self.f_sigmoid(x)


class CCAM(nn.Module):
    """
    Convolutional Coordinate Attention Module (CCAM) for enhanced feature extraction.

    CCAM is a combination of Channel Attention Module (CAM) and Coordinate Attention (CA) for improved feature representation,
    particularly in convolutional neural networks. It enhances the model's ability to focus on important features by
    applying attention mechanisms in both channel and spatial dimensions.

    卷积坐标注意力模块（CCAM）用于增强特征提取。

    CCAM 是通道注意力模块（CAM）和坐标注意力（CA）的组合，用于改进特征表示，
    特别是在卷积神经网络中。通过在通道和空间维度上应用注意力机制，它增强了模型关注重要特征的能力。
    """

    def __init__(self, c1, reduction=16):
        """
        Initialize CCAM module.

        初始化 CCAM 模块。

        Args:
            c1 (int): Input channels. 输入通道数。
            reduction (int): Reduction ratio for channel attention. 通道注意力的缩减比例。
        """
        super(CCAM, self).__init__()
        # Initialize the channel attention.
        # 初始化通道注意力
        self.cam_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=1, stride=1, bias=True)
        self.ReLU = nn.Sigmoid()

        # Initialize the coordinate attention.
        # 初始化坐标注意力
        self.ca_avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.ca_avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=mip, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(mip)
        self.act = FastSwish()
        # Height-wise and width-wise convolutional layers
        # 高度和宽度卷积层
        self.conv_h = nn.Conv2d(in_channels=mip, out_channels=c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(in_channels=mip, out_channels=c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass through CCAM module.

        通过 CCAM 模块执行前向传播。

        Args:
            x (torch.Tensor): Input tensor. 输入张量。

        Returns:
            (torch.Tensor): Output tensor after applying CCAM. 应用 CCAM 后的输出张量。
        """
        temp = x

        # Channel Attention (CAM) forward pass
        # 通道注意力（CAM）前向传播
        cam_avg_out = self.ReLU(self.fc1(self.cam_avg_pool(x)))
        x = x * cam_avg_out

        # Coordinate Attention (CA) forward pass
        # 坐标注意力（CA）前向传播
        n, c, h, w = x.size()
        ca_avg_h_out = self.ca_avg_pool_h(x)
        ca_avg_w_out = self.ca_avg_pool_w(x).permute(0, 1, 3, 2)
        # Concatenate along the channel dimension
        # 在通道维度上连接
        y = torch.cat([ca_avg_h_out, ca_avg_w_out], dim=2)
        y = self.conv1(y)
        y = self.BN(y)
        y = self.act(y)
        # Split the output into height and width components
        # 将输出拆分为高度和宽度组件
        ca_h_out, ca_w_out = torch.split(y, [h, w], dim=2)
        ca_w_out = ca_w_out.permute(0, 1, 3, 2)
        # Apply the coordinate attention
        # 应用坐标注意力
        h_attention = self.conv_h(ca_h_out).sigmoid()
        w_attention = self.conv_w(ca_w_out).sigmoid()

        # Apply the attention to the input tensor
        # 将注意力应用于输入张量
        output = temp * h_attention * w_attention

        return output

