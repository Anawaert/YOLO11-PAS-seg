# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Activation modules.

激活函数模块。
"""

import torch
import torch.nn as nn


class AGLU(nn.Module):
    """
    Unified activation function module from https://github.com/kostas1515/AGLU.

    This class implements a parameterized activation function with learnable parameters lambda and kappa.

    统一激活函数模块，来自 https://github.com/kostas1515/AGLU。

    该类实现了一个带有可学习参数 lambda 和 kappa 的参数化激活函数。

    Attributes:
        act (nn.Softplus): Softplus activation function with negative beta. 带有负 beta 的 Softplus 激活函数。
        lambd (nn.Parameter): Learnable lambda parameter initialized with uniform distribution. 初始化为均匀分布的可学习 lambda 参数。
        kappa (nn.Parameter): Learnable kappa parameter initialized with uniform distribution. 初始化为均匀分布的可学习 kappa 参数。
    """

    def __init__(self, device=None, dtype=None) -> None:
        """
        Initialize the Unified activation function with learnable parameters.

        使用可学习参数初始化统一激活函数。
        """
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter - lambda 参数
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter - kappa 参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the Unified activation function.

        计算统一激活函数的前向传播。
        """
        lam = torch.clamp(self.lambd, min=0.0001)  # Clamp lambda to avoid division by zero - 为避免除零，将 lambda 截断
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
