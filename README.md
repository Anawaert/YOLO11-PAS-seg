## YOLO11-PAS-seg

### 效果示意图 <br/> Effect Demonstration

![Segmentation Result 1](https://github.com/Anawaert/YOLO11-PAS-seg/blob/master/images/seg_prediction_1.jpg)

![Segmentation Result 2](https://github.com/Anawaert/YOLO11-PAS-seg/blob/master/images/seg_prediction_2.png)

### 快速开始 <br/> Quick Start
#### 检查系统 `Python` 版本 <br/> Check `Python` Version
在终端中，运行以下命令来检查当前使用的 `Python` 解释器版本：
```bash
python --version
# 或者
python3 --version
```

对于 YOLO11-PAS-seg，我们需要 `Python` 3.8 或更高。我们强烈建议使用 venv 或 conda 来创建一个新的虚拟环境，从而更好地管理依赖项。

In the terminal, run the following command to check the current `Python` interpreter version:
```bash
python --version
# or
python3 --version
```

For YOLO11-PAS-seg, we need `Python` 3.8 or higher. We strongly recommend using venv or conda to create a new virtual environment for better dependency management.

#### 检查 Git 与 pip 包管理器的安装 <br/> Check Git and pip Package Manager Installation
在终端中，运行以下命令来检查当前使用的 Git 和 pip 包管理器的版本：
```bash
git --version
pip --version
```

若您尚未安装 Git 和 pip 包管理器，请根据 [Git 安装指引](https://git-scm.com/downloads) 与 [pip 文档](https://pip.pypa.io/en/stable/installation/) 来安装它们。

In the terminal, run the following command to check the current Git and pip package manager versions:
```bash
git --version
pip --version
```

If you haven't installed Git and pip package manager yet, please refer to the [Git Installation Guide](https://git-scm.com/downloads) and [pip documentation](https://pip.pypa.io/en/stable/installation/) to install them.

#### 安装 YOLO11-PAS-seg <br/> Install YOLO11-PAS-seg
YOLO11-PAS-seg 基于 Ultralytics YOLO11 进行开发，因此安装方式与 Ultralytics 类似。

使用以下命令来克隆 YOLO11-PAS-seg 仓库：
```bash
git clone https://github.com/Anawaert/YOLO11-PAS-seg.git
```

切换到 YOLO11-PAS-seg 目录：
```bash
cd ./YOLO11-PAS-seg
```

然后，使用以下命令来安装 YOLO11-PAS-seg。若您期望以可编辑的方式安装 YOLO11-PAS-seg（例如希望在本地修改代码。当然，这需要在环境变量或 IDE 中配置包的路径），可以使用带 `-e` 参数的安装命令：
```bash
pip install .

# 可编辑模式下安装可使用：
pip install -e .
```

pip 包管理器会自动依据 `pyproject.toml` 文件中的依赖项来安装 YOLO11-PAS-seg 所需的所有依赖。但是，我们强烈建议您手动配置 PyTorch 与 TorchVision，以确保它们能与特定的 CUDA 版本兼容。

YOLO11-PAS-seg is developed based on Ultralytics YOLO11, so the installation method is similar to Ultralytics.

Use the following command to clone the YOLO11-PAS-seg repository:
```bash
git clone https://github.com/Anawaert/YOLO11-PAS-seg.git
```

Switch to the YOLO11-PAS-seg directory:
```bash
cd ./YOLO11-PAS-seg
```

Then, use the following command to install YOLO11-PAS-seg. If you expect to install YOLO11-PAS-seg in editable mode (for example, if you want to modify the code locally, this requires configuring the package path in environment variables or IDE), you can use the installation command with the `-e` parameter:
```bash
pip install .

# Editable mode installation can use:
pip install -e .
```

pip package manager will automatically install all the dependencies required by YOLO11-PAS-seg based on the dependencies in the `pyproject.toml` file. However, we strongly recommend that you manually configure PyTorch and TorchVision to ensure compatibility with specific CUDA versions.

### YOLO11-PAS-seg 的使用 <br/> Usage of YOLO11-PAS-seg
YOLO11-PAS-seg 的使用方式与 Ultralytics YOLO11 基本相同，可以参考 [Ultralytics 文档](https://docs.ultralytics.com/zh) 来使用各种功能，包括训练、推理与导出等。

请注意，由于 YOLO11-PAS-seg 的网络架构具有一些客制化的改动，因此在使用原版 Ultralytics 进行 YOLO11-PAS-seg 模型的训练及推理时会出现错误。但是，YOLO11-PAS-seg 基于 8.3.93 版本的 Ultralytics 进行修改，因此您可以使用 YOLO11-PAS-seg 来训练和推理绝大多数兼容 Ultralytics 的 YOLO 模型。

YOLO11-PAS-seg is used in a similar way to Ultralytics YOLO11, and you can refer to the [Ultralytics Documentation](https://docs.ultralytics.com/zh) for various features, including training, inference, and export.

Please note that due to some customized changes in the network architecture of YOLO11-PAS-seg, using the original Ultralytics to train and infer YOLO11-PAS-seg models will result in errors. However, YOLO11-PAS-seg is modified based on version 8.3.93 of Ultralytics, so you can use YOLO11-PAS-seg to train and infer most YOLO models compatible with Ultralytics.

### YOLO11-PAS-seg 的改变 <br/> Changes in YOLO11-PAS-seg
#### C3k2Ghost 模块 <br/> C3k2Ghost Module
YOLO11-PAS-seg 新增了一个名为 C3k2Ghost 的模块。该模块是一个轻量级的卷积块，具有更少的参数和更低的计算复杂度。C3k2Ghost 模块的设计灵感来自于 GhostNet 和 YOLO11 中的 C3k2 模块。它通过使用深度可分离卷积和 Ghost 模块来减少参数数量和计算复杂度，同时保持较高的性能。C3k2Ghost 模块在 YOLO11-PAS-seg 中被用作主干网络的一部分，以提高模型的效率和性能。

YOLO11-PAS-seg has introduced a new module called C3k2Ghost. This module is a lightweight convolutional block with fewer parameters and lower computational complexity. The design of the C3k2Ghost module is inspired by GhostNet and the C3k2 module in YOLO11. It reduces the number of parameters and computational complexity by using depthwise separable convolutions and Ghost modules while maintaining high performance. The C3k2Ghost module is used as part of the backbone network in YOLO11-PAS-seg to improve the efficiency and performance of the model.

#### 通道-坐标注意力机制 (CCAM) <br/> Convolutional Coordinate Attention Module (CCAM)
为了增强 YOLO11-PAS-seg 的特征提取能力，我们在主干网络中引入了通道-坐标注意力机制 (CCAM)。CCAM 是一种较为轻量级的注意力机制，设计灵感源于 CBAM (Convolutional Block Attention Module) 中的 CAM (Convolutional Attention Module) 和 CA (Convolutional Attention) 模块。CCAM 通过对输入特征图进行通道和空间注意力加权，来增强特征表示能力。

In order to enhance the feature extraction capability of YOLO11-PAS-seg, we introduced the Convolutional Coordinate Attention Module (CCAM) in the backbone network. CCAM is a relatively lightweight attention mechanism, inspired by the CAM (Convolutional Attention Module) and CA (Convolutional Attention) modules in CBAM (Convolutional Block Attention Module). CCAM enhances feature representation capability by applying channel and spatial attention weighting to the input feature map.

#### 双向特征金字塔网络 (BiFPN)  <br/> Bidirectional Feature Pyramid Network (BiFPN)
在 Neck 部分，我们使用了双向特征金字塔网络 (BiFPN) 来增强特征融合能力。BiFPN 是一种轻量级的特征金字塔网络，设计灵感源于 PANet (Path Aggregation Network) 和 FPN (Feature Pyramid Network)。BiFPN 通过双向连接和加权融合来增强特征图之间的信息流动，从而提高模型的性能。BiFPN 的实现在 `/ultralytics/models/cfg/11/yolo11-pas-seg.yaml` 文件中有完整的体现，并在 `/ultralytics/models/cfg/11/yolo11-pas-seg.yaml` 文件中单独列出了其配置方法。

In the Neck section, we used the Bidirectional Feature Pyramid Network (BiFPN) to enhance feature fusion capability. BiFPN is a lightweight feature pyramid network, inspired by PANet (Path Aggregation Network) and FPN (Feature Pyramid Network). BiFPN enhances the information flow between feature maps through bidirectional connections and weighted fusion, thereby improving the model's performance. The implementation of BiFPN is fully reflected in the `/ultralytics/models/cfg/11/yolo11-pas-seg.yaml` file, and its configuration method is listed separately in the `/ultralytics/models/cfg/11/yolo11-pas-seg.yaml` file.

### YOLO11-PAS-seg 在辣椒数据集上的表现 <br/> YOLO11-PAS-seg Performance on Pepper Dataset
由于 YOLO11-PAS-seg 的设计初衷是用于辣椒数据集的分割任务，因此在专有的自定义辣椒数据集上进行了实验。我们分别使用具有 C3k2Ghost、CCAM 和 BiFPN 的 YOLO11-PAS-seg 模型进行训练，并与原版 YOLO11m-seg 进行了对比。实验结果如下：

| C3k2Ghost | CCAM | BiFPN | $P$ | $R$ | ${mAP}_{50}$| ${mAP}_{50:95}$ | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| √ | √ | √ | **0.857** | **0.821** | **0.867** | **0.661** | **20.662 M** |
| √ | √ | × | **0.861** | 0.787 | 0.862 | 0.663 | **19.173 M** |
| √ | × | √ | 0.847 | 0.794 | 0.853 | 0.661 | **19.436 M** |
| × | √ | √ | 0.859 | **0.823** | 0.870 | **0.667** | 23.873 M |
| √ | × | × | 0.859 | 0.773 | 0.851 | 0.656 | **19.151 M** |
| × | √ | × | **0.885** | 0.779 | 0.870 | **0.667** | 23.587 M |
| × | × | √ | 0.822 | **0.820** | **0.875** | 0.665 | 22.600 M |
| × | × | × | 0.853 | 0.771 | 0.860 | 0.646 | 22.338 M |

Due to the original intention of YOLO11-PAS-seg being used for segmentation tasks on the pepper dataset, experiments were conducted on a proprietary custom pepper dataset. We trained YOLO11-PAS-seg models with C3k2Ghost, CCAM, and BiFPN, and compared them with the original YOLO11m-seg. The upon table shows the experimental results.


### 另请参阅
[Ultralytics 文档 <br/> Ultralytics Documentation](https://docs.ultralytics.com/zh)

[YOLOv8 的 BiFPN 改进示例 <br/> YOLOv8_BiFPN](https://github.com/Changping-Li/YOLOv8_BiFPN)

[YOLOv8 的 CA 改进示例 <br/> yolov8-with-coordinate_attention](https://github.com/easyssun/yolov8-with-coordinate_attention)

[CBAM 的 PyTorch 实现实例 <br/> CBAM.PyTorch](https://github.com/luuuyi/CBAM.PyTorch)

[YOLOv5 的 Ghost 模块改进示例 <br/> YOLOv5-Ghost](https://github.com/changhaochen-98/YOLOv5-Ghost)

[Anawaert Blog](https://blog.anawaert.tech/)