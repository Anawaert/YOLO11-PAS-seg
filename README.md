## YOLO11-PAS-seg

### Multi-language
[点击此处查看简体中文说明](https://github.com/Anawaert/YOLO11-PAS-seg/blob/master/README_zh.md)

### Introduction
YOLO11-PAS-seg (PAS: Peppers and Stems) is a project developed based on Ultralytics YOLO11, aiming to achieve higher performance with fewer parameters. The YOLO11-PAS-seg model is an improvement over YOLO11m-seg, primarily designed for instance segmentation tasks on peppers. YOLO11-PAS-seg introduces the C3k2Ghost module, Convolutional Coordinate Attention Module (CCAM), and Bidirectional Feature Pyramid Network (BiFPN) to enhance the overall performance and efficiency of the model.

### Effect Demonstration
![Segmentation Result 1](https://github.com/Anawaert/YOLO11-PAS-seg/blob/master/images/seg_prediction_1.jpg)

![Segmentation Result 2](https://github.com/Anawaert/YOLO11-PAS-seg/blob/master/images/seg_prediction_2.png)

### Quick Start
#### Check `Python` Version
In the terminal, run the following command to check the current `Python` interpreter version:
```bash
python --version
# or
python3 --version
```

For YOLO11-PAS-seg, we need `Python` 3.8 or higher. We strongly recommend using venv or conda to create a new virtual environment for better dependency management.

#### Check Git and pip Package Manager Installation
In the terminal, run the following command to check the current Git and pip package manager versions:
```bash
git --version
pip --version
```

If you haven't installed Git and pip package manager yet, please refer to the [Git Installation Guide](https://git-scm.com/downloads) and [pip documentation](https://pip.pypa.io/en/stable/installation/) to install them.

#### Install YOLO11-PAS-seg
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

pip package manager will automatically install all the dependencies required by YOLO11-PAS-seg based on the dependencies in the `pyproject.toml` file. However, **we strongly recommend that you manually configure PyTorch and TorchVision** to ensure compatibility with specific CUDA versions.

### Usage of YOLO11-PAS-seg
YOLO11-PAS-seg is used in a similar way to Ultralytics YOLO11, and you can refer to the [Ultralytics Documentation](https://docs.ultralytics.com/zh) for various features, including training, inference, and export.

Please note that due to some customized changes in the network architecture of YOLO11-PAS-seg, using the original Ultralytics to train and infer YOLO11-PAS-seg models will result in errors. However, YOLO11-PAS-seg is modified based on version 8.3.93 of Ultralytics, so you can use YOLO11-PAS-seg to train and infer most YOLO models compatible with Ultralytics 8.3.93 or earlier version.

### Changes in YOLO11-PAS-seg
#### C3k2Ghost Module
YOLO11-PAS-seg has introduced a new module called C3k2Ghost. This module is a lightweight convolutional block with fewer parameters and lower computational complexity. The design of the C3k2Ghost module is inspired by GhostNet and the C3k2 module in YOLO11. It reduces the number of parameters and computational complexity by using depthwise separable convolutions and Ghost modules while maintaining high performance. The C3k2Ghost module is used as part of the backbone network in YOLO11-PAS-seg to improve the efficiency and performance of the model.

#### Convolutional Coordinate Attention Module (CCAM)
In order to enhance the feature extraction capability of YOLO11-PAS-seg, we introduced the Convolutional Coordinate Attention Module (CCAM) in the backbone network. CCAM is a relatively lightweight attention mechanism, inspired by the CAM (Convolutional Attention Module) and CA (Convolutional Attention) modules in CBAM (Convolutional Block Attention Module). CCAM enhances feature representation capability by applying channel and spatial attention weighting to the input feature map.

#### Bidirectional Feature Pyramid Network (BiFPN)
In the Neck section, we used the Bidirectional Feature Pyramid Network (BiFPN) to enhance feature fusion capability. BiFPN is a lightweight feature pyramid network, inspired by PANet (Path Aggregation Network) and FPN (Feature Pyramid Network). BiFPN enhances the information flow between feature maps through bidirectional connections and weighted fusion, thereby improving the model's performance. The implementation of BiFPN is fully reflected in the `/ultralytics/cfg/models/11/yolo11-pas-seg.yaml` file, and its configuration method is separately listed in the `/ultralytics/cfg/models/11/yolo11-pas-seg-BiFPN.yaml` file.

### YOLO11-PAS-seg Performance on Pepper Dataset
Due to the original intention of YOLO11-PAS-seg being used for segmentation tasks on the pepper dataset, experiments were conducted on a proprietary custom pepper dataset. We trained YOLO11-PAS-seg models with C3k2Ghost, CCAM, and BiFPN, and compared them with the original YOLO11m-seg. The experimental results are as follows:

| C3k2Ghost | CCAM | BiFPN | $P$       | $R$       | ${mAP}_{50}$ | ${mAP}_{50:95}$ | Parameters   |
|-----------|------|-------|-----------|-----------|--------------|-----------------|--------------|
| √         | √    | √     | **0.857** | **0.821** | **0.867**    | **0.661**       | **20.662 M** |
| √         | √    | ×     | **0.861** | 0.787     | 0.862        | 0.663           | **19.173 M** |
| √         | ×    | √     | 0.847     | 0.794     | 0.853        | 0.661           | **19.436 M** |
| ×         | √    | √     | 0.859     | **0.823** | 0.870        | **0.667**       | 23.873 M     |
| √         | ×    | ×     | 0.859     | 0.773     | 0.851        | 0.656           | **19.151 M** |
| ×         | √    | ×     | **0.885** | 0.779     | 0.870        | **0.667**       | 23.587 M     |
| ×         | ×    | √     | 0.822     | **0.820** | **0.875**    | 0.665           | 22.600 M     |
| ×         | ×    | ×     | 0.853     | 0.771     | 0.860        | 0.646           | 22.338 M     |

### Uninstall YOLO11-PAS-seg
If you need to uninstall YOLO11-PAS-seg, you can use the following command:
```bash
pip uninstall YOLO11-PAS-seg
```

### Extra References
[Ultralytics Documentation](https://docs.ultralytics.com/zh)

[YOLOv8_BiFPN](https://github.com/Changping-Li/YOLOv8_BiFPN)

[yolov8-with-coordinate_attention](https://github.com/easyssun/yolov8-with-coordinate_attention)

[CBAM.PyTorch](https://github.com/luuuyi/CBAM.PyTorch)

[YOLOv5-Ghost](https://github.com/changhaochen-98/YOLOv5-Ghost)

[Anawaert Blog](https://blog.anawaert.tech/)