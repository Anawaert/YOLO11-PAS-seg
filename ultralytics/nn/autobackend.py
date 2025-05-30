# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import ast
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import ARM64, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, PYTHON_VERSION, ROOT, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml, is_rockchip
from ultralytics.utils.downloads import attempt_download_asset, is_url


def check_class_names(names):
    """
    Check class names and convert to dict format if needed.

    检查类名并在需要时转换为字典格式。
    """
    if isinstance(names, list):  # names is a list - names 是一个列表
        names = dict(enumerate(names))  # convert to dict - 转换为字典
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764' - ImageNet类代码
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names - 人类可读的名称
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data=None):
    """
    Applies default class names to an input YAML file or returns numerical class names.

    为输入的YAML文件应用默认类名或返回数字类名。
    """
    if data:
        try:
            return yaml_load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # return default if above errors - 如果以上错误，则返回默认值


class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

    AutoBackend类用于处理动态后端选择，以使用Ultralytics YOLO模型运行推理。

    AutoBackend类旨在为各种推理引擎提供抽象层。 它支持各种格式，每种格式都有特定的命名约定，如下所述：

        Supported Formats and Naming Conventions:
            | Format                | File Suffix       |
            | --------------------- | ----------------- |
            | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Lite       | *.tflite          |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |
            | IMX                   | *_imx_model/      |
            | RKNN                  | *_rknn_model/     |

    Attributes:
        model (torch.nn.Module): The loaded YOLO model. 加载的YOLO模型。
        device (torch.device): The device (CPU or GPU) on which the model is loaded. 模型加载的设备（CPU或GPU）。
        task (str): The type of task the model performs (detect, segment, classify, pose). 模型执行的任务类型（检测、分割、分类、姿势）。
        names (Dict): A dictionary of class names that the model can detect. 模型可以检测的类名字典。
        stride (int): The model stride, typically 32 for YOLO models. 模型步幅，通常为32。
        fp16 (bool): Whether the model uses half-precision (FP16) inference. 模型是否使用半精度（FP16）推理。

    Methods:
        forward: Run inference on an input image. 在输入图像上运行推理。
        from_numpy: Convert numpy array to tensor. 将numpy数组转换为张量。
        warmup: Warm up the model with a dummy input. 使用虚拟输入预热模型。
        _model_type: Determine the model type from file path. 从文件路径确定模型类型。

    Examples:
        >>> model = AutoBackend(weights="yolov8n.pt", device="cuda")
        >>> results = model(img)
    """

    @torch.no_grad()
    def __init__(
        self,
        weights="yolo11n.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=True,
        verbose=True,
    ):
        """
        Initialize the AutoBackend for inference.

        初始化用于推理的AutoBackend。

        Args:
            weights (str | torch.nn.Module): Path to the model weights file or a module instance. Defaults to 'yolo11n.pt'. 模型权重文件的路径或模块实例。默认为'yolo11n.pt'。
            device (torch.device): Device to run the model on. Defaults to CPU. 运行模型的设备。默认为 CPU。
            dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False. 使用 OpenCV DNN 模块进行 ONNX 推理。默认为 False。
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Defaults to None. 包含类名的附加 data.yaml 文件的路径。默认为 None。
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False. 启用半精度推理。仅在特定后端上支持。默认为 False。
            batch (int): Batch-size to assume for inference. Defaults to 1. 推理的批处理大小。默认为 1。
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True. 融合 Conv2D + BatchNorm 层以进行优化。默认为 True。
            verbose (bool): Enable verbose logging. Defaults to True. 启用详细日志记录。默认为 True。
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            mnn,
            ncnn,
            imx,
            rknn,
            triton,
        ) = self._model_type(w)
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu or rknn  # BHWC formats (vs torch BCWH) - BHWC格式（与torch BCWH相比）
        stride = 32  # default stride - 默认步幅
        end2end, dynamic = False, False
        model, metadata, task = None, None, None

        # Set device
        # 设置设备
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA - 使用CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx, paddle]):  # GPU dataloader formats - GPU数据加载器格式
            device = torch.device("cpu")
            cuda = False

        # Download if not local
        # 如果不是本地文件，则下载
        if not (pt or triton or nn_module):
            w = attempt_download_asset(w)

        # In-memory PyTorch model
        # 内存中的 PyTorch 模型
        if nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True

        # PyTorch
        elif pt:
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only - 仅姿势
            stride = max(int(model.stride.max()), 32)  # model stride - 模型步幅
            names = model.module.names if hasattr(model, "module") else model.names  # get class names - 获取类名
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half() - 显式分配给 to()、cpu()、cuda()、half()

        # TorchScript
        elif jit:
            import torchvision  # noqa - https://github.com/ultralytics/ultralytics/pull/19747

            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata - 模型元数据
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict - 加载元数据字典
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

        # ONNX OpenCV DNN
        elif dnn:
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)

        # ONNX Runtime and IMX
        elif onnx or imx:
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            if IS_RASPBERRYPI or IS_JETSON:
                # Fix 'numpy.linalg._umath_linalg' has no attribute '_ilp64' for TF SavedModel on RPi and Jetson
                # 修复树莓派和 Jetson 上的 TF SavedModel 的'numpy.linalg._umath_linalg'没有'_ilp64'属性
                check_requirements("numpy==1.23.5")
            import onnxruntime

            providers = ["CPUExecutionProvider"]
            if cuda:
                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                else:  # Only log warning if CUDA was requested but unavailable - 如果请求了但不可用，则仅记录警告
                    LOGGER.warning("WARNING ⚠️ Failed to start ONNX Runtime with CUDA. Using CPU...")
                    device = torch.device("cpu")
                    cuda = False
            LOGGER.info(f"Using ONNX Runtime {providers[0]}")
            if onnx:
                session = onnxruntime.InferenceSession(w, providers=providers)
            else:
                check_requirements(
                    ["model-compression-toolkit==2.1.1", "sony-custom-layers[torch]==0.2.0", "onnxruntime-extensions"]
                )
                w = next(Path(w).glob("*.onnx"))
                LOGGER.info(f"Loading {w} for ONNX IMX inference...")
                import mct_quantizers as mctq
                from sony_custom_layers.pytorch.object_detection import nms_ort  # noqa

                session = onnxruntime.InferenceSession(
                    w, mctq.get_ort_session_options(), providers=["CPUExecutionProvider"]
                )
                task = "detect"

            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
            dynamic = isinstance(session.get_outputs()[0].shape[0], str)
            fp16 = "float16" in session.get_inputs()[0].type
            if not dynamic:
                io = session.io_binding()
                bindings = []
                for output in session.get_outputs():
                    out_fp16 = "float16" in output.type
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(device)
                    io.bind_output(
                        name=output.name,
                        device_type=device.type,
                        device_id=device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    bindings.append(y_tensor)

        # OpenVINO
        elif xml:
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2024.0.0,!=2025.0.0")
            import openvino as ov

            core = ov.Core()
            w = Path(w)
            if not w.is_file():  # if not *.xml
                w = next(w.glob("*.xml"))  # get *.xml file from *_openvino_model dir - 从 *_openvino_model 目录中获取 *.xml 文件
            ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

            # OpenVINO inference modes are 'LATENCY', 'THROUGHPUT' (not recommended), or 'CUMULATIVE_THROUGHPUT'
            # OpenVINO 推理模式为'LATENCY'、'THROUGHPUT'（不推荐）或'CUMULATIVE_THROUGHPUT'
            inference_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 else "LATENCY"
            LOGGER.info(f"Using OpenVINO {inference_mode} mode for batch={batch} inference...")
            ov_compiled_model = core.compile_model(
                ov_model,
                device_name="AUTO",  # AUTO selects best available device, do not modify - AUTO选择最佳可用设备，不要修改
                config={"PERFORMANCE_HINT": inference_mode},
            )
            input_name = ov_compiled_model.input().get_any_name()
            metadata = w.parent / "metadata.yaml"

        # TensorRT
        elif engine:
            LOGGER.info(f"Loading {w} for TensorRT inference...")

            if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.0"):
                # fix error: `np.bool` was a deprecated alias for the builtin `bool` for JetPack 4 with Python <= 3.8.0
                # 修复错误：`np.bool`是JetPack 4的Python <= 3.8.0的内置`bool`的已弃用别名
                check_requirements("numpy==1.23.5")

            try:
                import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,!=10.1.0")
                import tensorrt as trt  # noqa
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # Read file
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length - 读取元数据长度
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata - 读取元数据
                except UnicodeDecodeError:
                    f.seek(0)  # engine file may lack embedded Ultralytics metadata - 引擎文件可能缺少嵌入的 Ultralytics 元数据
                dla = metadata.get("dla", None)
                if dla is not None:
                    runtime.DLA_core = int(dla)
                model = runtime.deserialize_cuda_engine(f.read())  # read engine - 读取引擎

            # Model context
            try:
                context = model.create_execution_context()
            except Exception as e:  # model is None - 模型为 None
                LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
                raise e

            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below - 默认下面更新
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic - 动态
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size - 如果动态，则这是最大批处理大小

        # CoreML
        elif coreml:
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
            metadata = dict(model.user_defined_metadata)

        # TF SavedModel
        elif saved_model:
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model - assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            metadata = Path(w) / "metadata.yaml"

        # TF GraphDef
        elif pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            from ultralytics.engine.exporter import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wrap frozen graphs for deployment."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped - 包装
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
            try:  # find metadata in SavedModel alongside GraphDef - 在 SavedModel 中找到 GraphDef 旁边的元数据
                metadata = next(Path(w).resolve().parent.rglob(f"{Path(w).stem}_saved_model*/metadata.yaml"))
            except StopIteration:
                pass

        # TFLite or TFLite Edge TPU
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                device = device[3:] if str(device).startswith("tpu") else ":0"
                LOGGER.info(f"Loading {w} on device {device[1:]} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(
                    model_path=w,
                    experimental_delegates=[load_delegate(delegate, options={"device": device})],
                )
                device = "cpu"  # Required, otherwise PyTorch will try to use the wrong device - 否则 PyTorch 将尝试使用错误的设备
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model - 加载 TFLite 模型
            interpreter.allocate_tensors()  # allocate - 分配
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # Load metadata
            try:
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))
            except zipfile.BadZipFile:
                pass

        # TF.js
        elif tfjs:
            raise NotImplementedError("YOLOv8 TF.js inference is not currently supported.")

        # PaddlePaddle
        elif paddle:
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi  # noqa

            w = Path(w)
            if not w.is_file():  # if not *.pdmodel - 如果不是 *.pdmodel
                w = next(w.rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir - 从 *_paddle_model 目录中获取 *.pdmodel 文件
            config = pdi.Config(str(w), str(w.with_suffix(".pdiparams")))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
            metadata = w.parents[1] / "metadata.yaml"

        # MNN
        elif mnn:
            LOGGER.info(f"Loading {w} for MNN inference...")
            check_requirements("MNN")  # requires MNN - 需要 MNN
            import os

            import MNN

            config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
            rt = MNN.nn.create_runtime_manager((config,))
            net = MNN.nn.load_module_from_file(w, [], [], runtime_manager=rt, rearrange=True)

            def torch_to_mnn(x):
                return MNN.expr.const(x.data_ptr(), x.shape)

            metadata = json.loads(net.get_info()["bizCode"])

        # NCNN
        elif ncnn:
            LOGGER.info(f"Loading {w} for NCNN inference...")
            check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn")  # requires NCNN - 需要 NCNN
            import ncnn as pyncnn

            net = pyncnn.Net()
            net.opt.use_vulkan_compute = cuda
            w = Path(w)
            if not w.is_file():  # if not *.param - 如果不是 *.param
                w = next(w.glob("*.param"))  # get *.param file from *_ncnn_model dir - 从 *_ncnn_model 目录中获取 *.param 文件
            net.load_param(str(w))
            net.load_model(str(w.with_suffix(".bin")))
            metadata = w.parent / "metadata.yaml"

        # NVIDIA Triton Inference Server
        # 英伟达 Triton 推理服务器
        elif triton:
            check_requirements("tritonclient[all]")
            from ultralytics.utils.triton import TritonRemoteModel

            model = TritonRemoteModel(w)
            metadata = model.metadata

        # RKNN
        elif rknn:
            if not is_rockchip():
                raise OSError("RKNN inference is only supported on Rockchip devices.")
            LOGGER.info(f"Loading {w} for RKNN inference...")
            check_requirements("rknn-toolkit-lite2")
            from rknnlite.api import RKNNLite

            w = Path(w)
            if not w.is_file():  # if not *.rknn - 如果不是 *.rknn
                w = next(w.rglob("*.rknn"))  # get *.rknn file from *_rknn_model dir - 从 *_rknn_model 目录中获取 *.rknn 文件
            rknn_model = RKNNLite()
            rknn_model.load_rknn(w)
            rknn_model.init_runtime()
            metadata = Path(w).parent / "metadata.yaml"

        # Any other format (unsupported)
        # 任何其他格式（不受支持）
        else:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' is not a supported model format. Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape", "args"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
            end2end = metadata.get("args", {}).get("nms", False)
            dynamic = metadata.get("args", {}).get("dynamic", dynamic)
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")

        # Check names
        # 检查类名
        if "names" not in locals():  # names missing - 缺少类名
            names = default_class_names(data)
        names = check_class_names(names)

        # Disable gradients
        # 禁用梯度
        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())  # assign all variables to self - 将所有变量分配给 self

    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        在 YOLOv8 MultiBackend 模型上运行推理。

        Args:
            im (torch.Tensor): The image tensor to perform inference on. 图像张量进行推理。
            augment (bool): Whether to perform data augmentation during inference. Defaults to False. 推理期间是否执行数据增强。默认为 False。
            visualize (bool): Whether to visualize the output predictions. Defaults to False. 是否可视化输出预测。默认为 False。
            embed (List, optional): A list of feature vectors/embeddings to return. Defaults to None. 要返回的特征向量/嵌入的列表。默认为 None。

        Returns:
            (torch.Tensor | List[torch.Tensor]): The raw output tensor(s) from the model. 模型的原始输出张量。
        """
        b, ch, h, w = im.shape  # batch, channel, height, width - 批处理、通道、高度、宽度
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3) - torch BCHW 转换为 numpy BHWC 形状(1,320,192,3)

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)

        # TorchScript
        elif self.jit:
            y = self.model(im)

        # ONNX OpenCV DNN
        elif self.dnn:
            im = im.cpu().numpy()  # torch to numpy - torch 转换为 numpy
            self.net.setInput(im)
            y = self.net.forward()

        # ONNX Runtime
        elif self.onnx or self.imx:
            if self.dynamic:
                im = im.cpu().numpy()  # torch to numpy- torch 转换为 numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            else:
                if not self.cuda:
                    im = im.cpu()
                self.io.bind_input(
                    name="images",
                    device_type=im.device.type,
                    device_id=im.device.index if im.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(im.shape),
                    buffer_ptr=im.data_ptr(),
                )
                self.session.run_with_iobinding(self.io)
                y = self.bindings
            if self.imx:
                # boxes, conf, cls
                y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)

        # OpenVINO
        elif self.xml:
            im = im.cpu().numpy()  # FP32

            if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:  # optimized for larger batch-sizes - 优化用于更大的批处理大小
                n = im.shape[0]  # number of images in batch - 批处理中的图像数量
                results = [None] * n  # preallocate list with None to match the number of images - 预先分配列表，使用 None 匹配图像数量

                def callback(request, userdata):
                    """
                    Places result in preallocated list using userdata index.

                    使用 userdata 索引将结果放入预分配的列表。
                    """
                    results[userdata] = request.results

                # Create AsyncInferQueue, set the callback and start asynchronous inference for each input image
                # 创建 AsyncInferQueue，设置回调并为每个输入图像启动异步推理
                async_queue = self.ov.AsyncInferQueue(self.ov_compiled_model)
                async_queue.set_callback(callback)
                for i in range(n):
                    # Start async inference with userdata=i to specify the position in results list
                    # 使用 userdata=i 开始异步推理，以指定结果列表中的位置
                    async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)  # keep image as BCHW - 保持图像为 BCHW
                async_queue.wait_all()  # wait for all inference requests to complete - 等待所有推理请求完成
                y = np.concatenate([list(r.values())[0] for r in results])

            else:  # inference_mode = "LATENCY", optimized for fastest first result at batch-size 1 - 优化为批处理大小 1 的最快第一个结果
                y = list(self.ov_compiled_model(im).values())

        # TensorRT
        elif self.engine:
            if self.dynamic and im.shape != self.bindings["images"].shape:
                if self.is_trt10:
                    self.context.set_input_shape("images", im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                else:
                    i = self.model.get_binding_index("images")
                    self.context.set_binding_shape(i, im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]

        # CoreML
        elif self.coreml:
            im = im[0].cpu().numpy()
            im_pil = Image.fromarray((im * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im_pil})  # coordinates are xywh normalized - 坐标是 xywh 标准化的
            if "confidence" in y:
                raise TypeError(
                    "Ultralytics only supports inference of non-pipelined CoreML models exported with "
                    f"'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."
                )
                # TODO: CoreML NMS inference handling
                # from ultralytics.utils.ops import xywh2xyxy
                # box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                # conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float32)
                # y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            y = list(y.values())
            if len(y) == 2 and len(y[1].shape) != 4:  # segmentation model - 分割模型
                y = list(reversed(y))  # reversed for segmentation models (pred, proto) - 分割模型（pred、proto）的反转

        # PaddlePaddle
        elif self.paddle:
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

        # MNN
        elif self.mnn:
            input_var = self.torch_to_mnn(im)
            output_var = self.net.onForward([input_var])
            y = [x.read() for x in output_var]

        # NCNN
        elif self.ncnn:
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
            with self.net.create_extractor() as ex:
                ex.input(self.net.input_names()[0], mat_in)
                # WARNING: 'output_names' sorted as a temporary fix for https://github.com/pnnx/pnnx/issues/130
                y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]

        # NVIDIA Triton Inference Server
        elif self.triton:
            im = im.cpu().numpy()  # torch to numpy
            y = self.model(im)

        # RKNN
        elif self.rknn:
            im = (im.cpu().numpy() * 255).astype("uint8")
            im = im if isinstance(im, (list, tuple)) else [im]
            y = self.rknn_model.inference(inputs=im)

        # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
        else:
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                details = self.input_details[0]
                is_int = details["dtype"] in {np.int8, np.int16}  # is TFLite quantized int8 or int16 model
                if is_int:
                    scale, zero_point = details["quantization"]
                    im = (im / scale + zero_point).astype(details["dtype"])  # de-scale
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if is_int:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    if x.ndim == 3:  # if task is not classification, excluding masks (ndim=4) as well
                        # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                        # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                        if x.shape[-1] == 6 or self.end2end:  # end-to-end model
                            x[:, :, [0, 2]] *= w
                            x[:, :, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, :, 6::3] *= w
                                x[:, :, 7::3] *= h
                        else:
                            x[:, [0, 2]] *= w
                            x[:, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, 5::3] *= w
                                x[:, 6::3] *= h
                    y.append(x)
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            # TF 分段修复：导出与 ONNX 导出相反，原型被转置
            if len(y) == 2:  # segment with (det, proto) output order reversed - 与 (det、proto) 输出顺序相反的分段
                if len(y[1].shape) != 4:
                    y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32) - 应该是 y = (1, 116, 8400), (1, 160, 160, 32)
                if y[1].shape[-1] == 6:  # end-to-end model - 端到端模型
                    y = [y[1]]
                else:
                    y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160) - 应该是 y = (1, 116, 8400), (1, 32, 160, 160)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

        # for x in y:
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined - 段和名称未定义
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 32, 160, 160), (1, 116, 8400) - y = (1, 32, 160, 160), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        将 numpy 数组转换为张量。

        Args:
            x (np.ndarray): The array to be converted. 要转换的数组。

        Returns:
            (torch.Tensor): The converted tensor. object. 转换后的张量对象。
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        通过使用虚拟输入运行一次前向传递来预热模型。

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width).
                Defaults to (1, 3, 640, 640). 虚拟输入张量的形状，格式为(批处理大小、通道、高度、宽度)。默认为(1, 3, 640, 640)。
        """
        import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time) - （在此处导入，以便 torchvision 导入时间不记录在后处理时间中）

        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml, engine, coreml,
        saved_model, pb, tflite, edgetpu, tfjs, ncnn, mnn, imx or paddle.

        获取模型文件的路径并返回模型类型。可能的类型有 pt、jit、onnx、xml、engine、coreml、saved_model、pb、tflite、edgetpu、tfjs、ncnn、mnn、imx 或 paddle。

        Args:
            p (str): Path to the model file. Defaults to path/to/model.pt - 模型文件的路径。默认为 path/to/model.pt。

        Returns:
            (List[bool]): List of booleans indicating the model type. 指示模型类型的布尔值列表。

        Examples:
            >>> model = AutoBackend(weights="path/to/model.onnx")
            >>> model_type = model._model_type()  # returns "onnx" - 返回 "onnx"
        """
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]  # export suffixes - 导出后缀
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)  # checks - 检查
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")  # retain support for older Apple CoreML *.mlmodel formats - 保留对较旧的 Apple CoreML *.mlmodel 格式的支持
        types[8] &= not types[9]  # tflite &= not edgetpu - tflite &= 不是 edgetpu
        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}

        return types + [triton]
