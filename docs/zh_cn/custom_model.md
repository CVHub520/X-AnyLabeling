# 模型加载

X-AnyLabeling 当前内置了许多通用模型，具体可参考 [模型列表](../../docs/zh_cn/model_zoo.md)。

## 加载内置模型

在启用 AI 辅助标定功能之前，用户需要先加载模型，并通过左侧菜单栏的 `AI` 标识按钮或直接使用快捷键 `Ctrl+A` 激活。

通常，当用户从模型下拉列表中选择对应的模型时，后台会检查当前用户目录下 `~/xanylabeling_data/models/${model_name}` 是否存在相应的模型文件。如果存在，则直接加载；如果不存在，则会通过网络自动下载到指定目录。

请注意，`X-AnyLabeling` 当前内置的所有模型默认托管在 GitHub 的 release 仓库。因此，用户需要配置科学上网条件，并保持网络畅通，否则可能会下载失败。对于由于网络问题未能成功加载模型的用户，可参考以下步骤进行配置：

- 打开 [model_zoo.md](./model_zoo.md) 文件，找到欲加载模型对应的配置文件。
- 编辑配置文件，修改模型路径，并根据需要选择性地修改其他超参数。
- 打开工具界面，点击**加载自定义模型**，选择配置文件所在路径即可。

## 加载已适配的用户自定义模型

> **已适配模型**是指当前已经在 X-AnyLabeling 中适配过的模型，无须用户编写模型推理代码。具体可参考 [模型列表](../../docs/zh_cn/model_zoo.md)。

以下以 [YOLOv5s](https://github.com/ultralytics/yolov5) 模型为例，介绍加载自定义模型的步骤：

**a. 模型转换**

假设您已经训练好一个本地模型，首先将 `PyTorch` 训练模型转换为 `ONNX` 文件格式：

```bash
python export.py --weights yolov5s.pt --include onnx
```

注意：当前版本不支持动态输入，因此请勿设置 `--dynamic` 参数。此外，您可以通过 [Netron](https://netron.app/) 在线查看 `onnx` 文件，检查输入和输出节点信息，确保输入节点的第一个维度为1。

<p align="center">
  <img src="../../assets/resources/netron.png" alt="Netron">
</p>

**b. 模型配置**

准备好 `onnx` 文件后，您可以浏览 [模型列表](../../docs/zh_cn/model_zoo.md) 文件，找到并下载对应模型的配置文件。这里以 [yolov5s.yaml](../../anylabeling/configs/auto_labeling/yolov5s.yaml) 为例，其内容如下：

```YAML
type: yolov5
name: yolov5s-r20230520
display_name: YOLOv5s Ultralytics
model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov5s.onnx
nms_threshold: 0.45
confidence_threshold: 0.25
classes:
  - person
  - bicycle
  - car
  ...
```

| 字段 | 描述 | 是否可修改 |
|------|------|------------|
| `type` | 模型类型标识，不支持自定义。| ❌ |
| `name` | 模型配置文件的索引名称，保留默认值即可。 | ❌ |
| `display_name` | 在界面上模型下拉列表中显示的名称，可自行修改。 | ✔️ |
| `model_path` | 模型加载路径，支持相对路径和绝对路径。 | ✔️ |

对于不同模型，X-AnyLabeling 提供了一些特有字段，具体可参考对应模型的定义。以下以 [YOLO](../../anylabeling/services/auto_labeling/__base__/yolo.py) 模型为例，提供了一些超参数配置：

| 字段 | 描述 |
|------|------|
| `classes` | 模型的标签列表，需与训练时的标签列表一致。| 
| `filter_classes` | 指定推理时使用的类别。| 
| `agnostic` | 是否使用单类 NMS。|
| `nms_threshold` | 非极大值抑制的阈值，用于过滤重叠的目标框。|
| `confidence_threshold` | 置信度阈值，用于过滤置信度较低的目标框。|

一个典型的参考示例如下：

```YAML
type: yolov5
name: yolov5s-r20230520
display_name: YOLOv5s Custom
model_path: yolov5s_custom.onnx
nms_threshold: 0.60
confidence_threshold: 0.45
agnostic: True
filter_classes:
  - person
  - car
classes:
  - person
  - bicycle
  - car
  - ...
```

特别地，当使用低版本的 YOLOv5（v5.0 及以下）时，请在配置文件中指定 `anchors` 和 `stride` 字段，否则请删除这些字段。示例如下：

```YAML
type: yolov5
...
stride: 32
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

此外：
- 对于 `nms_threshold` 和 `confidence_threshold` 字段，`v2.4.0` 及以上版本支持直接从 GUI 界面进行设置，用户可根据需要修改。
- 对于分割模型，则可指定 `epsilon_factor` 参数来控制输出轮廓点的平滑程度，默认值为 0.005。

**c. 模型加载**

建议将 `model_path` 字段设置为当前 `onnx` 模型的文件名称，并将模型文件和配置文件放在同一目录下，使用相对路径进行加载，以避免路径中出现转义字符的影响。

最后，在菜单栏下方的模型下拉框选项中，找到 `...加载自定义模型`，然后导入上一步准备的配置文件即可完成自定义模型加载。


## 加载未适配的用户自定义模型

> **未适配模型**指还未在 X-AnyLabeling 中适配过的模型即内置模型，需要用户参考以下实施步骤进行集成。

这里以多类别语义分割模型，可遵循以下实施步骤：

**a. 训练及导出模型**

导出 `ONNX` 模型，确保输出节点的维度为 `[1, C, H, W]`，其中 `C` 为总的类别数（包含背景类）。

**b. 定义配置文件**

首先，在[配置文件目录](../../anylabeling/configs/auto_labeling)下，新增一个配置文件，如`unet.yaml`：

```YAML
type: unet
name: unet-r20240101
display_name: U-Net (ResNet34)
model_path: /path/to/best.onnx
classes:
  - cat
  - dog
  - _background_
```

其中：

| 字段 | 描述   |
|-----|--------|
| `type` | 必填项，指定模型的类型，确保与现有模型类型不重复，以维护模型标识的唯一性。|
| `name` | 必填项，定义模型的索引，用于内部引用和管理，避免与现有模型的索引名称冲突。|
| `display_name` | 必填项，展示在用户界面的模型名称，便于识别和选择，同样需保证其独特性，不与其它模型重名。|

以上三个字段为不可缺省字段。最后，可根据实际需要添加其它字段，如模型路径、模型超参、模型类别等。

**c. 添加配置文件**

其次，将上述配置文件添加到[模型管理文件](../../anylabeling/configs/auto_labeling/models.yaml)中：

```
...

- model_name: "unet-r20240101"
  config_file: ":/unet.yaml"
...

```

**d. 定义推理服务**

在定义推理服务的过程中，继承 [Model](../../anylabeling/services/auto_labeling/model.py) 基类是关键步骤之一，它允许你实现特定于模型的前向推理逻辑。具体地，你可以在[模型推理服务路径](../../anylabeling/services/auto_labeling/)下新建一个`unet.py`文件，参考示例如下：

```python
import logging
import os

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel


class UNet(Model):
    """Semantic segmentation model using UNet"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.

        Args:
            input_image (numpy.ndarray): The input image to be processed.

        Returns:
            numpy.ndarray: The pre-processed output.
        """
        input_h, input_w = self.input_shape
        image = cv2.resize(input_image, (input_w, input_h))
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess(self, image, outputs):
        """
        Post-processes the network's output.

        Args:
            image (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output from the network.

        Returns:
            contours (list): List of contours for each detected object class.
                            Each contour is represented as a dictionary containing
                            the class label and a list of contour points.
        """
        n, c, h, w = outputs.shape
        image_height, image_width = image.shape[:2]
        # Obtain the category index of each pixel
        # target shape: (1, h, w)
        outputs = np.argmax(outputs, axis=1)
        results = []
        for i in range(c):
            # Skip the background label
            if self.classes[i] == '_background_':
                continue
            # Get the category index of each pixel for the first batch by adding [0].
            mask = outputs[0] == i
            # Rescaled to original shape
            mask_resized = cv2.resize(mask.astype(np.uint8), (image_width, image_height))
            # Get the contours
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Append the contours along with their respective class labels
            results.append((self.classes[i], [np.squeeze(contour).tolist() for contour in contours]))
        return results

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        blob = self.preprocess(image)
        outputs = self.net.get_ort_inference(blob)
        results = self.postprocess(image, outputs)
        shapes = []
        for item in results:
            label, contours = item
            for points in contours:
                # Make sure to close
                points += points[0]
                shape = Shape(flags={})
                for point in points:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = label
                shape.selected = False
                shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
```

**e. 添加至模型管理**

最后，我们仅需将实现好的模型类添加至对应的模型管理文件中即可。具体地，你可以打开 [model_manager.py](../../anylabeling/services/auto_labeling/model_manager.py)，将对应的模型类型字段（如`unet`）添加至 `CUSTOM_MODELS` 列表中，同时在 `_load_model` 方法中初始化你的实例。参考示例如下：

```python
...

class ModelManager(QObject):
    """Model manager"""

    MAX_NUM_CUSTOM_MODELS = 5
    CUSTOM_MODELS = [
      ...
      "unet",
      ...
    ]

    def __init__(self):
        ...

    ...

    def _load_model(self, model_id):
        """Load and return model info"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None
            self.auto_segmentation_model_unselected.emit()

        model_config = copy.deepcopy(self.model_configs[model_id])
        if model_config["type"] == "yolov5":
            ...
        elif model_config["type"] == "unet":
            from .unet import UNet

            try:
                model_config["model"] = UNet(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                return
          ...
    ...
```

⚠️注意：

- 如果是基于 `SAM` 的模式，请将 `self.auto_segmentation_model_unselected.emit()` 替换为 `self.auto_segmentation_model_selected.emit()` 以触发相应的功能。
- 模型类型字段需要与上述步骤**b. 定义配置文件**中定义的配置文件中的 `type` 字段保持一致。


# 模型导出

> 本章节将向您展示一些将自定义模型转换为 ONNX 模型的具体示例，以便您快速集成到 X-AnyLabeling 中。

## Classification

### [InternImage](https://github.com/OpenGVLab/InternImage)

InternImage 引入了一个大规模卷积神经网络 (CNN) 模型，利用可变形卷积作为核心操作符，以实现大的有效感受野、自适应空间聚合和减少的归纳偏置，从而从大量数据中学习到更强、更鲁棒的模式。它在基准测试中超越了当前的 CNN 和视觉Transformer。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions |
| 发表单位       | 上海人工智能实验室，清华大学，南京大学等                              |
| 发表时间       | CVPR 2023                                                          |

请参考此 [教程](../../tools/onnx_exporter/export_internimage_model_onnx.py)。

### [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_person_attribute.md)

本教程为用户提供了一种使用 PaddleClas PULC (实用超轻量图像分类) 快速构建轻量、高精度和实用的人员属性分类模型的方法。该模型可广泛用于行人分析场景、行人跟踪场景等。

请参考此 [教程](../../tools/onnx_exporter/export_pulc_attribute_model_onnx.py)。

### [VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md)

本教程为用户提供了一种使用 PaddleClas PULC (实用超轻量图像分类) 快速构建轻量、高精度和实用的车辆属性分类模型的方法。该模型可广泛用于车辆识别、道路监控等场景。

请参考此 [教程](../../tools/onnx_exporter/export_pulc_attribute_model_onnx.py)。

## Object Detection

### [YOLOv5_OBB](https://github.com/hukaixuan19970627/yolov5_obb)

> 作者: Kaixuan Hu

请参考此 [教程](https://github.com/CVHub520/yolov5_obb/tree/main)。

### [YOLOv7](https://github.com/WongKinYiu/yolov7)

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors |
| 发表单位       | 台湾中央研究院信息科学研究所                                         |

```bash
python export.py --weights yolov7.pt --img-size 640 --grid
```

> **注意：** 运行此命令时必须包含 `--grid` 参数。

### [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Efficient object detectors including Gold-YOLO                     |
| 发表单位       | 华为诺亚                                                           |
| 发表时间       | NeurIPS23                                                          |

```bash
$ git clone https://github.com/huawei-noah/Efficient-Computing.git
$ cd Detection/Gold-YOLO
$ python deploy/ONNX/export_onnx.py --weights Gold_n_dist.pt --simplify --ort
                                              Gold_s_pre_dist.pt                     
                                              Gold_m_pre_dist.pt
                                              Gold_l_pre_dist.pt
```

### [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)

`DAMO-YOLO` 是由阿里巴巴达摩院数据分析与智能实验室的 TinyML 团队开发的一种快速准确的目标检测方法。它通过引入新的技术，包括神经架构搜索 (NAS) 主干网、高效的重参数化通用-FPN (RepGFPN)、轻量级头部和 AlignedOTA 标签分配，并进行蒸馏增强，使其性能超过了最新的 YOLO 系列。更多细节请参阅 Arxiv 报告。这里不仅可以找到强大的模型，还可以找到从训练到部署的高效训练策略和完整工具。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | DAMO-YOLO: A Report on Real-Time Object Detection                  |
| 发表单位       | 阿里巴巴集团                                                       |
| 发表时间       | Arxiv22                                                            |

```bash
$ git clone https://github.com/tinyvision/DAMO-YOLO.git
$ cd DAMO-YOLO
$ python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640
```

### [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

实时检测变换器 (`RT-DETR`，又称 RTDETR) 是已知的第一个实时端到端目标检测器。RT-DETR-L 在 COCO val2017 上达到了 53.0% AP，并在 T4 GPU 上达到了 114 FPS，而 RT-DETR-X 达到了 54.8% AP 和 74 FPS，速度和准确性都超过了同规模的所有 YOLO 检测器。此外，RT-DETR-R50 达到了 53.1% AP 和 108 FPS，准确性比 DINO-Deformable-DETR-R50 高 2.2% AP，FPS 快约 21 倍。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | RT-DETR: DETRs Beat YOLOs on Real-time Object Detection            |
| 发表单位       | 百度                                                               |
| 发表时间       | Arxiv22                                                            |

请参考此 [文章](https://zhuanlan.zhihu.com/p/628660998)。

## Segment Anything

### [SAM](https://github.com/vietanhdev/samexporter)

分割一切模型 (`SAM`) 从输入提示（如点或框）中生成高质量的物体掩码。它可用于生成图像中所有物体的掩码，并在 1100 万张图像和 11 亿个掩码的数据集上进行了训练。SAM 在各种分割任务中具有强大的零样本性能。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Segment Anything                                                  |
| 发表单位       | Meta AI 研究院，FAIR                                                |
| 发表时间       | ICCV23                                                            |

请参考这些 [步骤](https://github.com/vietanhdev/samexporter#sam-exporter)。

### [Efficient-SAM](https://github.com/CVHub520/efficientvit)

`EfficientViT` 是一系列新的视觉模型，用于高效的高分辨率密集预测。它使用一种新的轻量级多尺度线性注意模块作为核心构建模块。该模块仅通过硬件高效操作实现全局感受野和多尺度学习。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction |
| 发表单位       | 麻省理工学院                                                        |
| 发表时间       | ICCV23                                                            |

请参考这些 [步骤](https://github.com/CVHub520/efficientvit#benchmarking-with-onnxruntime)。

### [SAM-Med2D](https://github.com/CVHub520/SAM-Med2D)

`SAM-Med2D` 是为解决将最先进的图像分割技术应用于医学图像挑战而开发的专业模型。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | SAM-Med2D                                                          |
| 发表单位       | OpenGVLab                                                         |
| 发表时间       | Arxiv23                                                            |

请参考这些 [步骤](https://github.com/CVHub520/SAM-Med2D#-deploy)。

### [HQ-SAM](https://github.com/SysCV/sam-hq)

`HQ-SAM` 是增强版的任意物体分割模型 (SAM)，旨在提高掩码预测质量，特别是针对复杂结构，同时保持 SAM 的效率和零样本能力。它通过改进的解码过程和在专用数据集上的额外训练来实现这一目标。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Segment Anything in High Quality                                  |
| 发表单位       | 苏黎世联邦理工学院和香港科技大学                                    |
| 发表时间       | NeurIPS 2023                                                      |

请参考此 [教程](https://github.com/CVHub520/sam-hq)。

### [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)

`EdgeSAM` 是任意物体分割模型 (SAM) 的加速变体，优化用于在边缘设备上高效执行，同时性能几乎没有妥协。它在性能上比原版 SAM

 提升了 40 倍，在边缘设备上的速度比 MobileSAM 快 14 倍，同时在 COCO 和 LVIS 数据集上的 mIoU 分别提高了 2.3 和 3.2。EdgeSAM 也是第一个在 iPhone 14 上能够运行超过 30 FPS 的 SAM 变体。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Prompt-In-the-Loop Distillation for On-Device Deployment of SAM   |
| 发表单位       | 南洋理工大学 S-Lab，上海人工智能实验室                               |
| 发表时间       | Arxiv 2023                                                        |

请参考此 [教程](https://github.com/chongzhou96/EdgeSAM/blob/master/scripts/export_onnx_model.py)。

## Grounding

### [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) 

`Grounding DINO` 是一款最先进的 (SOTA) 零样本目标检测模型，擅长检测训练中未定义的物体。其独特的能力使其能够适应新物体和场景，使其在现实世界应用中具有高度的多样性。它在指称表达理解 (REC) 方面表现出色，能够基于文本描述识别和定位图像中的特定物体或区域。Grounding DINO 简化了目标检测，通过消除手工设计的组件（如非极大值抑制 (NMS)），简化了模型架构，增强了效率和性能。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection |
| 发表单位       | IDEA-CVR，IDEA-Research                                             |
| 发表时间       | Arxiv23                                                            |

请参考此 [教程](../../tools/onnx_exporter/export_grounding_dino_onnx.py)。

### [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

`YOLO-World` 通过引入视觉语言建模来增强 YOLO 系列，实现高效的开放场景目标检测，在各种任务中表现出色。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Real-Time Open-Vocabulary Object Detection                        |
| 发表单位       | 腾讯人工智能实验室，ARC 实验室，腾讯 PCG，华中科技大学                |
| 发表时间       | Arxiv 2024                                                        |

```bash
$ git clone https://github.com/ultralytics/ultralytics.git
$ cd ultralytics
$ yolo export model=yolov8s-worldv2.pt format=onnx opset=13 simplify
```

## Image Tagging

### [Recognize Anything](https://github.com/xinyu1205/Tag2Text) 

`RAM` 是一款以其卓越图像识别能力著称的强大图像打标签模型。RAM 在零样本泛化方面表现出色，具有成本效益高和可复现的优点，依赖于开源和无注释数据集。RAM 的灵活性使其适用于广泛的应用场景，成为各种图像识别任务中的宝贵工具。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Recognize Anything: A Strong Image Tagging Model                  |
| 发表单位       | OPPO 研究院，IDEA-Research，AI Robotics                              |
| 发表时间       | Arxiv23                                                            |

请参考此 [教程](../../tools/onnx_exporter/export_recognize_anything_model_onnx.py)。


