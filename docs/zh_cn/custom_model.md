# 模型加载

X-AnyLabeling 当前内置了许多通用模型，具体可参考 [模型列表](../../docs/zh_cn/model_zoo.md)。

> [!TIP]
> 如果你需要通过远程服务器部署模型推理服务并支持多人协作，请参考 [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server)。

## 加载内置模型

在启用 AI 辅助标定功能之前，用户需要先加载模型，可通过左侧菜单栏的 `AI` 标识按钮或直接使用快捷键 `Ctrl+A` 激活。

通常，当用户从模型下拉列表中选择对应的模型时，后台会检查当前用户目录下 `~/xanylabeling_data/models/${model_name}` 是否存在相应的模型文件。如果存在，则直接加载；否则，直接通过网络自动下载到指定目录。

注意，当前软件内置的所有模型默认托管在 GitHub 的 release 仓库。因此，用户需要配置科学上网条件，并保持网络畅通，否则可能会下载失败。

对由于网络问题未能成功加载模型的用户，可选择离线下载并手动加载模型或修改模型下载源。

### 离线下载模型

- 打开 [model_zoo.md](./model_zoo.md) 文件，找到欲加载模型对应的配置文件。
- 编辑配置文件，修改模型路径，并根据需要选择性地修改其他超参数。
- 打开工具界面，点击**加载自定义模型**，选择配置文件所在路径即可。

### 修改模型下载源

详情可参考 [user_guide.md](./user_guide.md) 中的 `7.7 模型下载源配置` 章节。

## 加载已适配的用户自定义模型

> **已适配模型**是指当前已经在 X-AnyLabeling 中适配过的模型，无须用户编写模型推理代码。适配模型列表可参考 [模型列表](../../docs/zh_cn/model_zoo.md)。

本教程中，我们以 [YOLOv5s](https://github.com/ultralytics/yolov5) 模型为例，详细介绍如何加载自定义模型。

**a. 模型转换**

假设您已经在本地训练好一个模型，我们首先可以将 `PyTorch` 训练模型转换为 X-AnyLabeling 默认的 `ONNX` 文件格式（可选项）。具体地，执行：

```bash
python export.py --weights yolov5s.pt --include onnx
```

注意：当前版本暂不支持**动态输入**，因此请勿设置 `--dynamic` 参数。

此外，强烈建议通过 [Netron](https://netron.app/) 在线工具导入上一步导出的 `*.onnx` 文件，检查输入和输出节点信息，确保维度等信息符合预期。

<p align="center">
  <img src="../../assets/resources/netron.png" alt="Netron">
</p>

**b. 模型配置**

准备好 `onnx` 文件后，您可以浏览 [模型列表](../../docs/zh_cn/model_zoo.md) 文件，找到并拷贝对应模型的配置文件。

同样，以 [yolov5s.yaml](../../anylabeling/configs/auto_labeling/yolov5s.yaml) 为例，我们可以看下其内容：

```YAML
type: yolov5
name: yolov5s-r20230520
provider: Ultralytics
display_name: YOLOv5s
model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov5s.onnx
iou_threshold: 0.45
conf_threshold: 0.25
max_det: 300
classes:
  - person
  - bicycle
  - car
  ...
```

| 字段 | 描述 | 是否可修改 |
|------|------|------------|
| `type` | 模型类型标识，不支持自定义 | ❌ |
| `name` | 模型配置文件的索引名称，保留默认值即可 | ❌ |
| `provider` | 模型提供商，可根据实际情况修改 | ✔️ |
| `display_name` | 在界面上模型下拉列表中显示的名称，可自行修改 | ✔️ |
| `model_path` | 模型加载路径，支持相对路径和绝对路径 | ✔️ |
| `iou_threshold` | 用于非极大值抑制的交并比阈值 | ✔️ |
| `conf_threshold` | 用于非极大值抑制的置信度阈值 | ✔️ |
| `max_det` | 最大检测框数量 | ✔️ |
| `classes` | 模型的标签列表，需与训练时的标签列表一致 | ✔️ |

需要注意的是，以上字段并非所有模型都适用，具体可参考对应模型的定义。

例如，我们可以看下 [YOLO](../../anylabeling/services/auto_labeling/__base__/yolo.py) 模型的实现，其额外提供了以下可选配置项：

| 字段 | 描述 |
|------|------|
| `filter_classes` | 指定推理时使用的类别| 
| `agnostic` | 是否使用单类 NMS|

一个典型的参考示例如下：

```YAML
type: yolov5
name: yolov5s-r20230520
provider: Ultralytics
display_name: YOLOv5s
model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov5s.onnx
iou_threshold: 0.60
conf_threshold: 0.25
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

特别地，当且仅当使用低版本的 YOLOv5（v5.0 及以下）时，需要在配置文件中指定 `anchors` 和 `stride` 字段，否则请务必不要指定这些字段，以免造成模型推理错误。示例如下：

```YAML
type: yolov5
...
stride: 32
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

> **提示**: 对于分割模型，可指定 `epsilon_factor` 参数来控制输出轮廓点的平滑程度，默认值为 `0.005`。

**c. 模型加载**

了解完上述内容后，修改配置文件中的 `model_path` 字段，并根据需要选择性地修改其他超参数即可。

目前软件支持 **相对路径** 和 **绝对路径** 两种模型加载方式，用户在填写模型路径时，注意避免路径中转义字符的影响。

最后，在界面上方菜单栏中的模型下拉框列表中，找到 `...加载自定义模型` 选项，然后导入上一步准备的配置文件即可完成自定义模型加载。


## 加载未适配的用户自定义模型

> **未适配模型**指还未在 X-AnyLabeling 中适配过的模型，需要用户自行参考以下实施步骤进行集成。

这里以多类别语义分割模型 `U-Net` 为例，可遵循以下实施步骤：

**a. 训练及导出模型**

导出 `ONNX` 模型，确保输出节点的维度为 `[1, C, H, W]`，其中 `C` 为总的类别数（包含背景类）。

> **友情提示**：导出 `ONNX` 模型并非必选项，用户也可以根据需要选择其它模型格式，如 `PyTorch`、 `OpenVINO` 或 `TensorRT` 等。以 `Segment-Anything-2` 的视频目标追踪为例，可参考 [安装指南](../../examples/interactive_video_object_segmentation/README.md) 章节、配置文件定义 [sam2_hiera_base_video.yaml](../../anylabeling/configs/auto_labeling/sam2_hiera_base_video.yaml) 及相应的实现 [segment_anything_2_video.py](../../anylabeling/services/auto_labeling/segment_anything_2_video.py)。

**b. 定义配置文件**

首先，在[配置文件目录](../../anylabeling/configs/auto_labeling)下，新增一个配置文件，如`unet.yaml`：

```YAML
type: unet
name: unet-r20250101
display_name: U-Net (ResNet34)
provider: xxx
conf_threshold: 0.5
model_path: /path/to/best.onnx
classes:
  - cat
  - dog
  - _background_
```

其中：

| 字段 | 描述   |
|-----|--------|
| `type` | 指定模型类型，确保与现有模型类型不重复，以维护模型标识的唯一性。|
| `name` | 定义模型索引，用于内部引用和管理，避免与现有模型的索引名称冲突。|
| `display_name` | 展示在用户界面的模型名称，便于识别和选择，同样需保证其独特性，不与其它模型重名。|

以上三个字段为不可缺省字段。最后，可根据实际需要添加其它字段，如模型提供商、模型路径、模型超参等。

**c. 添加配置文件**

其次，将上述配置文件添加到[模型管理文件](../../anylabeling/configs/models.yaml)中：

```yaml
...

- model_name: "unet-r20250101"
  config_file: ":/unet.yaml"
...

```

**d. 配置UI组件**

这一步可根据需要自行添加UI组件，只需将模型名称添加到对应的列表即可，具体可参考此[文件](../../anylabeling/services/auto_labeling/__init__.py) 中的定义。

**e. 定义推理服务**

在定义推理服务的过程中，继承 [Model](../../anylabeling/services/auto_labeling/model.py) 基类是关键步骤之一，它允许你实现特定于模型的前向推理逻辑。

具体地，你可以在[模型推理服务目录](../../anylabeling/services/auto_labeling/)下新建一个 `unet.py` 文件，参考示例如下：

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
        input_h, input_w = self.input_shape
        image = cv2.resize(input_image, (input_w, input_h))
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess(self, image, outputs):
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
                shape.label = label
                shape.selected = False
                shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
```

这里：

- 元数据 `Meta` 类中：
    - `required_config_names`：用于指定模型配置文件中必须包含的配置项，确保模型推理服务能够正确初始化。
    - `widgets`：指定模型推理服务中需要显示的控件，如按钮、下拉框等，具体可参考此 [文件](../../anylabeling/services/auto_labeling/__init__.py) 中的定义。
    - `output_modes`：指定模型推理服务中输出的形状类型，支持多边形、矩形和旋转框等。
    - `default_output_mode`：指定模型推理服务中默认的输出形状类型。
- `predict_shapes` 和 `unload` 均属于抽象方法，分别用于定义模型推理过程和模型资源释放逻辑，因此一定需要实现。


**f. 添加至模型管理**

完成上述步骤后，我们需要打开 [模型配置文件](../../anylabeling/services/auto_labeling/__init__.py) 中，并将对应的模型类型字段（如`unet`）添加至 `_CUSTOM_MODELS` 列表中，并根据需要在不同配置项中添加对应的模型名称。

> **提示**: 如果你不知道如何实现对应的控件，可打开搜索面板，输入相应关键字，查看所有可用控件的实现逻辑。

最后，移步至 [模型管理类文件](../../anylabeling/services/auto_labeling/model_manager.py) 中，在 `_load_model` 方法中按照如下方式初始化你的实例：


```python
...

class ModelManager(QObject):
    """Model manager"""

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

- 模型类型字段需要与上述步骤**b. 定义配置文件**中定义的配置文件中的 `type` 字段保持一致。
- 如果是基于 `SAM` 的模式，请将 `self.auto_segmentation_model_unselected.emit()` 替换为 `self.auto_segmentation_model_selected.emit()` 以触发相应的功能。


# 模型导出

> 本章节将向您展示一些将自定义模型转换为 ONNX 模型的具体示例，以便您快速集成到 X-AnyLabeling 中。

## Classification

### [InternImage](https://github.com/OpenGVLab/InternImage)

InternImage 引入了一个大规模卷积神经网络 (CNN) 模型，利用可变形卷积作为核心操作符，以实现大的有效感受野、自适应空间聚合和减少的归纳偏置，从而从大量数据中学习到更强、更鲁棒的模式。它在基准测试中超越了当前的 CNN 和视觉Transformer。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions |
| 发表单位       | 上海人工智能实验室，清华大学，南京大学等                              |
| 发表时间       | CVPR'23                                                          |

请参考此 [教程](../../tools/onnx_exporter/export_internimage_model_onnx.py)。

### [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_person_attribute.md)

本教程为用户提供了一种使用 PaddleClas PULC (实用超轻量图像分类) 快速构建轻量、高精度和实用的人员属性分类模型的方法。该模型可广泛用于行人分析场景、行人跟踪场景等。

请参考此 [教程](../../tools/onnx_exporter/export_pulc_attribute_model_onnx.py)。

### [VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md)

本教程为用户提供了一种使用 PaddleClas PULC (实用超轻量图像分类) 快速构建轻量、高精度和实用的车辆属性分类模型的方法。该模型可广泛用于车辆识别、道路监控等场景。

请参考此 [教程](../../tools/onnx_exporter/export_pulc_attribute_model_onnx.py)。

## Object Detection

### [RF-DETR](https://github.com/roboflow/rf-detr)

`RF-DETR` 是第一个在 Microsoft COCO 基准测试中超过 60 AP 的实时模型，同时在小尺寸模型中表现出色。它还在 RF100-VL 上实现了最先进的性能，这是一个衡量模型对现实世界问题适应能力的对象检测基准。RF-DETR 的性能与当前的实时目标检测模型相当。

> 机构: Roboflow

请参考此 [教程](../../tools/onnx_exporter/export_rfdetr_onnx.py)。

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
| 发表时间       | NeurIPS'23                                                          |

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
| 发表时间       | Arxiv'22                                                            |

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
| 发表时间       | Arxiv'22                                                            |

请参考此 [文章](https://zhuanlan.zhihu.com/p/628660998)。

### [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO)

Hyper-YOLO 是一种新型目标检测方法，通过集成超图计算来捕获视觉特征之间的复杂高阶关联。该模型引入了超图计算增强的语义收集和散射（HGC-SCS）框架，将视觉特征图转换到语义空间并构建超图以进行高阶信息传播。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation |
| 发表单位       | 清华大学，西安交通大学                                              |
| 发表时间       | TAPMI'25                                                            |

下载模型，安装依赖后，修改`Hyper-YOLO/ultralytics/export.py`文件，设置`batch=1`和`half=False`：

```bash
import sys
import os
sys.path.append(os.getcwd())
from pathlib import Path
from ultralytics import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.device_count.cache_clear()

if __name__ == '__main__':
    model = 'hyper-yolon-seg.pt'
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    filename = model.export(imgsz=640, batch=1, format='ONNX', int8=False, half=False, device="0", verbose=False)
```

然后运行以下命令导出即可：

```bash
python3 ultralytics/utils/export_onnx.py
```

### [D-FINE](https://github.com/Peterande/D-FINE)

`D-FINE`是一款强大的实时目标检测器，它将DETR中的边界框回归任务重新定义为细粒度分布优化(FDR)，并引入全局最优定位自蒸馏(GO-LSD)，在不增加额外推理和训练成本的情况下实现了卓越性能。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement                      |
| 发表单位       | 中国科学技术大学                                                    |
| 发表时间       | ICLR'25 Spotlight                                                  |

请参考此[教程](../../tools/onnx_exporter/export_dfine_onnx.py)。

### [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)

`DEIMv2` 是 DEIM 框架的进化版本，同时利用了 DINOv3 的丰富特征。该方法设计了从超轻量级版本到 S、M、L 和 X 等不同规模的模型，以适应各种应用场景。在这些变体中，DEIMv2 都达到了最先进的性能，其中 S 级模型在具有挑战性的 COCO 基准测试中显著超过了 50 AP。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Real-Time Object Detection Meets DINOv3                      |
| 发表单位       | 英特灵达 & 厦门大学                                 |
| 发表时间       | Arxiv'25                                                 |

请参考此[教程](../../tools/onnx_exporter/export_deimv2_onnx.py)。

## Segment Anything

### [SAM](https://github.com/vietanhdev/samexporter)

分割一切模型 (`SAM`) 从输入提示（如点或框）中生成高质量的物体掩码。它可用于生成图像中所有物体的掩码，并在 1100 万张图像和 11 亿个掩码的数据集上进行了训练。SAM 在各种分割任务中具有强大的零样本性能。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Segment Anything                                                  |
| 发表单位       | Meta AI 研究院，FAIR                                                |
| 发表时间       | ICCV'23                                                            |

请参考这些 [步骤](https://github.com/vietanhdev/samexporter#sam-exporter)。

### [Efficient-SAM](https://github.com/CVHub520/efficientvit)

`EfficientViT` 是一系列新的视觉模型，用于高效的高分辨率密集预测。它使用一种新的轻量级多尺度线性注意模块作为核心构建模块。该模块仅通过硬件高效操作实现全局感受野和多尺度学习。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction |
| 发表单位       | 麻省理工学院                                                        |
| 发表时间       | ICCV'23                                                            |

请参考这些 [步骤](https://github.com/CVHub520/efficientvit#benchmarking-with-onnxruntime)。

### [SAM-Med2D](https://github.com/CVHub520/SAM-Med2D)

`SAM-Med2D` 是为解决将最先进的图像分割技术应用于医学图像挑战而开发的专业模型。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | SAM-Med2D                                                          |
| 发表单位       | OpenGVLab                                                         |
| 发表时间       | Arxiv'23                                                            |

请参考这些 [步骤](https://github.com/CVHub520/SAM-Med2D#-deploy)。

### [HQ-SAM](https://github.com/SysCV/sam-hq)

`HQ-SAM` 是增强版的任意物体分割模型 (SAM)，旨在提高掩码预测质量，特别是针对复杂结构，同时保持 SAM 的效率和零样本能力。它通过改进的解码过程和在专用数据集上的额外训练来实现这一目标。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Segment Anything in High Quality                                  |
| 发表单位       | 苏黎世联邦理工学院和香港科技大学                                    |
| 发表时间       | NeurIPS'23                                                      |

请参考此 [教程](https://github.com/CVHub520/sam-hq)。

### [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)

`EdgeSAM` 是任意物体分割模型 (SAM) 的加速变体，优化用于在边缘设备上高效执行，同时性能几乎没有妥协。它在性能上比原版 SAM

 提升了 40 倍，在边缘设备上的速度比 MobileSAM 快 14 倍，同时在 COCO 和 LVIS 数据集上的 mIoU 分别提高了 2.3 和 3.2。EdgeSAM 也是第一个在 iPhone 14 上能够运行超过 30 FPS 的 SAM 变体。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Prompt-In-the-Loop Distillation for On-Device Deployment of SAM   |
| 发表单位       | 南洋理工大学 S-Lab，上海人工智能实验室                               |
| 发表时间       | Arxiv'23                                                        |

请参考此 [教程](https://github.com/chongzhou96/EdgeSAM/blob/master/scripts/export_onnx_model.py)。

## Grounding

### [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) 

`Grounding DINO` 是一款最先进的 (SOTA) 零样本目标检测模型，擅长检测训练中未定义的物体。其独特的能力使其能够适应新物体和场景，使其在现实世界应用中具有高度的多样性。它在指称表达理解 (REC) 方面表现出色，能够基于文本描述识别和定位图像中的特定物体或区域。Grounding DINO 简化了目标检测，通过消除手工设计的组件（如非极大值抑制 (NMS)），简化了模型架构，增强了效率和性能。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection |
| 发表单位       | IDEA-CVR，IDEA-Research                                             |
| 发表时间       | Arxiv'23                                                            |

请参考此 [教程](../../tools/onnx_exporter/export_grounding_dino_onnx.py)。

### [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

`YOLO-World` 通过引入视觉语言建模来增强 YOLO 系列，实现高效的开放场景目标检测，在各种任务中表现出色。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Real-Time Open-Vocabulary Object Detection                        |
| 发表单位       | 腾讯人工智能实验室，ARC 实验室，腾讯 PCG，华中科技大学                |
| 发表时间       | Arxiv'24                                                        |

```bash
$ git clone https://github.com/ultralytics/ultralytics.git
$ cd ultralytics
$ yolo export model=yolov8s-worldv2.pt format=onnx opset=13 simplify
```

### [GeCo](https://github.com/jerpelhan/GeCo.git)

`GeCo` 是一种统一架构的少样本计数器，通过新颖的密集查询和计数损失，实现了高精度的目标检测、分割和计数。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | A Novel Unified Architecture for Low-Shot Counting by Detection and Segmentation |
| 发表单位       | 卢布尔雅那大学                                                        |
| 发表时间       | NeurIPS'24                                                        |

请参考此 [教程](../../tools/onnx_exporter/export_geco_onnx.py)。

## Image Tagging

### [Recognize Anything](https://github.com/xinyu1205/Tag2Text) 

`RAM` 是一款以其卓越图像识别能力著称的强大图像打标签模型。RAM 在零样本泛化方面表现出色，具有成本效益高和可复现的优点，依赖于开源和无注释数据集。RAM 的灵活性使其适用于广泛的应用场景，成为各种图像识别任务中的宝贵工具。

| 属性           | 值                                                                 |
|----------------|--------------------------------------------------------------------|
| 论文标题       | Recognize Anything: A Strong Image Tagging Model                  |
| 发表单位       | OPPO 研究院，IDEA-Research，AI Robotics                              |
| 发表时间       | Arxiv'23                                                            |

请参考此 [教程](../../tools/onnx_exporter/export_recognize_anything_model_onnx.py)。
