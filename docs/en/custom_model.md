# Model Loading

X-AnyLabeling currently comes with a variety of built-in general models. For specific details, refer to the [Model List](../../docs/en/model_zoo.md).

## Loading Built-in Models

Before using AI-assisted labeling features, users need to load a model, which can be activated by clicking the `AI` button in the left menu bar or by using the shortcut `Ctrl+A`.

Typically, when a user selects a model from the model dropdown list, the system checks if the corresponding model file exists in the user's directory at `~/xanylabeling_data/models/${model_name}`. If the file exists, it is loaded directly; otherwise, it will be automatically downloaded to the specified directory from the internet.

Please note that all built-in models in `X-AnyLabeling` are hosted on GitHub's release repository. Therefore, users need to ensure they have a reliable internet connection and access to the required resources. If downloading fails due to network issues, users can follow these steps:

- Open the [model_zoo.md](./model_zoo.md) file and locate the configuration file for the desired model.
- Edit the configuration file to modify the model path and optionally adjust other hyperparameters as needed.
- Open the tool interface, click on **Load Custom Model**, and select the path to the configuration file.

## Loading Adapted Custom Models

> **Adapted models** are those that have already been integrated into X-AnyLabeling, so you don't need to write any code. Refer to the [Model List](../../docs/en/model_zoo.md) for more details.

Here is an example of loading a custom model using the [YOLOv5s](https://github.com/ultralytics/yolov5) model:

### a. Model Conversion

Assuming you have trained a custom model, first, you should convert it into the `ONNX` file format:

```bash
python export.py --weights yolov5s.pt --include onnx
```

> [!Note]
> The current version does not support **dynamic input**, so do not set the `--dynamic` parameter. 

Additionally, you can use [Netron](https://netron.app/) to view the `onnx` file and check the input and output node information, ensuring that the first dimension of the input node is 1.

<p align="center">
  <img src="../../assets/resources/netron.png" alt="Netron">
</p>

### b. Model Configuration

After preparing the `onnx` file, you can browse the [Model List](../../docs/en/model_zoo.md) to find and download the corresponding model configuration file. Here, we continue use [yolov5s.yaml](../../anylabeling/configs/auto_labeling/yolov5s.yaml) as an example, with the following content:

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

| Field | Description | Modifiable |
|-------|-------------|------------|
| `type` | Model type identifier, not customizable. | ❌ |
| `name` | Index name of the model configuration file, keep the default value. | ❌ |
| `display_name` | The name displayed in the model dropdown list, can be customized. | ✔️ |
| `model_path` | Path to load the model, supports relative and absolute paths. | ✔️ |

For different models, X-AnyLabeling provides specific fields. For example, in the [YOLO](../../anylabeling/services/auto_labeling/__base__/yolo.py) model, the following hyperparameter configurations are provided:

| Field | Description |
|-------|-------------|
| `classes` | List of labels used by the model, must match the labels used during training. |
| `filter_classes` | Specifies the classes to use during inference. |
| `agnostic` | Whether to use class-agnostic NMS. |
| `nms_threshold` | Threshold for non-maximum suppression, used to filter overlapping bounding boxes. |
| `confidence_threshold` | Confidence threshold, used to filter low-confidence bounding boxes. |

A typical example is as follows:

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

For older versions of YOLOv5 (v5.0 and below), please specify the `anchors` and `stride` fields in the configuration file; otherwise, you must remove these fields. For example:

```YAML
type: yolov5
...
stride: 32
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

Additionally:
- For the `nms_threshold` and `confidence_threshold` fields, versions v2.4.0 and above support setting these directly from the GUI, allowing users to adjust them as needed.
- For segmentation models, you can specify the `epsilon_factor` parameter to control the smoothing degree of the output contour points, with a default value of 0.005.

### c. Model Loading

It is recommended to set the `model_path` field to the file name of the current `onnx` model and place the model and configuration files in the same directory, using a relative path to avoid issues with escape characters.

Finally, in the model dropdown at the bottom of the menu, select `...Load Custom Model` and import the configuration file prepared in the previous step to load the custom model.

## Loading Unadapted Custom Models

> **Unadapted models** refer to models that have not yet been integrated into X-AnyLabeling. Users must follow the implementation steps below to integrate them.

For a multi-class semantic segmentation model, follow these steps:

### a. Train and Export the Model

Export the model to `ONNX`, ensuring the output node's dimensions are `[1, C, H, W]`, where `C` is the total number of classes (including the background class).

### b. Define the Configuration File

First, add a new configuration file under the [configuration file directory](../../anylabeling/configs/auto_labeling), such as `unet.yaml`:

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

In it:

| Field | Description |
|-------|-------------|
| `type` | Required. Specifies the model type, ensuring it does not conflict with existing model types to maintain unique identification. |
| `name` | Required. Defines the model index for internal referencing and management, avoiding conflicts with existing model index names. |
| `display_name` | Required. The name displayed in the user interface, allowing for easy identification and selection. It must be unique and not duplicate other models' names. |

These three fields are mandatory. You can also add other fields as needed, such as model path, hyperparameters, and classes.

### c. Add Configuration File

Next, add the above configuration file to the [Model Management File](../../anylabeling/configs/auto_labeling/models.yaml):

```
...

- model_name: "unet-r20240101"
  config_file: ":/unet.yaml"
...

```

### d. Define the Inference Service

In defining the inference service, extending the [Model](../../anylabeling/services/auto_labeling/model.py) base class is a critical step. It allows you to implement model-specific inference logic. Specifically, you can create a new `unet.py` file under the [model inference service path](../../anylabeling/services/auto_labeling/), with a reference example as follows:

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

### e. Add to Model Management

Finally, add the implemented model class to the corresponding model management file. Specifically, open [model_manager.py](../../anylabeling/services/auto_labeling/model_manager.py), add the model type field (e.g., `unet`) to the `CUSTOM_MODELS` list, and initialize your instance in the `_load_model` method. Refer to the example below:

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

⚠️ Note:

- If using the `SAM` mode, replace `self.auto_segmentation_model_unselected.emit()` with `self.auto_segmentation_model_selected.emit()` to trigger the corresponding functionality.
- The model type field must match the `type` field defined in the configuration file from step **b. Define Configuration File**.


# Model Export

> This section provides specific examples of converting custom models to ONNX format, enabling quick integration into X-AnyLabeling.

## Classification

### [InternImage](https://github.com/OpenGVLab/InternImage)

InternImage introduces a large-scale convolutional neural network (CNN) model, leveraging deformable convolution as the core operator to achieve a large effective receptive field, adaptive spatial aggregation, and reduced inductive bias, leading to stronger and more robust pattern learning from massive data. It outperforms current CNNs and vision transformers on benchmarks

| Attribute       | Value                                                                 |
|-----------------|------------------------------------------------------------------------|
| Paper Title     | InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions |
| Affiliation     | Shanghai AI Laboratory, Tsinghua University, Nanjing University, etc. |
| Published       | CVPR 2023                                                             |

Refer to this [tutorial](../../tools/onnx_exporter/export_internimage_model_onnx.py).

### [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_person_attribute.md)

This tutorial provides a way for users to quickly build a lightweight, high-precision, and practical classification model of person attributes using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in pedestrian analysis scenarios, pedestrian tracking scenarios, etc.

Refer to this [tutorial](../../tools/onnx_exporter/export_pulc_attribute_model_onnx.py).

### [VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md)

This tutorial provides a way for users to quickly build a lightweight, high-precision, and practical classification model of vehicle attributes using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in vehicle identification, road monitoring, and other scenarios.

Refer to this [tutorial](../../tools/onnx_exporter/export_pulc_attribute_model_onnx.py).


## Object Detection

### [YOLOv5_OBB](https://github.com/hukaixuan19970627/yolov5_obb)

> Author: Kaixuan Hu

Refer to this [tutorial](https://github.com/CVHub520/yolov5_obb/tree/main).

### [YOLOv7](https://github.com/WongKinYiu/yolov7)

| Attribute       | Value                                                                 |
|-----------------|------------------------------------------------------------------------|
| Paper Title     | YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors |
| Affiliation     | Institute of Information Science, Academia Sinica, Taiwan                   |

```bash
python export.py --weights yolov7.pt --img-size 640 --grid
```

> **Note:** It is crucial to include the `--grid` parameter when running this command.

### [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)

| Attribute       | Value                                  |
|-----------------|----------------------------------------|
| Paper Title     | Efficient object detectors including Gold-YOLO |
| Affiliation     | huawei-noah                            |
| Published       | NeurIPS23                              |

```bash
$ git clone https://github.com/huawei-noah/Efficient-Computing.git
$ cd Detection/Gold-YOLO
$ python deploy/ONNX/export_onnx.py --weights Gold_n_dist.pt --simplify --ort
                                              Gold_s_pre_dist.pt                     
                                              Gold_m_pre_dist.pt
                                              Gold_l_pre_dist.pt
```

### [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)

`DAMO-YOLO` is a fast and accurate object detection method developed by the TinyML Team from Alibaba DAMO Data Analytics and Intelligence Lab. It achieves higher performance than state-of-the-art YOLO series, extending YOLO with new technologies, including Neural Architecture Search (NAS) backbones, efficient Reparameterized Generalized-FPN (RepGFPN), a lightweight head with AlignedOTA label assignment, and distillation enhancement. For more details, refer to the Arxiv Report. Here you can find not only powerful models but also highly efficient training strategies and complete tools from training to deployment.

| Attribute       | Value                                 |
|-----------------|---------------------------------------|
| Paper Title     | DAMO-YOLO: A Report on Real-Time Object Detection |
| Affiliation     | Alibaba Group                         |
| Published       | Arxiv22                               |

```bash
$ git clone https://github.com/tinyvision/DAMO-YOLO.git
$ cd DAMO-YOLO
$ python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640
```

### [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

Real-Time DEtection TRansformer (`RT-DETR`, aka RTDETR) is the first real-time end-to-end object detector known to the authors. RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS.

| Attribute       | Value                                       |
|-----------------|---------------------------------------------|
| Paper Title     | RT-DETR: DETRs Beat YOLOs on Real-time Object Detection |
| Affiliation     | Baidu                                       |
| Published       | Arxiv22                                     |

Refer to this [article](https://zhuanlan.zhihu.com/p/628660998).


## Segment Anything

### [SAM](https://github.com/vietanhdev/samexporter)

The Segment Anything Model (`SAM`) produces high-quality object masks from input prompts such as points or boxes. It can be used to generate masks for all objects in an image, trained on a dataset of 11 million images and 1.1 billion masks. SAM has strong zero-shot performance on various segmentation tasks.

| Attribute       | Value                                       |
|-----------------|---------------------------------------------|
| Paper Title     | Segment Anything                             |
| Affiliation     | Meta AI Research, FAIR                      |
| Published       | ICCV23                                      |

Refer to these [steps](https://github.com/vietanhdev/samexporter#sam-exporter).

### [Efficient-SAM](https://github.com/CVHub520/efficientvit)

`EfficientViT` is a new family of vision models for efficient high-resolution dense prediction. It uses a new lightweight multi-scale linear attention module as the core building block. This module achieves a global receptive field and multi-scale learning with only hardware-efficient operations.

| Attribute       | Value                                                         |
|-----------------|--------------------------------------------------------------|
| Paper Title     | EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction |
| Affiliation     | MIT                                                           |
| Published       | ICCV23                                                        |

Refer to these [steps](https://github.com/CVHub520/efficientvit#benchmarking-with-onnxruntime).

### [SAM-Med2D](https://github.com/CVHub520/SAM-Med2D)

`SAM-Med2D` is a specialized model developed to address the challenge of applying state-of-the-art image segmentation techniques to medical images.

| Attribute       | Value                |
|-----------------|----------------------|
| Paper Title     | SAM-Med2D            |
| Affiliation     | OpenGVLab            |
| Published       | Arxiv23              |

Refer to these [steps](https://github.com/CVHub520/SAM-Med2D#-deploy).

### [HQ-SAM](https://github.com/SysCV/sam-hq)

`HQ-SAM` is an enhanced version of the Segment Anything Model (SAM) designed to improve mask prediction quality, particularly for complex structures, while preserving SAM's efficiency and zero-shot capabilities. It achieves this through a refined decoding process and additional training on a specialized dataset.

| Attribute       | Value                                       |
|-----------------|---------------------------------------------|
| Paper Title     | Segment Anything in High Quality            |
| Affiliation     | ETH Zurich & HKUST                          |
| Published       | NeurIPS 2023                                |

Refer to this [tutorial](https://github.com/CVHub520/sam-hq).

### [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)

`EdgeSAM` is an accelerated variant of the Segment Anything Model (SAM), optimized for efficient execution on edge devices with minimal compromise in performance. It achieves a 40-fold speed increase compared to the original SAM, and outperforms MobileSAM, being 14 times as fast when deployed on edge devices while enhancing the mIoUs on COCO and LVIS by 2.3 and 3.2 respectively. EdgeSAM is also the first SAM variant that can run at over 30 FPS on an iPhone 14.

| Attribute       | Value                                                                                      |
|-----------------|-------------------------------------------------------------------------------------------|
| Paper Title     | Prompt-In-the-Loop Distillation for On-Device Deployment of SAM                           |
| Affiliation     | S-Lab, Nanyang Technological University, Shanghai Artificial Intelligence Laboratory.      |
| Published       | Arxiv 2023                                                                                 |


Refer to this [tutorial](https://github.com/chongzhou96/EdgeSAM/blob/master/scripts/export_onnx_model.py).

## Grounding

### [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) 

`Grounding DINO` is a state-of-the-art (SOTA) zero-shot object detection model excelling in detecting objects beyond the predefined training classes. Its unique capability allows adaptation to new objects and scenarios, making it highly versatile for real-world applications. It also performs well in Referring Expression Comprehension (REC), identifying and localizing specific objects or regions within an image based on textual descriptions. Grounding DINO simplifies object detection by eliminating hand-designed components like Non-Maximum Suppression (NMS), streamlining the model architecture, and enhancing efficiency and performance.

| Attribute       | Value                                                         |
|-----------------|--------------------------------------------------------------|
| Paper Title     | Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection |
| Affiliation     | IDEA-CVR, IDEA-Research                                       |
| Published       | Arxiv23                                                        |

Refer to this [tutorial](../../tools/onnx_exporter/export_grounding_dino_onnx.py).

### [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

`YOLO-World` enhances the YOLO series by incorporating vision-language modeling, achieving efficient open-scenario object detection with impressive performance on various tasks.

| Attribute       | Value                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------|
| Paper Title     | Real-Time Open-Vocabulary Object Detection                                                    |
| Affiliation     | Tencent AI Lab, ARC Lab, Tencent PCG, Huazhong University of Science and Technology.          |
| Published       | Arxiv 2024                                                                                     |

```bash
$ git clone https://github.com/ultralytics/ultralytics.git
$ cd ultralytics
$ yolo export model=yolov8s-worldv2.pt format=onnx opset=13 simplify
```

## Image Tagging

### [Recognize Anything](https://github.com/xinyu1205/Tag2Text) 

`RAM` is a robust image tagging model known for its exceptional capabilities in image recognition. RAM stands out for its strong and versatile performance, excelling in zero-shot generalization. It offers the advantages of being both cost-effective and reproducible, relying on open-source and annotation-free datasets. RAM's flexibility makes it suitable for a wide range of application scenarios, making it a valuable tool for various image recognition tasks.

| Attribute       | Value                                                             |
|-----------------|-------------------------------------------------------------------|
| Paper Title     | Recognize Anything: A Strong Image Tagging Model                  |
| Affiliation     | OPPO Research Institute, IDEA-Research, AI Robotics              |
| Published       | Arxiv23                                                           |

Refer to this [tutorial](../../tools/onnx_exporter/export_recognize_anything_model_onnx.py).
