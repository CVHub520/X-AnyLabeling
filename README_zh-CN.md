<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img width="100%" src="https://user-images.githubusercontent.com/72010077/273420485-bdf4a930-8eca-4544-ae4b-0e15f3ebf095.png"></a>
  </p>

[简体中文](README.zh-CN.md) | [English](README.md)

</div>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-LGPL%20v3-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/github/v/release/CVHub520/X-AnyLabeling?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/CVHub520/X-AnyLabeling/stargazers"><img src="https://img.shields.io/github/stars/CVHub520/X-AnyLabeling?color=ccf"></a>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

<div align=center>
  <figure>
    <img src="https://user-images.githubusercontent.com/72010077/277691916-58be8e7d-133c-4df8-9416-d3243fc7a335.gif" alt="Grounding DINO">
    <figcaption>SOTA Zero-Shot Openset Object Detection Model</figcaption>
  </figure>
</div>

</br>

<div align=center>
  <figure>
    <img src="https://user-images.githubusercontent.com/72010077/277692001-b58832b3-4c21-4c6f-9121-02d9daf2b02b.gif" alt="Recognize Anything Model">
    <figcaption>Strong Image Tagging Model</figcaption>
  </figure>
</div>

</br>

<div align=center>
  <figure>
    <img src="https://user-images.githubusercontent.com/72010077/277405591-5ebffdcf-83e8-4999-9594-ee4058627d47.gif" alt="Segment Anything Model">
    <figcaption>Powerful Object Segmentation Anything Model</figcaption>
  </figure>
</div>

<div align=center>
  <figure>
    <img src="https://user-images.githubusercontent.com/72010077/282393906-059920cc-0f65-4d2c-9350-941aaa8bbd02.png" alt="PULC PersonAttribute Model">
    <figcaption>Advanced Multi-Label Classification Model</figcaption>
  </figure>
</div>

## 📄 目录

- [🥳 新功能](#🥳-新功能-⏏️)
- [👋 简介](#👋-简介-⏏️)
- [🔥 亮点](#🔥-亮点-⏏️)
  - [🗝️关键功能](#🗝️关键功能-)
  - [⛏️模型库](#⛏️模型库-)
- [📖 教程](#📖-教程-⏏️)
  - [🔜快速开始](#🔜快速开始)
  - [👨🏼‍💻从源码构建](#👨🏼‍💻从源码构建)
  - [📦编译](#📦编译)
- [📋 用法](#📋-用法-⏏️)
  - [📌基础用法](#📌基础用法)
  - [🚀高级用法](#🚀高级用法)
  - [📜文档](#📜文档)
  - [🧷快捷键](#🧷快捷键)
- [📧 联系](#📧-联系-⏏️)
- [✅ 许可](#✅-许可-⏏️)
- [🏷️ 引用](#🏷️-引用-⏏️)

## 🥳 新功能 [⏏️](#📄-目录)

- Nov. 2023:
  - 🤗 Release the latest version [2.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.1.0) 🤗
  - Supoort [InternImage](https://arxiv.org/abs/2211.05778) model (**CVPR'23**).
  - Release version [2.0.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.0.0).
  - Added support for Grounding-SAM, combining [GroundingDINO](https://github.com/wenyi5608/GroundingDINO) with [HQ-SAM](https://github.com/SysCV/sam-hq) to achieve sota zero-shot high-quality predictions!
  - Enhanced support for [HQ-SAM](https://github.com/SysCV/sam-hq) model to achieve high-quality mask predictions.
  - Support the [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/en/PULC/PULC_person_attribute_en.md) and [VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/en/PULC/PULC_vehicle_attribute_en.md) model for multi-label classification task.
  - Introducing a new multi-label attribute annotation functionality.
  - Release version [1.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v1.1.0).
  - Support pose estimation: [YOLOv8-Pose](https://github.com/ultralytics/ultralytics).
  - Support object-level tag with yolov5_ram.
  - Add a new feature enabling batch labeling for arbitrary unknown categories based on Grounding-DINO.
- Oct. 2023:
  - Release version [1.0.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v1.0.0).
  - Add a new feature for rotation box.
  -  Support [YOLOv5-OBB](https://github.com/hukaixuan19970627/yolov5_obb) with [DroneVehicle](https://github.com/VisDrone/DroneVehicle) and [DOTA](https://captain-whu.github.io/DOTA/index.html)-v1.0/v1.5/v2.0 model.
  - SOTA Zero-Shot Object Detection - [GroundingDINO](https://github.com/wenyi5608/GroundingDINO) is released.
  - SOTA Image Tagging Model - [Recognize Anything](https://github.com/xinyu1205/Tag2Text) is released.
  - Support **YOLOv5-SAM** and **YOLOv8-EfficientViT_SAM** union task.
  - Support **YOLOv5** and **YOLOv8** segmentation task.
  - Release [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO) and [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) models.
  - Release MOT algorithms: [OC_Sort](https://github.com/noahcao/OC_SORT) (**CVPR'23**).
  - Add a new feature for small object detection using [SAHI](https://github.com/obss/sahi).
- Sep. 2023:
  - Release version [0.2.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.4).
  - Release [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit) (**ICCV'23**),[SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D), [MedSAM](https://arxiv.org/abs/2304.12306) and YOLOv5-SAM.
  - Support [ByteTrack](https://github.com/ifzhang/ByteTrack) (**ECCV'22**) for MOT task.
  - Support [PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR) model.
  - Add `video` annotation feature.
  - Add `yolo`/`coco`/`voc`/`mot`/`dota` export functionality.
  - Add the ability to process all images at once.
- Aug. 2023:
  - Release version [0.2.0]((https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.0)).
  - Release [LVMSAM](https://arxiv.org/abs/2306.11925) and it's variants [BUID](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/buid), [ISIC](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/isic), [Kvasir](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/kvasir).
  - Support lane detection algorithm: [CLRNet](https://github.com/Turoad/CLRNet) (**CVPR'22**).
  - Support 2D human whole-body pose estimation: [DWPose](https://github.com/IDEA-Research/DWPose/tree/main) (**ICCV'23 Workshop**).
- Jul. 2023:
  - Add [label_converter.py](./tools/label_converter.py) script.
  - Release [RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/rtdetr/README.md) model.
- Jun. 2023:
  - Release [YOLO-NAS](https://github.com/Deci-AI/super-gradients/tree/master) model.
  - Support instance segmentation: [YOLOv8-seg](https://github.com/ultralytics/ultralytics).
  - Add [README_zh-CN.md](README_zh-CN.md) of X-AnyLabeling.
- May. 2023:
  - Release version [0.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.1.0).
  - Release [YOLOv6-Face](https://github.com/meituan/YOLOv6/tree/yolov6-face) for face detection and facial landmark detection.
  - Release [SAM](https://arxiv.org/abs/2304.02643) and it's faster version [MobileSAM](https://arxiv.org/abs/2306.14289).
  - Release [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).


## 👋 简介 [⏏️](#📄-目录)

`X-AnyLabeling` 是一款出色的标注工具，汲取了[LabelImg](https://github.com/HumanSignal/labelImg)、[roLabelImg](https://github.com/cgvict/roLabelImg)、[Labelme](https://github.com/wkentaro/labelme)以及[Anylabeling](https://github.com/vietanhdev/anylabeling )等知名标注软件的灵感。它代表了自动数据标注的未来重要一步。这一创新工具不仅简化了标注过程，还无缝集成了先进的人工智能模型，以提供卓越的结果。X-AnyLabeling 专注于实际应用，致力于为开发人员提供工业级、功能丰富的解决方案，用于自动进行各种复杂任务的标注和数据处理。

## 🔥 亮点 [⏏️](#📄-目录)

### 🗝️关键功能

- 支持导入 `图像` 和 `视频`。
- 支持 `CPU` 和 `GPU` 推理，可按需选择。
- 兼容多种领先的深度学习算法。
- 单帧预测和一键处理所有图像。
- 导出选项，支持格式如 `COCO-JSON`、`VOC-XML`、`YOLOv5-TXT`、`DOTA-TXT` 和 `MOT-CSV`。
- 与流行框架集成，包括 [PaddlePaddle](https://www.paddlepaddle.org.cn/)、[OpenMMLab](https://openmmlab.com/)、[timm](https://github.com/huggingface/pytorch-image-models) 等。
- 提供全面的 `帮助文档`，并提供积极的 `开发者社区支持`。
- 支持各种视觉任务，如 `目标检测`、`图像分割`、`人脸识别` 等。
- 模块化设计，赋予用户根据其具体需求自行编译系统的能力，同时支持自定义和二次开发。
- 图像标注功能，包括 `多边形`、`矩形`、`旋转框`、`圆形`、`线条`、`点`，以及 `文本检测`、`识别` 和 `KIE` 标注。

### ⛏️模型库

<div align="center">

| **Object Detection** | **SOD with [SAHI](https://github.com/obss/sahi)** | **Facial Landmark Detection** | **2D Pose Estimation** |
| :---: | :---: | :---: | :---: |
| <img src='https://user-images.githubusercontent.com/72010077/273488633-fc31da5c-dfdd-434e-b5d0-874892807d95.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095892-934be83a-f869-4a31-8e52-1074184149d1.jpg' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206095684-72f42233-c9c7-4bd8-9195-e34859bd08bf.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100220-ab01d347-9ff9-4f17-9718-290ec14d4205.gif' height="126px" width="180px"> |
|  **2D Lane Detection** | **OCR** | **MOT** | **Instance Segmentation** |
| <img src='https://user-images.githubusercontent.com/72010077/273764641-65f456ed-27ce-4077-8fce-b30db093b988.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273421210-30d20e08-3b72-4f4d-8976-05b564e13d87.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206111753-836e7827-968e-4c80-92ef-7a78766892fc.gif' height="126px" width="180px"  > | <img src='https://user-images.githubusercontent.com/61035602/206095831-cc439557-1a23-4a99-b6b0-b6f2e97e8c57.jpg' height="126px" width="180px"> |
|  **Image Tagging** | **Grounding DINO** | **Recognition** | **Rotation** |
| <img src='https://user-images.githubusercontent.com/72010077/277670825-8797ac7e-e593-45ea-be6a-65c3af17b12b.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277395884-4d500af3-3e4e-4fb3-aace-9a56a09c0595.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277396071-79daec2c-6b0a-4d42-97cf-69fd098b3400.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277395955-aab54ea0-88f5-41af-ab0a-f4158a673f5e.png' height="126px" width="180px"> |
|  **[SAM](https://segment-anything.com/)** | **BC-SAM** | **Skin-SAM** | **Polyp-SAM** |
| <img src='https://user-images.githubusercontent.com/72010077/273421331-2c0858b5-0b92-405b-aae6-d061bc25aa3c.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764259-718dce97-d04d-4629-b6d2-95f17670ce2a.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764288-e26767d1-3c44-45cb-a72e-124efb4e8263.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764318-e8b6a197-e733-478e-a210-e4386bafa1e4.png' height="126px" width="180px"> |

更多详情，敬请参考[模型列表](./docs/models_list.md)。

</div>

## 📖 教程 [⏏️](#📄-目录)

### 🔜快速开始

直接从 [Release](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.1.0) 或 [百度网盘](https://pan.baidu.com/s/1XKflqsbb7C_7seL-ROD3bg?pwd=a64z) 下载并运行 `GUI` 版本。

注意事项：
- 对于 MacOS：
  - 安装完成后，前往 Applications 文件夹。
  - 右键单击应用程序并选择打开。
  - 从第二次开始，您可以使用 Launchpad 正常打开应用程序。

- 由于当前工具缺乏必要的硬件支持，所以仅提供 `Windows` 和 `Linux` 可执行版本。如果您需要其他操作系统的可执行程序，例如 `MacOS`，请参考以下步骤进行自行编译。
- 为了获得更稳定的性能和功能支持，强烈建议从源码进行构建。

### 👨🏼‍💻从源码构建

- 安装所需的库：

```bash
pip install -r requirements.txt
```

> 如果您需要使用 GPU 推理，请安装相应的 requirements-gpu.txt 文件，并根据您本地的 CUDA 和 CuDNN 版本下载相应版本的 onnxruntime-gpu。更多详细信息，请参阅[帮助文档](./docs/Q&A.md).

- 生成资源 [可选]:

```
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- 运行应用程序：

```
python anylabeling/app.py
```

### 📦编译

> 请注意，以下步骤是非必要的，这些构建脚本仅为可能需要自定义和编译软件以在特定环境中分发的用户提供的。

```bash
#Windows-CPU
bash scripts/build_executable.sh win-cpu

#Windows-GPU
bash scripts/build_executable.sh win-gpu

#Linux-CPU
bash scripts/build_executable.sh linux-cpu

#Linux-GPU
bash scripts/build_executable.sh linux-gpu
```

<details open>

<summary>注意：</summary>

1. 在编译之前，请根据适用的GPU/CPU版本，在 "anylabeling/app_info.py" 文件中修改 `__preferred_device__` 参数。
2. 如果您需要编译GPU版本，请使用 "pip install -r requirements-gpu*.txt" 安装相应的环境。具体来说，对于编译GPU版本，需要手动修改 "anylabeling-*-gpu.spec" 文件中的 "datas" 列表参数，以包括您本地 onnxruntime-gpu 的相关动态库（*.dll 或 *.so）。此外，在下载 onnxruntime-gpu 包时，请确保与您的CUDA版本兼容。您可以参考官方[文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)以获取特定兼容性表格。
3. 对于 macOS 版本，您可以参考 "anylabeling-win-*.spec" 脚本进行修改。
</details>

## 📋 用法 [⏏️](#📄-目录)

### 📌基础用法

1. 按照上述说明进行构建和启动。
2. 在 `菜单/文件` 中点击 `更改输出目录` 以指定输出目录；否则，它将默认保存在当前图像路径下。
3. 点击 `打开`/`打开目录`/`打开视频` 以选择特定的文件、文件夹或视频。
4. 在左侧工具栏上点击 `开始绘制 xxx` 按钮或 `自动标注` 控制以启动标注。
5. 单击并释放鼠标左键以选择要注释的矩形区域。或者，您可以按 "运行 (i)" 键进行一键处理。

> 注意：标注文件将保存到您指定的文件夹中，并且您可以参考下面的热键以加快您的工作流程。

### 🚀高级用法

- 选择左侧的 **AutoLalbeing 按钮** 或按下快捷键 "Ctrl + A" 以启动自动标注。
- 从下拉菜单 "Model" 中选择一个 `Segment Anything-liked Models`，其中 "Quant" 表示模型的量化程度。
- 使用 `自动分割标记工具` 标记对象。
    - +Point：添加属于对象的点。
    - -Point：删除您希望从对象中排除的点。
    - +Rect：绘制包含对象的矩形。Segment Anything 将自动分割对象。
    - 清除：清除所有自动分割标记。
    - 完成对象 (f)：完成当前标记。完成对象后，您可以输入标签名称并保存对象。

### 📜文档

- [帮助文档](./docs/Q&A.md)
- [模型库](./docs/models_list.md)
- [加载自定义模型](./docs/custom_model.md)

### 🧷快捷键

<details>

<summary>点击展开/关闭</summary>

| 快捷键         | 功能                                  |
|-----------------|---------------------------------------|
| d               | 打开下一个文件                        |
| a               | 打开上一个文件                        |
| p               | 创建多边形                            |
| o               | 创建旋转                              |
| r               | 创建矩形                              |
| i               | 运行模型                              |
| r               | 创建矩形                              |
| +               | SAM 模式下的 "+point"                 |
| -               | SAM 模式下的 "-point"                 |
| g               | 组合选定的形状                       |
| u               | 取消组合选定的形状                   |
| Ctrl + q        | 退出                                  |
| Ctrl + i        | 打开图像文件                          |
| Ctrl + o        | 打开视频文件                          |
| Ctrl + u        | 从目录加载所有图像                    |
| Ctrl + e        | 编辑标签                             |
| Ctrl + j        | 编辑多边形                           |
| Ctrl + d        | 复制多边形                           |
| Ctrl + p        | 切换保留先前模式                     |
| Ctrl + y        | 切换自动使用上一标签                 |
| Ctrl + m        | 一次运行所有图片                           |
| Ctrl + a        | 启用自动标注                         |
| Ctrl + s        | 保存当前信息                         |
| Ctrl + Shift + s | 更改输出目录                        |
| Ctrl -          | 缩小                                  |
| Ctrl + 0        | 缩放到原始大小                        |
| [Ctrl++, Ctrl+=] | 放大                              |
| Ctrl + f        | 适应窗口大小                         |
| Ctrl + Shift + f | 适应宽度                           |
| Ctrl + z        | 撤销上次操作                         |
| Ctrl + Delete   | 删除文件                              |
| Delete          | 删除多边形                            |
| Esc             | 取消选定的对象                        |
| Backspace       | 移除选定点                            |
| ↑→↓←           | 键盘箭头移动选定对象                  |
| zxcv            | 旋转选定的矩形框的键盘操作            |


</details>

## 📧 联系 [⏏️](#📄-目录)

<p align="center">
🤗 亲，给个 Star 支持一下吧！ 🤗
</p>

如果您觉得这个项目有用或有趣，请考虑给它点赞以表示支持。如果您在使用这个项目时遇到任何问题或有任何疑问，请随时使用以下方式寻求帮助：


- [创建问题](https://github.com/CVHub520/X-AnyLabeling/issues)
- 邮箱: cv_hub@163.com
- 微信: `ww10874` （请在您的消息中包含`X-Anylabeing+问题的简要描述`）

## ✅ 许可 [⏏️](#📄-目录)

本项目采用 [GPL-3.0 开源许可证](./LICENSE)。

## 🏷️ 引用 [⏏️](#📄-目录)

### BibTeX

如果您在研究中使用了这个软件，请按照以下方式引用它：

```
@misc{X-AnyLabeling,
  year = {2023},
  author = {Wei Wang},
  publisher = {Github},
  organization = {CVHub},
  journal = {Github repository},
  title = {Advanced Auto Labeling Solution with Added Features},
  howpublished = {\url{https://github.com/CVHub520/X-AnyLabeling}}
}
```