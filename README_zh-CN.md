<p align="center">
  <img alt="X-AnyLabeling" style="width: 128px; max-width: 100%; height: auto;" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/logo.png"/>
  <h1 align="center"> 💫 X-AnyLabeling 💫</h1>
  <p align="center"><b>X-AnyLabeling：一款多 SOTA 模型集成的高级自动标注工具！</b></p>
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-LGPL%20v3-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/CVHub520/X-AnyLabeling/stargazers"><img src="https://img.shields.io/github/stars/CVHub520/X-AnyLabeling?color=ccf"></a>
</p>

<div align="center">


👉[帮助文档](./docs/Q&A.md)👈

简体中文 | [English](README_us-EN.md)

</div>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

<a href="https://www.bilibili.com/video/BV1AV4y1U7h3/?spm_id_from=333.999.0.0">
  <img style="width: 800px; margin-left: auto; margin-right: auto; display: block;" alt="AnyLabeling-SegmentAnything" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/demo.gif"/>
</a>
<p style="text-align: center; margin-top: 10px;">使用 Segment Anything 轻松进行自动标注</p>


**😀基础特性：**

- [x] 支持图像、视频一键导入功能。
- [x] 支持单帧预测及一键运行所有图片。
- [x] 支持 `SAM`、`YOLO`、`DETR` 等多个主流模型。
- [x] 支持分类、检测、分割、人脸、姿态估计等多种视觉任务。
- [x] 支持 `PaddlePadlle`、`OpenMMLab`、`timm` 等多个主流框架。
- [x] 支持`COCO-JSON`、`VOC-XML`、`YOLOv5-TXT`、`MOT-CSV`导出格式。
- [x] 支持多边形、矩形、圆形、直线和点的图像标注以及文本检测、识别和KIE（关键信息提取）标注。

**🔥亮点功能：**

- Segment Anything Model
  - [SAM](https://arxiv.org/abs/2304.02643): 通用自然图像分割一切模型；
  - [MobileSAM](https://arxiv.org/abs/2306.14289): 快速版 `SAM`；
  - [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D): 2D 医学图像分割一切模型（推荐）🤗;
  - [EfficientViT-SAM](https://github.com/CVHub520/efficientvit/tree/main): 高效语义分割模型 (ICCV 2023) 🆕;
  - [MedSAM](https://arxiv.org/abs/2304.12306): 通用医学图像分割一切模型；
  - [LVMSAM](https://arxiv.org/abs/2306.11925)
      - [BUID](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/buid): 超声乳腺癌分割模型；
      - [ISIC](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/isic): 皮肤镜病灶分割模型；
      - [Kvasir](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/kvasir): 结直肠息肉分割模型；
- Object Detection
  - [YOLOv5](https://github.com/ultralytics/yolov5)
  - [YOLOv6](https://github.com/meituan/YOLOv6)
  - [YOLOv7](https://github.com/WongKinYiu/yolov7)
  - [YOLOv8](https://github.com/ultralytics/ultralytics)
  - [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
  - [YOLO-NAS](https://github.com/Deci-AI/super-gradients/tree/master)
  - [RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/rtdetr/README.md)
- Image Segmentation
  - [YOLOv8-seg](https://github.com/ultralytics/ultralytics)
- Pose Estimation
  - [YOLOv6-Face](https://github.com/meituan/YOLOv6/tree/yolov6-face)：人脸关键点检测模型；
  - [DWPose](https://github.com/IDEA-Research/DWPose/tree/main): 全身人体姿态估计模型；
- Union Task
  - YOLOv5-ResNet：检测+分类级联模型；
  - YOLOv5-SAM
- Lane Detection
  - [CLRNet](https://github.com/Turoad/CLRNet) 
- OCR
  - [PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR)
- MOT
  - [ByteTrack](https://github.com/ifzhang/ByteTrack)

更多详情，请点击[模型列表](./docs/models_list.md)（持续更新中）

## 一、安装和运行

### 1.1 可执行文件

- 从[百度网盘(提取码: h5pl)](https://pan.baidu.com/s/1WPMGV4INKtQGs1nCJZ6Iwg?pwd=h5pl)下载并运行`GUI`版本直接运行。

注意：
- 对于MacOS：
  -  安装完成后，转到Applications文件夹。
  - 右键单击应用程序并选择打开。
  - 从第二次开始，您可以使用Launchpad正常打开应用程序。

- 目前我们仅为`Windows`和`Linux`系统提供带有图形用户界面（GUI）的可执行程序。对于其他操作系统的用户，您可以按照[步骤二](#build)的说明自行编译程序。


### 1.2 源码运行

- 安装基础依赖库

```bash
pip install -r requirements.txt
```

> 如果需要使用 GPU 推理，请根据需要安装对应的 `requirements-gpu.txt` 文件，并根据本机 `CUDA` 和 `CuDNN` 版本下载对应的 `onnxruntime-gpu` 版本，具体可参考[帮助文档](./docs/Q&A.md)。

- 生成资源：

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- 运行应用程序：

```bash
python anylabeling/app.py
```

## 二、<span id="build">打包编译</span>

- 安装 `PyInstaller`：

```bash
pip install -r requirements-dev.txt
# pip install -r requirements-gpu-dev.txt
```

- 构建：

请参考[帮助文档](./docs/Q&A.md)中的**工具使用**章节。

- 移步至目录 `dist/` 下检查输出。

## 三、其它

加载自定义模型等更多功能与问题反馈请参考[帮助文档](./docs/Q&A.md)。

## 许可证

本项目采用 [GPL-3.0 开源许可证](./LICENSE)。

## 引用

如果在你的研究中有使用到此项目，请参考以下格式引用它：

```
@misc{X-AnyLabeling,
title={Advanced Auto Labeling Solution with Added Features},
author={CVHub},
howpublished = {\url{https://github.com/CVHub520/X-AnyLabeling}},
year={2023}
}
```

## 👋联系我们

如果您在使用本项目的过程中有任何的疑问或碰到什么问题，请搜索微信号：`cv_huber`，备注 `X-Anylabeing+简要问题描述` 添加微信好友，我们将给予力所能及的帮助！