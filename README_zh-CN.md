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
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/CVHub520/X-AnyLabeling/stargazers"><img src="https://img.shields.io/github/stars/CVHub520/X-AnyLabeling?color=ccf"></a>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

<video src="https://github.com/CVHub520/Resources/assets/72010077/a1fb281a-856c-493e-8989-84f4f783576b" 
       controls 
       width="100%" 
       height="auto" 
       style="max-width: 720px; height: auto; display: block; object-fit: contain;">
</video>

## 📄 目录

- [🥳 新功能](#🥳-新功能-⏏️)
- [👋 简介](#👋-简介-⏏️)
- [🔥 亮点](#🔥-亮点-⏏️)
  - [🗝️关键功能](#🗝️关键功能-)
  - [⛏️模型库](#⛏️模型库-)
- [📋 教程](#📋-教程-⏏️)
  - [📜 文档](#📜-文档-⏏️)
    - [🔜快速开始](#🔜-快速开始-⏏️)
    - [📋用户手册](#📋-用户手册-⏏️)
    - [🚀加载自定义模型](#🚀-加载自定义模型-⏏️)
  - [🧷快捷键](#🧷-快捷键-⏏️)
- [📧 联系](#📧-联系-⏏️)
- [✅ 许可](#✅-许可-⏏️)
- [🏷️ 引用](#🏷️-引用-⏏️)

## 🥳 新功能 [⏏️](#📄-目录)

- 2024年3月：
  - 🤗 发布[2.3.5](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.5)最新版本 🤗
- 2024年2月：
  - 发布[2.3.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.4)版本。
  - 支持标签显示功能。
  - 发布[2.3.3](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.3)版本。
  - ✨✨✨ 支持[YOLO-World](https://github.com/AILab-CVC/YOLO-World)模型。
  - 发布[2.3.2](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.2)版本。
  - 支持[YOLOv9](https://github.com/WongKinYiu/yolov9)模型。
  - 支持将水平框一键转换为旋转框。
  - 支持批量标签删除及重命名，详情可参考[用户手册](./docs/zh_cn/user_guide.md)。
  - 支持快速标签纠正功能，详情可参考[用户手册](./docs/zh_cn/user_guide.md)。
  - 发布[2.3.1](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.1)版本。
- 2024年1月：
  - 支持一键截取子图功能。
  - 👏👏👏 结合CLIP和SAM模型，实现更强大的语义和空间理解。具体可参考此[示例](./anylabeling/configs/auto_labeling/edge_sam_with_chinese_clip.yaml)。
  - 🔥🔥🔥 在深度估计任务中增加对[Depth Anything](https://github.com/LiheYoung/Depth-Anything.git)模型的支持。
  - 发布[2.3.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.0)版本。
  - 支持 [YOLOv8-OBB](https://github.com/ultralytics/ultralytics) 模型。
  - 支持 [RTMDet](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet) 和 [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) 模型。
  - 支持基于YOLOv5的[中文车牌](https://github.com/we0091234/Chinese_license_plate_detection_recognition)检测和识别模型。
- 2023年12月：
  - 发布[2.2.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.2.0)版本。
  - 支持CPU及边缘设备端高效分割一切推理模型：[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)。
  - 支持 YOLOv5-Cls 和 YOLOv8-Cls 图像分类模型。
- 2023年11月：
  - 发布[2.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.1.0)版本。
  - 支持[InternImage](https://arxiv.org/abs/2211.05778)图像分类模型（**CVPR'23**）。
  - 发布[2.0.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.0.0)版本。
  - 增加对Grounding-SAM的支持，结合[GroundingDINO](https://github.com/wenyi5608/GroundingDINO)和[HQ-SAM](https://github.com/SysCV/sam-hq)，实现sota零样本高质量预测！
  - 增强对[HQ-SAM](https://github.com/SysCV/sam-hq)模型的支持，实现高质量的掩码预测。
  - 支持 [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/en/PULC/PULC_person_attribute_en.md) 和 [VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/en/PULC/PULC_vehicle_attribute_en.md) 多标签分类模型。
  - 支持多标签属性分类标注功能。
  - 发布[1.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v1.1.0)版本。
  - 支持[YOLOv8-Pose](https://github.com/ultralytics/ultralytics)姿态估计模型。
- 2023年10月：
  - 发布[1.0.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v1.0.0)版本。
  - 添加旋转框的新功能。
  - 支持 [YOLOv5-OBB](https://github.com/hukaixuan19970627/yolov5_obb) 与 [DroneVehicle](https://github.com/VisDrone/DroneVehicle) 和 [DOTA](https://captain-whu.github.io/DOTA/index.html)-v1.0/v1.5/v2.0 旋转目标检测模型。
  - 支持SOTA级零样本目标检测：[GroundingDINO](https://github.com/wenyi5608/GroundingDINO)。
  - 支持SOTA级图像标签模型：[Recognize Anything](https://github.com/xinyu1205/Tag2Text)。
  - 支持 **YOLOv5-SAM** 和 **YOLOv8-EfficientViT_SAM** 联合检测及分割任务。
  - 支持 **YOLOv5** 和 **YOLOv8** 实例分割算法。
  - 支持 [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO) 和 [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) 模型。
  - 支持多目标跟踪算法：[OC_Sort](https://github.com/noahcao/OC_SORT)（**CVPR'23**）。
  - 添加使用[SAHI](https://github.com/obss/sahi)进行小目标检测的新功能。
- 2023年9月：
  - 发布[0.2.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.4)版本。
  - 支持[EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)（**ICCV'23**），[SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D)，[MedSAM](https://arxiv.org/abs/2304.12306) 和 YOLOv5-SAM 模型。
  - 支持 [ByteTrack](https://github.com/ifzhang/ByteTrack)（**ECCV'22**）用于MOT任务。
  - 支持 [PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR) 模型。
  - 支持视频解析功能。
  - 开发`yolo`/`coco`/`voc`/`mot`/`dota`/`mask`一键导入及导出功能。
  - 开发一键运行功能。
- 2023年8月：
  - 发布[0.2.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.0)版本。
  - 支持[LVMSAM](https://arxiv.org/abs/2306.11925) 及其变体 [BUID](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/buid)，[ISIC](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/isic)，[Kvasir](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/kvasir)。
  - 支持车道检测算法：[CLRNet](https://github.com/Turoad/CLRNet)（**CVPR'22**）。
  - 支持2D人体全身姿态估计：[DWPose](https://github.com/IDEA-Research/DWPose/tree/main)（**ICCV'23 Workshop**）。
- 2023年7月：
  - 添加[label_converter.py](./tools/label_converter.py)脚本。
  - 发布[RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/rtdetr/README.md)模型。
- 2023年6月：
  - 支持[YOLO-NAS](https://github.com/Deci-AI/super-gradients/tree/master)模型。
  - 支持[YOLOv8-seg](https://github.com/ultralytics/ultralytics)实例分割模型。
- 2023年5月：
  - 发布[0.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.1.0)版本。
  - 支持用于人脸检测和关键点识别的[YOLOv6-Face](https://github.com/meituan/YOLOv6/tree/yolov6-face)模型。
  - 支持[SAM](https://arxiv.org/abs/2304.02643)及蒸馏版本[MobileSAM](https://arxiv.org/abs/2306.14289)模型。
  - 支持[YOLOv5](https://github.com/ultralytics/yolov5)，[YOLOv6](https://github.com/meituan/YOLOv6)，[YOLOv7](https://github.com/WongKinYiu/yolov7)，[YOLOv8](https://github.com/ultralytics/ultralytics)，[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)模型。


## 👋 简介 [⏏️](#📄-目录)

`X-AnyLabeling` 是一款基于AI推理引擎和丰富功能特性于一体的强大辅助标注工具，其专注于实际应用，致力于为图像数据工程师提供工业级的一站式解决方案，可自动快速进行各种复杂任务的标定。


## 🔥 亮点 [⏏️](#📄-目录)

### 🗝️关键功能

- 支持`GPU`推理加速；
- 支持`图像`和`视频`处理；
- 支持单帧和批量预测所有任务；
- 支持自定义模型和二次开发设计；
- 支持一键导入和导出主流的标签格式，如COCO\VOC\YOLO\DOTA\MOT\MASK；
- 支持多种图像标注样式，包括 `多边形`、`矩形`、`旋转框`、`圆形`、`线条`、`点`，以及 `文本检测`、`识别` 和 `KIE` 标注；
- 支持各类视觉任务，如`图像分类`、`目标检测`、`实例分割`、`姿态估计`、`旋转检测`、`多目标跟踪`、`光学字符识别`、`图像文本描述`、`车道线检测`、`分割一切`系列等。


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

更多详情，请参考>>>[模型库](./docs/zh_cn/model_zoo.md)

</div>

## 📋 教程 [⏏️](#📄-目录)

- ### 📜文档

  - ##### 🔜[快速开始](./docs/zh_cn/get_started.md)

  - ##### 📋[用户手册](./docs/zh_cn/user_guide.md)

  - ##### 🚀[加载自定义模型](./docs/zh_cn/custom_model.md)

- ### 🧷快捷键

<details>

<summary>点击展开/关闭</summary>

| 快捷键            | 功能                                    |
|-------------------|----------------------------------------|
| d                 | 打开下一个文件                          |
| a                 | 打开上一个文件                          |
| p 或 [Ctrl+n]     | 创建多边形                              |
| o                 | 创建旋转框                              |
| r 或 [Crtl+r]     | 创建矩形框                              |
| i                 | 运行模型                                |
| q                 | `SAM 模式` 的正样本点                   |
| e                 | `SAM 模式` 的负样本点                    |
| b                 | `SAM 模式` 快速清除提示点               |
| g                 | 组合选定的对象                         |
| u                 | 取消组合选定的对象                     |
| s                 | 隐藏选定的对象                         |
| w                 | 显示选定的对象                         |
| Ctrl + q          | 退出当前应用程序                        |
| Ctrl + i          | 打开图像文件                           |
| Ctrl + o          | 打开视频文件                           |
| Ctrl + u          | 从目录加载所有图像                    |
| Ctrl + e          | 编辑标签                               |
| Ctrl + j          | 编辑多边形                             |
| Ctrl + c          | 复制选定的对象                         |
| Ctrl + v          | 粘贴选定的对象                         |
| Ctrl + d          | 复制多边形                             |
| Ctrl + g          | 显示当前任务的标注统计                       |
| Ctrl + h          | 切换可见性对象                         |
| Ctrl + p          | 切换保留上一个模式                     |
| Ctrl + y          | 切换自动使用上一个标签                |
| Ctrl + m          | 唤醒批量标注                       |
| Ctrl + a          | 启用自动标注                           |
| Ctrl + s          | 保存当前标注                           |
| Ctrl + l          | 显示/隐藏标签                       |
| Ctrl + t          | 显示/隐藏文本                       |
| Ctrl + Shift + s  | 更改输出目录                           |
| Ctrl -            | 缩小                                   |
| Ctrl + 0          | 缩放至原始大小                         |
| [Ctrl++, Ctrl+=]  | 放大                                   |
| Ctrl + f          | 适应窗口                               |
| Ctrl + Shift + f  | 适应宽度                               |
| Ctrl + z          | 撤销上一操作                           |
| Ctrl + Delete     | 删除文件                               |
| Delete            | 删除多边形                             |
| Esc               | 取消选择的对象                         |
| Backspace         | 删除选定的点                           |
| ↑→↓←              | 键盘箭头移动选定的对象                 |
| zxcv              | 键盘旋转选定的矩形框                   |


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

<div align="right"><a href="#top">🔝 Back to Top</a></div>
