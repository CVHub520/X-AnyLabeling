<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img alt="X-AnyLabeling" height="200px" src="https://github.com/user-attachments/assets/0714a182-92bd-4b47-b48d-1c5d7c225176"></a>
  </p>

[简体中文](README_zh-CN.md) | [English](README.md)

</div>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-LGPL%20v3-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/github/v/release/CVHub520/X-AnyLabeling?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/github/downloads/CVHub520/X-AnyLabeling/total?label=downloads"></a>
    <a href="https://modelscope.cn/collections/X-AnyLabeling-7b0e1798bcda43"><img src="https://img.shields.io/badge/modelscope-X--AnyLabeling-6750FF?link=https%3A%2F%2Fmodelscope.cn%2Fcollections%2FX-AnyLabeling-7b0e1798bcda43"></a>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

<video src="https://github.com/user-attachments/assets/f517fa94-c49c-4f05-864e-96b34f592079" width="100%" controls>
</video>

</br>

<details>
<summary><strong>基于文本/视觉提示或免提示的检测和分割统一模型</strong></summary>

<video src="https://github.com/user-attachments/assets/52cbdb5d-cc60-4be5-826f-903ea4330ca8" width="100%" controls>
</video>
</details>

<details>
<summary><strong>检测一切</strong></summary>

<img src="https://github.com/user-attachments/assets/7f43bcec-96fd-48d1-bd36-9e5a440a66f6" width="100%" />
</details>

<details>
<summary><strong>分割一切</strong></summary>

<img src="https://github.com/user-attachments/assets/208dc9ed-b8c9-4127-9e5b-e76f53892f03" width="100%" />
</details>

<details>
<summary><strong>聊天机器人</strong></summary>

<img src="https://github.com/user-attachments/assets/56c9a20b-c836-47aa-8b54-bad5bb99b735" width="100%" />
</details>

<details>
<summary><strong>视觉问答</strong></summary>

<video src="https://github.com/user-attachments/assets/92807672-f101-475b-b790-9caa1c31da05" width="100%" controls>
</video>
</details>


## 🥳 新功能

- X-AnyLabeling [v3.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v3.1.0) 最新版本发布
- 更多详情，请参考[更新日志](./CHANGELOG.md)


## 简介

**X-AnyLabeling** 是一款基于AI推理引擎和丰富功能特性于一体的强大辅助标注工具，其专注于实际应用，致力于为多模态数据工程师提供工业级的一站式解决方案，可自动快速进行各种复杂任务的标定。


## 新特性

<img src="https://github.com/user-attachments/assets/c65db18f-167b-49e8-bea3-fcf4b43a8ffd" width="100%" />

- 支持`GPU`加速推理。
- 支持一键预测所有图像。
- 支持`图像`和`视频`处理。
- 支持自定义模型和二次开发。
- 支持一键导入和导出多种标签格式，如 COCO\VOC\YOLO\DOTA\MOT\MASK\PPOCR\VLM-R1 等；
- 支持多种图像标注样式，包括 `多边形`、`矩形`、`旋转框`、`圆形`、`线条`、`点`，以及 `文本检测`、`识别` 和 `KIE` 标注；
- 支持各类视觉任务，如`图像分类`、`目标检测`、`实例分割`、`姿态估计`、`旋转检测`、`多目标跟踪`、`光学字符识别`、`图像文本描述`、`车道线检测`、`分割一切`等。


### 模型库

| **任务类别** | **支持模型** |
| :--- | :--- |
| 🖼️ **图像分类** | YOLOv5-Cls, YOLOv8-Cls, YOLO11-Cls, InternImage, PULC |
| 🎯 **目标检测** | YOLOv5/6/7/8/9/10, YOLO11/12, YOLOX, YOLO-NAS, D-FINE, DAMO-YOLO, Gold_YOLO, RT-DETR, RF-DETR |
| 🖌️ **实例分割** | YOLOv5-Seg, YOLOv8-Seg, YOLO11-Seg, Hyper-YOLO-Seg |
| 🏃 **姿态估计** | YOLOv8-Pose, YOLO11-Pose, DWPose, RTMO |
| 👣 **目标跟踪** | Bot-SORT, ByteTrack |
| 🔄 **旋转目标检测** | YOLOv5-Obb, YOLOv8-Obb, YOLO11-Obb |
| 📏 **深度估计** | Depth Anything |
| 🧩 **分割一切** | SAM, SAM-HQ, SAM-Med2D, EdgeSAM, EfficientViT-SAM, MobileSAM |
| ✂️ **图像抠图** | RMBG 1.4/2.0 |
| 💡 **候选框提取** | UPN |
| 🏷️ **图像标记** | RAM, RAM++ |
| 📄 **光学字符识别** | PP-OCR |
| 🗣️ **视觉语言模型** | Florence2 |
| 🛣️ **车道线检测** | CLRNet |
| 📍 **Grounding** | CountGD, GeCO, Grunding DINO, YOLO-World, YOLOE |
| 📚 **其他** | 👉 [model_zoo](./docs/en/model_zoo.md) 👈 |


## 文档

1. [安装文档](./docs/zh_cn/get_started.md)
2. [用户手册](./docs/zh_cn/user_guide.md)
3. [自定义模型](./docs/zh_cn/custom_model.md)
4. [常见问题答疑](./docs/zh_cn/faq.md)
5. [聊天机器人](./docs/zh_cn/chatbot.md)
6. [视觉问答](./docs/zh_cn/vqa.md)


## 示例

- [Classification](./examples/classification/)
  - [Image-Level](./examples/classification/image-level/README.md)
  - [Shape-Level](./examples/classification/shape-level/README.md)
- [Detection](./examples/detection/)
  - [HBB Object Detection](./examples/detection/hbb/README.md)
  - [OBB Object Detection](./examples/detection/obb/README.md)
- [Segmentation](./examples/segmentation/README.md)
  - [Instance Segmentation](./examples/segmentation/instance_segmentation/)
  - [Binary Semantic Segmentation](./examples/segmentation/binary_semantic_segmentation/)
  - [Multiclass Semantic Segmentation](./examples/segmentation/multiclass_semantic_segmentation/)
- [Description](./examples/description/)
  - [Tagging](./examples/description/tagging/README.md)
  - [Captioning](./examples/description/captioning/README.md)
- [Estimation](./examples/estimation/)
  - [Pose Estimation](./examples/estimation/pose_estimation/README.md)
  - [Depth Estimation](./examples/estimation/depth_estimation/README.md)
- [OCR](./examples/optical_character_recognition/)
  - [Text Recognition](./examples/optical_character_recognition/text_recognition/)
  - [Key Information Extraction](./examples/optical_character_recognition/key_information_extraction/README.md)
- [MOT](./examples/multiple_object_tracking/README.md)
  - [Tracking by HBB Object Detection](./examples/multiple_object_tracking/README.md)
  - [Tracking by OBB Object Detection](./examples/multiple_object_tracking/README.md)
  - [Tracking by Instance Segmentation](./examples/multiple_object_tracking/README.md)
  - [Tracking by Pose Estimation](./examples/multiple_object_tracking/README.md)
- [iVOS](./examples/interactive_video_object_segmentation/README.md)
- [Matting](./examples/matting/)
  - [Image Matting](./examples/matting/image_matting/README.md)
- [Vision-Language](./examples/vision_language/)
  - [Florence 2](./examples/vision_language/florence2/README.md)
- [Counting](./examples/counting/)
  - [GeCo](./examples/counting/geco/README.md)


## 赞助

| **微信支付** | **支付宝** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/0178cf76-3627-426e-8432-ec031c9278ae" width="200px" /> | <img src="https://github.com/user-attachments/assets/87544ff8-3560-4696-b035-1fd26ecd162b" width="200px" /> |


感谢您的支持！


## 贡献指南

我们欢迎社区协作！**X‑AnyLabeling** 项目的成长离不开开发者们的共同参与，无论是修复 Bug、优化文档、还是添加新功能，您的贡献都非常宝贵。

在参与前请阅读我们的 [贡献指南](./CONTRIBUTING.md)，并在提交 Pull Request 前确认您已同意 [贡献者许可协议 (CLA)](./CLA.md)。

如果你觉得这个项目有帮助，请点亮右上角的⭐星标⭐。如有任何问题或疑问，欢迎[创建 issue](https://github.com/CVHub520/X-AnyLabeling/issues) 或发送邮件至 cv_hub@163.com。

衷心感谢每一位为项目贡献力量的朋友 🙏


## 许可

本项目遵循 [GPL-3.0 license](./LICENSE) 协议，个人非商业用途可免费使用。若用于学术、科研或教学目的，也可免费使用，但请在[此处](https://forms.gle/MZCKhU7UJ4TRSWxR7)填写登记表。如计划将本项目用于商业或企业环境，请务必联系微信申请商业授权: `ww10874`。


## 引用

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

---

![Star History Chart](https://api.star-history.com/svg?repos=CVHub520/X-AnyLabeling&type=Date)

<div align="center"><a href="#top">🔝 返回顶部</a></div>
