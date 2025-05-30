<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img alt="X-AnyLabeling" height="200px" src="https://github.com/user-attachments/assets/0714a182-92bd-4b47-b48d-1c5d7c225176"></a>
  </p>

[English](README.md) | [简体中文](README_zh-CN.md)

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

https://github.com/user-attachments/assets/f517fa94-c49c-4f05-864e-96b34f592079

https://github.com/user-attachments/assets/52cbdb5d-cc60-4be5-826f-903ea4330ca8

<div align="center"><strong>Text/Visual Prompting and Prompt-free for Detection & Segmentation</strong></div>

<br>

<img src="https://github.com/user-attachments/assets/7f43bcec-96fd-48d1-bd36-9e5a440a66f6" width="100%" />
<div align="center"><strong>Detect Anything</strong></div>

<br>

<img src="https://github.com/user-attachments/assets/208dc9ed-b8c9-4127-9e5b-e76f53892f03" width="100%" />
<div align="center"><strong>Segment Anything</strong></div>

<br>

<img src="https://github.com/user-attachments/assets/56c9a20b-c836-47aa-8b54-bad5bb99b735" width="100%" />
<div align="center"><strong>Chatbot</strong></div>


## 🥳 What's New

- Support `RMBG v2.0` model for image matting
- Bump version to [3.0.3](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v3.0.3)
- For more details, please refer to the [CHANGELOG](./CHANGELOG.md)


## X-AnyLabeling

**X-AnyLabeling** is a powerful annotation tool that integrates an AI engine for fast and automatic labeling. It's designed for multi-modal data engineers, offering industrial-grade solutions for complex tasks.

## Features

<img src="https://github.com/user-attachments/assets/c65db18f-167b-49e8-bea3-fcf4b43a8ffd" width="100%" />

- Processes both `images` and `videos`.
- Accelerates inference with `GPU` support.
- Allows custom models and secondary development.
- Supports one-click inference for all images in the current task.
- Enable import/export for formats like COCO, VOC, YOLO, DOTA, MOT, MASK, PPOCR, VLM-R1.
- Handles tasks like `classification`, `detection`, `segmentation`, `caption`, `rotation`, `tracking`, `estimation`, `ocr` and so on.
- Supports diverse annotation styles: `polygons`, `rectangles`, `rotated boxes`, `circles`, `lines`, `points`, and annotations for `text detection`, `recognition`, and `KIE`.


### Model library

| **Task Category** | **Supported Models** |
| :--- | :--- |
| 🖼️ Image Classification | YOLOv5-Cls, YOLOv8-Cls, YOLO11-Cls, InternImage, PULC |
| 🎯 Object Detection | YOLOv5/6/7/8/9/10, YOLO11/12, YOLOX, YOLO-NAS, D-FINE, DAMO-YOLO, Gold_YOLO, RT-DETR, RF-DETR |
| 🖌️ Instance Segmentation | YOLOv5-Seg, YOLOv8-Seg, YOLO11-Seg, Hyper-YOLO-Seg |
| 🏃 Pose Estimation | YOLOv8-Pose, YOLO11-Pose, DWPose, RTMO |
| 👣 Tracking | Bot-SORT, ByteTrack |
| 🔄 Rotated Object Detection | YOLOv5-Obb, YOLOv8-Obb, YOLO11-Obb |
| 📏 Depth Estimation | Depth Anything |
| 🧩 Segment Anything | SAM, SAM-HQ, SAM-Med2D, EdgeSAM, EfficientViT-SAM, MobileSAM, 
| ✂️ Image Matting | RMBG 1.4/2.0 |
| 💡 Proposal | UPN |
| 🏷️ Tagging | RAM, RAM++ |
| 📄 OCR | PP-OCR |
| 🗣️ VLM | Florence2 |
| 🛣️ Land Detection | CLRNet |
| 📍 Grounding | CountGD, GeCO, Grunding DINO, YOLO-World, YOLOE |
| 📚 Other | 👉 [model_zoo](./docs/en/model_zoo.md) 👈 |


## Docs

1. [Installation & Quickstart](./docs/en/get_started.md)
2. [Usage](./docs/en/user_guide.md)
3. [Customize a model](./docs/en/custom_model.md)
4. [Chatbot](./docs/en/chatbot.md)

## Examples

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


## Contact

If you find this project helpful, please give it a ⭐star⭐, and for any questions or issues, feel free to [create an issue](https://github.com/CVHub520/X-AnyLabeling/issues) or email cv_hub@163.com.


## License

This project is released under the [GPL-3.0 license](./LICENSE).


## Acknowledgement

I extend my heartfelt thanks to the developers and contributors of [AnyLabeling](https://github.com/vietanhdev/anylabeling), [LabelMe](https://github.com/wkentaro/labelme), [LabelImg](https://github.com/tzutalin/labelIm), [roLabelImg](https://github.com/cgvict/roLabelImg), [PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel) and [CVAT](https://github.com/opencv/cvat), whose work has been crucial to the success of this project.


## Citing

If you use this software in your research, please cite it as below:

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