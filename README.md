<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img width="100%" src="https://user-images.githubusercontent.com/72010077/273420485-bdf4a930-8eca-4544-ae4b-0e15f3ebf095.png"></a>
  </p>

[English](README.md) | [简体中文](README_zh-CN.md)

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

## 📄 Table of Contents

- [🥳 What's New](#🥳-whats-new-⏏️)
- [👋 Brief Introduction](#👋-brief-introduction-⏏️)
- [🔥 Highlight](#🔥-highlight-⏏️)
  - [🗝️Key Features](#🗝️key-features)
  - [⛏️Model Zoo](#⛏️model-zoo)
- [📋 Usage](#📋-usage-⏏️)
  - [📜 Docs](#📜-docs-⏏️)
    - [🔜Quick Start](#🔜quick-start-⏏️)
    - [📋User Guide](#📋quick-guide-⏏️)
    - [🚀Load Custom Model](#🚀load-custom-model-⏏️)
  - [🧷Hotkeys](#🧷-hotkeys-⏏️)
- [📧 Contact](#📧-contact-⏏️)
- [✅ License](#✅-license-⏏️)
- [🙏🏻 Acknowledgments](#🙏🏻-acknowledgments-⏏️)
- [🏷️ Citing](#🏷️-citing-⏏️)

## 🥳 What's New [⏏️](#📄-table-of-contents)

- Mar. 2024:
  - 🤗 Release the latest version [2.3.5](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.5) 🤗
- Feb. 2024:
  - Release version [2.3.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.4).
  - Enable label display feature.
  - Release version [2.3.3](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.3).
  - ✨✨✨ Support [YOLO-World](https://github.com/AILab-CVC/YOLO-World) model.
  - Release version [2.3.2](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.2).
  - Support [YOLOv9](https://github.com/WongKinYiu/yolov9) model.
  - Support the conversion from a horizontal bounding box to a rotated bounding box.
  - Supports label deletion and renaming. For more details, please refer to the [document](./docs/zh_cn/user_guide.md).
  - Support for quick tag correction is available; please refer to this [document](./docs/en/user_guide.md) for guidance.
  - Release version [2.3.1](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.1).
- Jan. 2024:
  - 👏👏👏 Combining CLIP and SAM models for enhanced semantic and spatial understanding. An example can be found [here](./anylabeling/configs/auto_labeling/edge_sam_with_chinese_clip.yaml).
  - 🔥🔥🔥 Adding support for the [Depth Anything](https://github.com/LiheYoung/Depth-Anything.git) model in the depth estimation task.
  - Release version [2.3.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.0).
  - Support [YOLOv8-OBB](https://github.com/ultralytics/ultralytics) model.
  - Support [RTMDet](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet) and [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) model.
  - Release a [chinese license plate](https://github.com/we0091234/Chinese_license_plate_detection_recognition) detection and recognition model based on YOLOv5.
- Dec. 2023:
  - Release version [2.2.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.2.0).
  - Support [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) to optimize for efficient execution on edge devices with minimal performance compromise.
  - Support YOLOv5-Cls and YOLOv8-Cls model.
- Nov. 2023:
  - Release version [2.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.1.0).
  - Support [InternImage](https://arxiv.org/abs/2211.05778) model (**CVPR'23**).
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
  - Support YOLOv5-SAM and YOLOv8-EfficientViT_SAM union task.
  - Support YOLOv5 and YOLOv8 segmentation task.
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


## 👋 Brief Introduction [⏏️](#📄-table-of-contents)

`X-AnyLabeling` stands out as a robust annotation tool seamlessly incorporating an AI inference engine alongside an array of sophisticated features. Tailored for practical applications, it is committed to delivering comprehensive, industrial-grade solutions for image data engineers. This tool excels in swiftly and automatically executing annotations across diverse and intricate tasks.


## 🔥 Highlight [⏏️](#📄-table-of-contents)

### 🗝️Key Features

- Supports inference acceleration using `GPU`.
- Handles both `image` and `video` processing.
- Allows single-frame and batch predictions for all tasks.
- Facilitates customization of models and supports secondary development design.
- Enables one-click import and export of mainstream label formats such as COCO, VOC, YOLO, DOTA, MOT, and MASK.
- Covers a range of visual tasks, including `classification`, `detection`, `segmentation`, `caption`, `rotation`, `tracking`, `estimation`, and `ocr`.
- Supports various image annotation styles, including `polygons`, `rectangles`, `rotated boxes`, `circles`, `lines`, `points`, as well as annotations for `text detection`, `recognition`, and `KIE`.


### ⛏️Model Zoo

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

For more details, please refer to 👉 [model_zoo](./docs/en/model_zoo.md) 👈

</div>

## 📋 Usage [⏏️](#📄-table-of-contents)

- ### 📜Docs

  - ##### 🔜[Quick Start](./docs/en/get_started.md)

  - ##### 📋[User Guide](./docs/en/user_guide.md)

  - ##### 🚀[Load Custom Model](./docs/en/custom_model.md)

- ### 🧷Hotkeys

<details>

<summary>Click to Expand/Collapse</summary>

| Shortcut          | Function                                |
|-------------------|-----------------------------------------|
| d                 | Open next file                          |
| a                 | Open previous file                      |
| p or [Ctrl+n]     | Create polygon                          |
| o                 | Create rotation                         |
| r or [Crtl+r]     | Create rectangle                        |
| i                 | Run model                               |
| q                 | `positive point` of SAM mode            |
| e                 | `negative point` of SAM mode            |
| b                 | Quickly clear points of SAM mode        |
| g                 | Group selected shapes                   |
| u                 | Ungroup selected shapes                 |
| s                 | Hide selected shapes                    |
| w                 | Show selected shapes                    |
| Ctrl + q          | Quit                                    |
| Ctrl + i          | Open image file                         |
| Ctrl + o          | Open video file                         |
| Ctrl + u          | Load all images from a directory        |
| Ctrl + e          | Edit label                              |
| Ctrl + j          | Edit polygon                            |
| Ctrl + c          | Copy selected shapes                    |
| Ctrl + v          | Paste selected shapes                   |
| Ctrl + d          | Duplicate polygon                       |
| Ctrl + g          | Display overview annotation statistics  |
| Ctrl + h          | Toggle visibility shapes                |
| Ctrl + p          | Toggle keep previous mode               |
| Ctrl + y          | Toggle auto use last label              |
| Ctrl + m          | Run all images at once                  |
| Ctrl + a          | Enable auto annotation                  |
| Ctrl + s          | Save current annotation                 |
| Ctrl + l          | Toggle visibility Labels                |
| Ctrl + t          | Toggle visibility Texts                 |
| Ctrl + Shift + s  | Change output directory                 |
| Ctrl -            | Zoom out                                |
| Ctrl + 0          | Zoom to Original                        |
| [Ctrl++, Ctrl+=]  | Zoom in                                 |
| Ctrl + f          | Fit window                              |
| Ctrl + Shift + f  | Fit width                               |
| Ctrl + z          | Undo the last operation                 |
| Ctrl + Delete     | Delete file                             |
| Delete            | Delete polygon                          |
| Esc               | Cancel the selected object              |
| Backspace         | Remove selected point                   |
| ↑→↓←              | Keyboard arrows to move selected object |
| zxcv              | Keyboard to rotate selected rect box    |


</details>


## 📧 Contact [⏏️](#📄-table-of-contents)

<p align="center">
🤗 Enjoying this project? Please give it a star! 🤗
</p>

If you find this project helpful or interesting, consider starring it to show your support, and if you have any questions or encounter any issues while using this project, feel free to reach out for assistance using the following methods:

- [Create an issue](https://github.com/CVHub520/X-AnyLabeling/issues)
- Email: cv_hub@163.com


## ✅ License [⏏️](#📄-table-of-contents)

This project is released under the [GPL-3.0 license](./LICENSE).

## 🙏🏻 Acknowledgments [⏏️](#📄-table-of-contents)

I extend my heartfelt thanks to the developers and contributors of the projects [LabelMe](https://github.com/wkentaro/labelme), [LabelImg](https://github.com/tzutalin/labelIm), [roLabelImg](https://github.com/cgvict/roLabelImg), [AnyLabeling](https://github.com/vietanhdev/anylabeling), and [Computer Vision Annotation Tool](https://github.com/opencv/cvat). Their dedication and contributions have played a crucial role in shaping the success of this project.

## 🏷️ Citing [⏏️](#📄-table-of-contents)

### BibTeX

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
