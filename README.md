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
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/CVHub520/X-AnyLabeling/stargazers"><img src="https://img.shields.io/github/stars/CVHub520/X-AnyLabeling?color=ccf"></a>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)


<img src="https://github.com/user-attachments/assets/0b1e3c69-a800-4497-9bad-4332c1ce1ebf" width="100%" />
<div align="center"><strong>Segment Anything 2.1</strong></div>

<br>

[![Open Vision](https://github.com/user-attachments/assets/b2c1419b-540b-44fb-988e-a48572268df7)](https://www.youtube.com/watch?v=QtoVMiTwXqk)
<div align="center"><strong>Interactive Visual-Text Prompting for Generic Vision Tasks</strong></div>

</br>

| **Tracking by HBB Detection** | **Tracking by OBB Detection** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/be67d4f8-eb31-4bb3-887c-d954bb4a5d6d" width="100%" /> | <img src="https://github.com/user-attachments/assets/d85b1102-124a-4971-9332-c51fd2b1c47b" width="100%" /> |
| **Tracking by Instance Segmentation** | **Tracking by Pose Estimation** | 
| <img src="https://github.com/user-attachments/assets/8d412dc6-62c7-4bb2-9a1e-026448acf2bf" width="100%" /> | <img src="https://github.com/user-attachments/assets/bab038a7-3023-4097-bdcc-90e5009477c0" width="100%" /> |


## 🥳 What's New

- Dec. 2024:
  - 🍊🍊🍊 Added support for [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO) model.
  - 🎉🎉🎉 Release version [2.5.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.5.0).
  - 🤡🤡🤡 Added support for [Open Vision](./examples/detection/hbb/README.md) model. [[Youtube](https://www.youtube.com/watch?v=QtoVMiTwXqk) | [Bilibili](https://www.bilibili.com/video/BV1jyqrYyE74)]
  - 👻👻👻 Added support for [Segment Anything 2.1](./docs/en/model_zoo.md) model.
  - 🤗🤗🤗 Added support for [Florence-2](./examples/vision_language/florence2/README.md), a unified vision foundation model for multi-modal tasks.
- Nov. 2024:
  - ✨✨✨ Added support for the [UPN](./examples/detection/hbb/README.md) model to generate proposal boxes.
  - 🌟🌟🌟 Added support for [YOLOv5-SAHI](./anylabeling/configs/auto_labeling/yolov5s_sahi.yaml).
- Oct. 2024:
  - 🎯🎯🎯 Added support for [DocLayout-YOLO](examples/optical_character_recognition/document_layout_analysis/README.md) model.
- Sep. 2024:
  - Release version [2.4.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.4.4)
  - 🐻‍❄️🐻‍❄️🐻‍❄️ Added support for [YOLO11-Det/OBB/Pose/Seg/Track model](https://github.com/ultralytics/ultralytics).
  - 🧸🧸🧸 Added support for image matting based on [RMBG v1.4 model](https://huggingface.co/briaai/RMBG-1.4).
  - 🦄🦄🦄 Added support for interactive video object tracking based on [Segment-Anything-2](https://github.com/CVHub520/segment-anything-2). [[Tutorial](examples/interactive_video_object_segmentation/README.md)]

<br>

<details> 
<summary>Click to view more news.</summary>

- Aug. 2024:
  - Release version [2.4.1](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.4.1)
  - Support [tracking-by-det/obb/seg/pose](./examples/multiple_object_tracking/README.md) tasks.
  - Support [Segment-Anything-2](https://github.com/facebookresearch/segment-anything-2) model!
  - Support [Grounding-SAM2](./docs/en/model_zoo.md) model.
  - Support lightweight model for Japanese recognition.
- Jul. 2024:
  - Add PPOCR-Recognition and KIE import/export functionality for training PP-OCR task.
  - Add ODVG import/export functionality for training grounding task.
  - Add support to annotate KIE linking field.
  - Support [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) model.
  - Support [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2) model.
- Jun. 2024:
  - Support [YOLOv8-Pose](https://docs.ultralytics.com/tasks/pose/) model.
  - Add [yolo-pose](./docs/en/user_guide.md) import/export functionality.
- May. 2024:
  - Support [YOLOv8-World](https://docs.ultralytics.com/models/yolo-world), [YOLOv8-oiv7](https://docs.ultralytics.com/models/yolov8), [YOLOv10](https://github.com/THU-MIG/yolov10) model.
  - Release version [2.3.6](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.6).
  - Add feature to display confidence score.
- Mar. 2024:
  - Release version [2.3.5](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.5).
- Feb. 2024:
  - Release version [2.3.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.4).
  - Enable label display feature.
  - Release version [2.3.3](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.3).
  - Release version [2.3.2](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.2).
  - Support [YOLOv9](https://github.com/WongKinYiu/yolov9) model.
  - Support the conversion from a horizontal bounding box to a rotated bounding box.
  - Supports label deletion and renaming. For more details, please refer to the [document](./docs/zh_cn/user_guide.md).
  - Support for quick tag correction is available; please refer to this [document](./docs/en/user_guide.md) for guidance.
  - Release version [2.3.1](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.1).
- Jan. 2024:
  - Combining CLIP and SAM models for enhanced semantic and spatial understanding. An example can be found [here](./anylabeling/configs/auto_labeling/edge_sam_with_chinese_clip.yaml).
  - Add support for the [Depth Anything](https://github.com/LiheYoung/Depth-Anything.git) model in the depth estimation task.
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
</details>


## X-AnyLabeling

**X-AnyLabeling** is a powerful annotation tool that integrates an AI engine for fast and automatic labeling. It’s designed for visual data engineers, offering industrial-grade solutions for complex tasks.

## Features

<img src="https://github.com/user-attachments/assets/c65db18f-167b-49e8-bea3-fcf4b43a8ffd" width="100%" />

- Processes both `images` and `videos`.
- Accelerates inference with `GPU` support.
- Allows custom models and secondary development.
- Supports one-click inference for all images in the current task.
- Enable import/export for formats like COCO, VOC, YOLO, DOTA, MOT, MASK, PPOCR.
- Handles tasks like `classification`, `detection`, `segmentation`, `caption`, `rotation`, `tracking`, `estimation`, `ocr` and so on.
- Supports diverse annotation styles: `polygons`, `rectangles`, `rotated boxes`, `circles`, `lines`, `points`, and annotations for `text detection`, `recognition`, and `KIE`.


### Model library

<div align="center">

| **Object Detection** | **SOD with [SAHI](https://github.com/obss/sahi)** | **Facial Landmark Detection** | **Pose Estimation** |
| :---: | :---: | :---: | :---: |
| <img src='https://user-images.githubusercontent.com/72010077/273488633-fc31da5c-dfdd-434e-b5d0-874892807d95.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095892-934be83a-f869-4a31-8e52-1074184149d1.jpg' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206095684-72f42233-c9c7-4bd8-9195-e34859bd08bf.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100220-ab01d347-9ff9-4f17-9718-290ec14d4205.gif' height="126px" width="180px"> |
|  **Lane Detection** | **OCR** | **MOT** | **Instance Segmentation** |
| <img src='https://user-images.githubusercontent.com/72010077/273764641-65f456ed-27ce-4077-8fce-b30db093b988.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273421210-30d20e08-3b72-4f4d-8976-05b564e13d87.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206111753-836e7827-968e-4c80-92ef-7a78766892fc.gif' height="126px" width="180px"  > | <img src='https://user-images.githubusercontent.com/61035602/206095831-cc439557-1a23-4a99-b6b0-b6f2e97e8c57.jpg' height="126px" width="180px"> |
|  **Tagging** | **Grounding** | **Recognition** | **Rotation** |
| <img src='https://user-images.githubusercontent.com/72010077/277670825-8797ac7e-e593-45ea-be6a-65c3af17b12b.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277395884-4d500af3-3e4e-4fb3-aace-9a56a09c0595.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277396071-79daec2c-6b0a-4d42-97cf-69fd098b3400.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277395955-aab54ea0-88f5-41af-ab0a-f4158a673f5e.png' height="126px" width="180px"> |
|  **Segment Anything** | **BC-SAM** | **Skin-SAM** | **Polyp-SAM** |
| <img src='https://user-images.githubusercontent.com/72010077/273421331-2c0858b5-0b92-405b-aae6-d061bc25aa3c.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764259-718dce97-d04d-4629-b6d2-95f17670ce2a.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764288-e26767d1-3c44-45cb-a72e-124efb4e8263.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764318-e8b6a197-e733-478e-a210-e4386bafa1e4.png' height="126px" width="180px"> |

For more details, please refer to 👉 [model_zoo](./docs/en/model_zoo.md) 👈

</div>


## Docs

1. [Installation & Quickstart](./docs/en/get_started.md)
2. [Usage](./docs/en/user_guide.md)
3. [Customize a model](./docs/en/custom_model.md)

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