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

- Support [Grounding-DINO-1.6-API](https://algos.deepdataspace.com/en#/model/grounding_dino) open-set object detection model
- Support [GeCo](./examples/counting/geco/README.md) zero-shot counting model
- For more details, please refer to the [Changelog](./docs/en/changelog.md)


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