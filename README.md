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
    <a href=""><img src="https://img.shields.io/pypi/v/x-anylabeling-cvhub?logo=pypi&logoColor=white"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/github/downloads/CVHub520/X-AnyLabeling/total?label=downloads"></a>
    <a href="https://modelscope.cn/collections/X-AnyLabeling-7b0e1798bcda43"><img src="https://img.shields.io/badge/modelscope-X--AnyLabeling-6750FF?link=https%3A%2F%2Fmodelscope.cn%2Fcollections%2FX-AnyLabeling-7b0e1798bcda43"></a>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

<img src="https://github.com/user-attachments/assets/8b5f290a-dddf-410c-a004-21e5a7bcd1cc" width="100%" />

<details>
<summary><strong>Auto-Training</strong></summary>

<video src="https://github.com/user-attachments/assets/c0ab2056-2743-4a2c-ba93-13f478d3481e" width="100%" controls>
</video>
</details>

<details>
<summary><strong>Auto-Labeling</strong></summary>

<video src="https://github.com/user-attachments/assets/f517fa94-c49c-4f05-864e-96b34f592079" width="100%" controls>
</video>
</details>

<details>
<summary><strong>Detect Anything</strong></summary>

<img src="https://github.com/user-attachments/assets/7f43bcec-96fd-48d1-bd36-9e5a440a66f6" width="100%" />
</details>

<details>
<summary><strong>Segment Anything</strong></summary>

<img src="https://github.com/user-attachments/assets/208dc9ed-b8c9-4127-9e5b-e76f53892f03" width="100%" />
</details>

<details>
<summary><strong>Promptable Concept Grounding</strong></summary>

<video src="https://github.com/user-attachments/assets/52cbdb5d-cc60-4be5-826f-903ea4330ca8" width="100%" controls>
</video>
</details>

<details>
<summary><strong>VQA</strong></summary>

<video src="https://github.com/user-attachments/assets/53adcff4-b962-41b7-a408-3afecd8d8c82" width="100%" controls>
</video>
</details>

<details>
<summary><strong>Chatbot</strong></summary>

<img src="https://github.com/user-attachments/assets/56c9a20b-c836-47aa-8b54-bad5bb99b735" width="100%" />
</details>

<details>
<summary><strong>Image Classifier</strong></summary>

<video src="https://github.com/user-attachments/assets/0652adfb-48a4-4219-9b18-16ff5ce31be0" width="100%" controls>
</video>
</details>

## 🥳 What's New

- Add text-prompted video object tracking feature based on [Segment Anything 3](./examples/interactive_video_object_segmentation/sam3/README.md) (#1258)
- Add support for [Segment Anything 3](./examples/grounding/sam3/README.md) model with text and visual promptable segmentation (#1207)
- Add TinyObj mode for Segment Anything Model to improve small object detection accuracy in high-resolution images by local cropping (#1193)
- For more details, please refer to the [CHANGELOG](./CHANGELOG.md)

## X-AnyLabeling

**X-AnyLabeling** is a powerful annotation tool that integrates an AI engine for fast and automatic labeling. It's designed for multi-modal data engineers, offering industrial-grade solutions for complex tasks.

<img src="https://github.com/user-attachments/assets/632e629b-0dec-407b-95a6-728052e1dd7b" width="100%" />

Also, we highly recommend trying out [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server), a simple, lightweight, and extensible framework that enables remote inference capabilities for X-AnyLabeling.

## Features

<img src="https://github.com/user-attachments/assets/c65db18f-167b-49e8-bea3-fcf4b43a8ffd" width="100%" />

- Supports remote inference service.
- Processes both `images` and `videos`.
- Accelerates inference with `GPU` support.
- Allows custom models and secondary development.
- Supports one-click inference for all images in the current task.
- Supports import/export for formats like COCO, VOC, YOLO, DOTA, MOT, MASK, PPOCR, MMGD, VLM-R1.
- Handles tasks like `classification`, `detection`, `segmentation`, `caption`, `rotation`, `tracking`, `estimation`, `ocr`, `vqa`, `grounding` and so on.
- Supports diverse annotation styles: `polygons`, `rectangles`, `rotated boxes`, `circles`, `lines`, `points`, and annotations for `text detection`, `recognition`, and `KIE`.

### Model library

<img src="https://github.com/user-attachments/assets/7da2da2e-f182-4a1b-85f6-bfd0dfcc6a1b" width="100%" />

| **Task Category** | **Supported Models** |
| :--- | :--- |
| 🖼️ Image Classification | YOLOv5-Cls, YOLOv8-Cls, YOLO11-Cls, InternImage, PULC |
| 🎯 Object Detection | YOLOv5/6/7/8/9/10, YOLO11/12, YOLOX, YOLO-NAS, D-FINE, DAMO-YOLO, Gold_YOLO, RT-DETR, RF-DETR, DEIMv2 |
| 🖌️ Instance Segmentation | YOLOv5-Seg, YOLOv8-Seg, YOLO11-Seg, Hyper-YOLO-Seg, RF-DETR-Seg |
| 🏃 Pose Estimation | YOLOv8-Pose, YOLO11-Pose, DWPose, RTMO |
| 👣 Tracking | Bot-SORT, ByteTrack, SAM2/3-Video |
| 🔄 Rotated Object Detection | YOLOv5-Obb, YOLOv8-Obb, YOLO11-Obb |
| 📏 Depth Estimation | Depth Anything |
| 🧩 Segment Anything | SAM 1/2/3, SAM-HQ, SAM-Med2D, EdgeSAM, EfficientViT-SAM, MobileSAM |
| ✂️ Image Matting | RMBG 1.4/2.0 |
| 💡 Proposal | UPN |
| 🏷️ Tagging | RAM, RAM++ |
| 📄 OCR | PP-OCRv4, PP-OCRv5 |
| 🗣️ Vision Foundation Models | Florence2 |
| 👁️ Vision Language Models | Qwen3-VL, Gemini, ChatGPT |
| 🛣️ Land Detection | CLRNet |
| 📍 Grounding | CountGD, GeCO, Grounding DINO, YOLO-World, YOLOE |
| 📚 Other | 👉 [model_zoo](./docs/en/model_zoo.md) 👈 |

## Docs

0. [Remote Inference Service](https://github.com/CVHub520/X-AnyLabeling-Server)
1. [Installation & Quickstart](./docs/en/get_started.md)
2. [Usage](./docs/en/user_guide.md)
3. [Command Line Interface](./docs/en/cli.md)
4. [Customize a model](./docs/en/custom_model.md)
5. [Chatbot](./docs/en/chatbot.md)
6. [VQA](./docs/en/vqa.md)
7. [Multi-class Image Classifier](./docs/en/image_classifier.md)

<img src="https://github.com/user-attachments/assets/0d67311c-f441-44b6-9ee0-932f25f51b1c" width="100%" />

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
- [iVOS](./examples/interactive_video_object_segmentation)
  - [SAM2-Video](./examples/interactive_video_object_segmentation/sam2/README.md)
  - [SAM3-Video](./examples/interactive_video_object_segmentation/sam3/README.md)
- [Matting](./examples/matting/)
  - [Image Matting](./examples/matting/image_matting/README.md)
- [Vision-Language](./examples/vision_language/)
  - [Florence 2](./examples/vision_language/florence2/README.md)
- [Counting](./examples/counting/)
  - [GeCo](./examples/counting/geco/README.md)
- [Grounding](./examples/grounding/)
  - [YOLOE](./examples/grounding/yoloe/README.md)
  - [SAM 3](./examples/grounding/sam3/README.md)
- [Training](./examples/training/)
  - [Ultralytics](./examples/training/ultralytics/README.md)


## Contribute

We believe in open collaboration! **X‑AnyLabeling** continues to grow with the support of the community. Whether you're fixing bugs, improving documentation, or adding new features, your contributions make a real impact.

To get started, please read our [Contributing Guide](./CONTRIBUTING.md) and make sure to agree to the [Contributor License Agreement (CLA)](./CLA.md) before submitting a pull request.

If you find this project helpful, please consider giving it a ⭐️ star! Have questions or suggestions? Open an [issue](https://github.com/CVHub520/X-AnyLabeling/issues) or email us at cv_hub@163.com.

A huge thank you 🙏 to everyone helping to make X‑AnyLabeling better.

## License

This project is licensed under the [GPL-3.0 license](./LICENSE) and is completely open source and free. The original intention is to enable more developers, researchers, and enterprises to conveniently use this AI application platform, promoting the development of the entire industry. We encourage everyone to use it freely (including commercial use), and you can also add features based on this project and commercialize it, but you must retain the brand identity and indicate the source project address.

Additionally, to understand the ecosystem and usage of X-AnyLabeling, if you use this project for academic, research, teaching, or enterprise purposes, please fill out the [registration form](https://forms.gle/MZCKhU7UJ4TRSWxR7). This registration is only for statistical purposes and will not incur any fees. We will strictly keep all information confidential.

X-AnyLabeling is independently developed and maintained by an individual. If this project has been helpful to you, we welcome your support through the donation links below to help sustain the project's continued development. Your support is the greatest encouragement! If you have any questions about the project or would like to collaborate, please feel free to contact via WeChat: ww10874 or email provided above.

## Sponsors

- [buy-me-a-coffee](https://ko-fi.com/cvhub520)
- [Wechat/Alipay](https://github.com/CVHub520/X-AnyLabeling/blob/main/README_zh-CN.md#%E8%B5%9E%E5%8A%A9)

## Acknowledgement

I extend my heartfelt thanks to the developers and contributors of [AnyLabeling](https://github.com/vietanhdev/anylabeling), [LabelMe](https://github.com/wkentaro/labelme), [LabelImg](https://github.com/tzutalin/labelImg), [roLabelImg](https://github.com/cgvict/roLabelImg), [PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel) and [CVAT](https://github.com/opencv/cvat), whose work has been crucial to the success of this project.

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

---

![Star History Chart](https://api.star-history.com/svg?repos=CVHub520/X-AnyLabeling&type=Date)

<div align="center"><a href="#top">🔝 Back to Top</a></div>
