<p align="center">
  <img alt="X-AnyLabeling" style="width: 128px; max-width: 100%; height: auto;" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/logo.png"/>
  <h1 align="center"> 💫 X-AnyLabeling 💫</h1>
  <p align="center">Effortless data labeling with AI support from <b>Segment Anything</b> and other awesome models!</p>
  <p align="center"><b>X-AnyLabeling: Advanced Auto Labeling Solution with Added Features</b></p>
</p>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)


**Auto Labeling with Segment Anything**

<a href="https://www.bilibili.com/video/BV1AV4y1U7h3/?spm_id_from=333.999.0.0&vd_source=938654fc70710bf1d11daa4b779d2418">
  <img style="width: 800px; margin-left: auto; margin-right: auto; display: block;" alt="AnyLabeling-SegmentAnything" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/demo.gif"/>
</a>


**Features:**

- [x] Image annotation for polygon, rectangle, circle, line and point.
- [x] Auto-labeling with YOLOv5 and Segment Anything.
- [x] Text detection, recognition and KIE (Key Information Extraction) labeling.
- [x] Multiple languages availables: English, Chinese.

**Highlight:**

- [x] Detection-Guided Fine-grained Classification.
- [x] Offer face detection with keypoint detection.
- [x] Provide advanced detectors, including YOLOX, YOLOv6, YOLOv7, YOLOv8, and DETR series.
- [x] Enables seamless conversion to industry-standard formats such as COCO-JSON, VOC-XML, and YOLOv5-TXT.

**🚀 New:**

- [x] Support YOLO-NAS [2023-06-15]
- [x] Support YOLOv8-Segmentation [2023-06-20]

## I. Install and run

### 1. Download and run executable

- Download and run newest version from [Releases](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.1.1).

- For MacOS:
  - After installing, go to Applications folder
  - Right click on the app and select Open
  - From the second time, you can open the app normally using Launchpad

Note: At present, we exclusively offer a graphical user interface (GUI) executable program designed specifically for the Windows operating system. For users on other operating systems, we provide instructions in [Step Ⅲ](#build) to compile the program independently.

### 2. Install from Pypi

Not ready yet, coming soon...

## II. Development

- Install packages

```bash
pip install -r requirements.txt
```

- Generate resources:

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- Run app:

```bash
python anylabeling/app.py
```

## III. Build executable <span id="build">Build</span>

- Install PyInstaller:

```bash
pip install -r requirements-dev.txt
```

- Build:

Note: Please replace the 'pathex' in the anylabeling.spec file according to the local conda environment before running.

```bash
bash build_executable.sh
```

- Check the outputs in: `dist/`.

## IV. References

- This project is built upon [Anylabeling](https://github.com/vietanhdev/anylabeling) and extends it with a variety of additional features. We would like to express our sincere gratitude to @[vietanhdev](https://github.com/vietanhdev) for open-sourcing such an awesome tool.
- Supports popular frameworks with [MMPreTrain](https://github.com/open-mmlab/mmpretrain), [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [timm](https://github.com/huggingface/pytorch-image-models), and etc.
- Auto-labeling with [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [Segment Anything Models](https://segment-anything.com/), [YOLO-NAS](https://github.com/Deci-AI/super-gradients).

## Contact 👋

Welcome to CVHub, a loving, fun, and informative platform for sharing computer vision expertise. We provide original, multidisciplinary, and in-depth interpretations of cutting-edge AI research papers, along with mature industrial-grade application solutions. We offer a one-stop service for academia, technology, and career needs.


| Platform | Account |
| --- | --- |
| Wechat 💬 | cv_huber |
| Zhihu  🧠 | [CVHub](https://www.zhihu.com/people/cvhub-40) |
| CSDN   📚 | [CVHub](https://blog.csdn.net/CVHub?spm=1010.2135.3001.5343) |
| Github 🐱 | [CVHub](https://github.com/CVHub520) |

If you have any questions or encounter any issues while using this project, please scan the QR code below and add me as a friend on WeChat with the note "X-AnyLabeling". I'll be happy to assist you!

![](https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/Wechat.jpg)
