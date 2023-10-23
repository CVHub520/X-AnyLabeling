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
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/CVHub520/X-AnyLabeling/stargazers"><img src="https://img.shields.io/github/stars/CVHub520/X-AnyLabeling?color=ccf"></a>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

<div align=center>
<img src="https://user-images.githubusercontent.com/72010077/273759618-31a716ed-3366-4fad-b564-20d99f7ab2e4.gif"/>
X-Anylabeling: Better, Faster, Stronger
</div>

</br>

<div align=center>
<img src="https://user-images.githubusercontent.com/72010077/274632730-bfbca9c6-ecb9-4dd2-b6d1-35523aac2322.gif"/>
GroundingDINO
</div>

## 📄 Table of Contents

- [🥳 What's New](#🥳-whats-new-⏏️)
- [👋 Brief Introduction](#👋-brief-introduction-⏏️)
- [🔥 Highlight](#🔥-highlight-⏏️)
  - [🗝️Key Features](#🗝️key-features)
  - [⛏️Model Zoo](#⛏️model-zoo)
- [📖 Tutorials](#📖-tutorials-⏏️)
  - [🔜Quick Start](#🔜quick-start)
  - [👨🏼‍💻Build from source](#👨🏼‍💻build-from-source)
  - [📦Build executable](#📦build-executable)
- [📋 Usage](#📋-usage-⏏️)
  - [📌Basic usage](#📌basic-usage)
  - [🚀Advanced usage](#🚀advanced-usage)
  - [📜Docs](#📜docs)
  - [🧷Hotkeys](#🧷hotkeys)
- [📧 Contact](#📧-contact-⏏️)
- [✅ License](#✅-license-⏏️)
- [🏷️ Citing](#🏷️-citing-⏏️)

## 🥳 What's New [⏏️](#📄-table-of-contents)

- Oct. 2023:
  - 🚀🚀🚀 Support [YOLOv5-OBB](https://github.com/hukaixuan19970627/yolov5_obb) with [DroneVehicle](https://github.com/VisDrone/DroneVehicle) and [DOTA](https://captain-whu.github.io/DOTA/index.html)-v1.0/v1.5/v2.0 model.
  - 🆕🆕🆕 Add a new feature for rotation box.
  - 🔥🔥🔥 SOTA Zero-Shot Object Detection - [GroundingDINO](https://github.com/wenyi5608/GroundingDINO) is released.
  - Support **YOLOv5-SAM** and **YOLOv8-EfficientViT_SAM** union task.
  - Release [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO) and [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) models.
  - Release MOT algorithms: [OC_Sort](https://github.com/noahcao/OC_SORT) (**CVPR'23**).
  - Add a new feature for small object detection using [SAHI](https://github.com/obss/sahi).
- Sep. 2023:
  - Release version [0.2.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.4).
  - Release [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit) (**ICCV'23**), [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D), [MedSAM](https://arxiv.org/abs/2304.12306) and YOLOv5-SAM.
  - Support [ByteTrack](https://github.com/ifzhang/ByteTrack) (**ECCV'22**) for MOT task.
  - Support [PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR) model.
  - Add `video` annotation feature.
  - Add `yolo`/`voc`/`mot` export functionality.
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
  - Release [YOLOv5](https://github.com/ultralytics/yolov5)-v7.0, [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), 


## 👋 Brief Introduction [⏏️](#📄-table-of-contents)

`X-AnyLabeling` is an awesome annotation tool built on [LabelImg](https://github.com/HumanSignal/labelImg), [Labelme](https://github.com/wkentaro/labelme) and [Anylabeling](https://github.com/vietanhdev/anylabeling). What sets it apart is that it not only provides a variety of leading-edge SOTA models but also prioritizes practical applications, aiming to create an industrial-grade, feature-rich tool to assist developers in effortlessly achieving automated annotation and data processing for various complex tasks.</br>
X-Anylabeling is designed to streamline the annotation workflow, allowing you to allocate more time to problem-solving and model optimization, thereby accelerating project progress and achieving outstanding results.

## 🔥 Highlight [⏏️](#📄-table-of-contents)

### 🗝️Key Features

- Support for importing `images` and `videos`.
- `CPU` and `GPU` inference support with on-demand selection.
- Compatibility with multiple SOTA deep-learning algorithms.
- Single-frame prediction and `one-click` processing for all images.
- Export options for formats like `COCO-JSON`, `VOC-XML`, `YOLOv5-TXT`, `DOTA-TXT` and `MOT-CSV`.
- Integration with popular frameworks such as [PaddlePaddle](https://www.paddlepaddle.org.cn/), [OpenMMLab](https://openmmlab.com/), [timm](https://github.com/huggingface/pytorch-image-models), and others.
- Providing comprehensive `help documentation` along with active `developer community support`.
- Accommodation of various visual tasks such as `detection`, `segmentation`, `face recognition`, and so on.
- Modular design that empowers users to compile the system according to their specific needs and supports customization and further development.
- Image annotation capabilities for `polygons`, `rectangles`, `rotation`, `circles`, `lines`, and `points`, as well as `text detection`, `recognition`, and `KIE` annotations.

### ⛏️Model Zoo

<div align="center">

| **Object Detection** | **SOD with [SAHI](https://github.com/obss/sahi)** | **Facial Landmark Detection** | **2D Pose Estimation** |
| :---: | :---: | :---: | :---: |
| <img src='https://user-images.githubusercontent.com/72010077/273488633-fc31da5c-dfdd-434e-b5d0-874892807d95.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095892-934be83a-f869-4a31-8e52-1074184149d1.jpg' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206095684-72f42233-c9c7-4bd8-9195-e34859bd08bf.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100220-ab01d347-9ff9-4f17-9718-290ec14d4205.gif' height="126px" width="180px"> |
|  **2D Lane Detection** | **OCR** | **MOT** | **Instance Segmentation** |
| <img src='https://user-images.githubusercontent.com/72010077/273764641-65f456ed-27ce-4077-8fce-b30db093b988.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273421210-30d20e08-3b72-4f4d-8976-05b564e13d87.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206111753-836e7827-968e-4c80-92ef-7a78766892fc.gif' height="126px" width="180px"  > | <img src='https://user-images.githubusercontent.com/61035602/206095831-cc439557-1a23-4a99-b6b0-b6f2e97e8c57.jpg' height="126px" width="180px"> |
|  **[SAM](https://segment-anything.com/)** | **BC-SAM** | **Skin-SAM** | **Polyp-SAM** |
| <img src='https://user-images.githubusercontent.com/72010077/273421331-2c0858b5-0b92-405b-aae6-d061bc25aa3c.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764259-718dce97-d04d-4629-b6d2-95f17670ce2a.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764288-e26767d1-3c44-45cb-a72e-124efb4e8263.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764318-e8b6a197-e733-478e-a210-e4386bafa1e4.png' height="126px" width="180px"> |

For more details, please refer to [models_list](./docs/models_list.md).

</div>

## 📖 Tutorials [⏏️](#📄-table-of-contents)

### 🔜Quick Start

Download and run the `GUI` version directly from [Release](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.4.0) or [Baidu Disk](https://pan.baidu.com/s/1Rn_YHJUZetuSzfanSvSynQ?pwd=snqt).

Note:
- For MacOS:
  - After installation, go to the Applications folder.
  - Right-click on the application and choose Open.
  - From the second time onwards, you can open the application normally using Launchpad.

- Due to the lack of necessary hardware, the current tool is only available in executable versions for `Windows` and `Linux`. If you require executable programs for other operating systems, e.g., `MacOS`, please refer to the following steps for self-compilation.

### 👨🏼‍💻Build from source

- Install the required libraries:

```bash
pip install -r requirements.txt
```

> If you need to use GPU inference, install the corresponding requirements-gpu.txt file and download the appropriate version of onnxruntime-gpu based on your local CUDA and CuDNN versions. For more details, refer to the [FAQ](./docs/Q&A.md).

- Generate resources [Option]:

```
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- Run the application:

```
python anylabeling/app.py
```

### 📦Build executable

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

<summary>Note:</summary>

1. Before compiling, please modify the `__preferred_device__` parameter in the "anylabeling/app_info.py" file according to the appropriate GPU/CPU version.
2. If you need to compile the GPU version, install the corresponding environment using "pip install -r requirements-gpu-dev.txt". Specifically, for compiling the Windows-GPU version, manually modify the "datas" list parameters in the "anylabeling-win-gpu.spec" file to include the relevant dynamic libraries (*.dll) of your local onnxruntime-gpu. Additionally, when downloading the onnxruntime-gpu package, ensure compatibility with your CUDA version. You can refer to the official documentation for the specific compatibility table.
3. For macOS versions, you can make modifications by referring to the "anylabeling-win-*.spec" script.

</details>

## 📋 Usage [⏏️](#📄-table-of-contents)

### 📌Basic usage

1. Build and launch using the instructions above.
2. Click `Change Output Dir` in the `Menu/File` to specify a output directory; otherwise, it will save by default in the current image path.
3. Click `Open`/`Open Dir`/`Open Video` to select a specific file, folder, or video.
4. Click the `Start drawing xxx` button on the left-hand toolbar or the `Auto Lalbeling` control to initiate. 
5. Click and release left mouse to select a region to annotate the rect box. Alternatively, you can press the "Run (i)" key for one-click processing.

> Note: The annotation will be saved to the folder you specify and you can refer to the below hotkeys to speed up your workflow.

### 🚀Advanced usage

- Select **AutoLalbeing Button** on the left side or press the shortcut key "Ctrl + A" to activate auto labeling.
- Select one of the `Segment Anything-liked Models` from the dropdown menu Model, where the Quant indicates the quantization of the model.
- Use `Auto segmentation marking tools` to mark the object.
    - +Point: Add a point that belongs to the object.
    - -Point: Remove a point that you want to exclude from the object.
    - +Rect: Draw a rectangle that contains the object. Segment Anything will automatically segment the object.
    - Clear: Clear all auto segmentation markings.
    - Finish Object (f): Finish the current marking. After finishing the object, you can enter the label name and save the object.

### 📜Docs

- [FAQ](./docs/Q&A.md)
- [Model Zoo](./docs/models_list.md)
- [Loading Custom Models](./docs/custom_model.md)

### 🧷Hotkeys

<details open>

<summary>Click to Expand/Collapse</summary>

| Shortcut          | Function                                |
|-------------------|-----------------------------------------|
| d                 | Open next file                          |
| a                 | Open previous file                      |
| p                 | Create polygon                          |
| o                 | Create rotation                         |
| r                 | Create rectangle                        |
| i                 | Run model                               |
| r                 | Create rectangle                        |
| +                 | `+point` of SAM mode                    |
| -                 | `-point` of SAM mode                    |
| g                 | Group selected shapes                   |
| u                 | Ungroup selected shapes                 |
| Ctrl + q          | Quit                                    |
| Ctrl + i          | Open image file                         |
| Ctrl + o          | Open video file                         |
| Ctrl + u          | Load all images from a directory        |
| Ctrl + e          | Edit label                              |
| Ctrl + j          | Edit polygon                            |
| Ctrl + d          | Duplicate polygon                       |
| Ctrl + p          | Toggle keep previous mode               |
| Ctrl + y          | Toggle auto use last label              |
| Ctrl + m          | Run all images at once                  |
| Ctrl + a          | Enable auto annotation                  |
| Ctrl + s          | Save current information                |
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
- WeChat: `ww10874` (Please include `X-Anylabeing+brief description of the issue` in your message)

## ✅ License [⏏️](#📄-table-of-contents)

This project is released under the [GPL-3.0 license](./LICENSE).

## 🏷️ Citing [⏏️](#📄-table-of-contents)

### BibTeX

If you use this software in your research, please cite it as below:

```
@misc{X-AnyLabeling,
  year = {2023},
  author = {CVHub},
  publisher = {Github},
  journal = {Github repository},
  title = {Advanced Auto Labeling Solution with Added Features},
  howpublished = {\url{https://github.com/CVHub520/X-AnyLabeling}}
}
```