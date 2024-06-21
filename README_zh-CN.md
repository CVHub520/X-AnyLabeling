<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img width="100%" src="https://user-images.githubusercontent.com/72010077/273420485-bdf4a930-8eca-4544-ae4b-0e15f3ebf095.png"></a>
  </p>

[简体中文](README.zh-CN.md) | [English](README.md) | [日本語](README_ja-JP.md)

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

- 2024年6月:
  - 支持[yolov8-pose](https://docs.ultralytics.com/tasks/pose/)模型。
  - 支持[yolo-pose](./docs/zh_cn/user_guide.md)标签导入/导出功能。
- 2024年5月：
  - ✨✨✨ 支持[YOLOv8-World](https://docs.ultralytics.com/models/yolo-world), [YOLOv8-oiv7](https://docs.ultralytics.com/models/yolov8), [YOLOv10](https://github.com/THU-MIG/yolov10)模型。
  - 🤗 发布[2.3.6](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.6)最新版本 🤗
  - 支持显示模型预测得分。
- 2024年3月：
  - 发布[2.3.5](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.5)版本。
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
