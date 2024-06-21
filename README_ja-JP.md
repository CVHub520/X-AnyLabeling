<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img width="100%" src="https://user-images.githubusercontent.com/72010077/273420485-bdf4a930-8eca-4544-ae4b-0e15f3ebf095.png"></a>
  </p>

[English](README.md) | [简体中文](README_zh-CN.md) | 日本語

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

## 📄 目次

- [🥳 新機能](#🥳-新機能-⏏️)
- [👋 紹介](#👋-紹介-⏏️)
- [🔥 ハイライト](#🔥-ハイライト-⏏️)
  - [🗝️主な特徴](#🗝️主な特徴)
  - [⛏️モデル動物園](#⛏️モデル動物園)
- [📋 使い方](#📋-使い方-⏏️)
  - [📜 ドキュメント](#📜-ドキュメント-⏏️)
    - [🔜クイックスタート](#🔜クイックスタート-⏏️)
    - [📋ユーザーガイド](#📋ユーザーガイド-⏏️)
    - [🚀カスタムモデルの読み込み](#🚀カスタムモデルの読み込み-⏏️)
  - [🧷ホットキー](#🧷-ホットキー-⏏️)
- [📧 連絡先](#📧-連絡先-⏏️)
- [✅ ライセンス](#✅-ライセンス-⏏️)
- [🙏🏻 謝辞](#🙏🏻-謝辞-⏏️)
- [🏷️ 引用](#🏷️-引用-⏏️)

## 🥳 新機能 [⏏️](#📄-目次)

- 2024年6月:
  - [YOLOv8-Pose](https://docs.ultralytics.com/tasks/pose/)モデルをサポート。
  - [yolo-pose](./docs/ja/user_guide.md)のインポート/エクスポート機能を追加。
- 2024年5月:
  - ✨✨✨ [YOLOv8-World](https://docs.ultralytics.com/models/yolo-world)、[YOLOv8-oiv7](https://docs.ultralytics.com/models/yolov8)、[YOLOv10](https://github.com/THU-MIG/yolov10)モデルをサポート。
  - 🤗 最新バージョン[2.3.6](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.6)をリリース 🤗
  - 信頼度スコアの表示機能を追加。
- 2024年3月:
  - バージョン[2.3.5](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.5)をリリース。
- 2024年2月:
  - バージョン[2.3.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.4)をリリース。
  - ラベル表示機能を有効にする。
  - バージョン[2.3.3](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.3)をリリース。
  - バージョン[2.3.2](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.2)をリリース。
  - [YOLOv9](https://github.com/WongKinYiu/yolov9)モデルをサポート。
  - 水平バウンディングボックスを回転バウンディングボックスに変換する機能をサポート。
  - ラベルの削除と名前変更をサポート。詳細は[ドキュメント](./docs/ja/user_guide.md)を参照してください。
  - クイックタグ修正機能をサポート。詳細はこの[ドキュメント](./docs/en/user_guide.md)を参照してください。
  - バージョン[2.3.1](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.1)をリリース。
- 2024年1月:
  - サブイメージのクイックキャプチャ機能をサポート。
  - CLIPとSAMモデルを組み合わせて、より強力な意味理解と空間理解を実現。具体的には、この[例](./anylabeling/configs/auto_labeling/edge_sam_with_chinese_clip.yaml)を参照してください。
  - 深度推定タスクで[Depth Anything](https://github.com/LiheYoung/Depth-Anything.git)モデルをサポート。
  - バージョン[2.3.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.0)をリリース。
  - [YOLOv8-OBB](https://github.com/ultralytics/ultralytics)モデルをサポート。
  - [RTMDet](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet)および[RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)モデルをサポート。
  - YOLOv5に基づく[中国のナンバープレート](https://github.com/we0091234/Chinese_license_plate_detection_recognition)検出および認識モデルをリリース。
- 2023年12月:
  - バージョン[2.2.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.2.0)をリリース。
  - CPUおよびエッジデバイスでの効率的なセグメンテーションのための[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)をサポート。
  - YOLOv5-ClsおよびYOLOv8-Clsモデルをサポート。
- 2023年11月:
  - バージョン[2.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.1.0)をリリース。
  - [InternImage](https://arxiv.org/abs/2211.05778)モデル(**CVPR'23**)をサポート。
  - バージョン[2.0.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.0.0)をリリース。
  - Grounding-SAMを追加し、[GroundingDINO](https://github.com/wenyi5608/GroundingDINO)と[HQ-SAM](https://github.com/SysCV/sam-hq)を組み合わせて、sotaゼロショット高品質予測を実現！
  - [HQ-SAM](https://github.com/SysCV/sam-hq)モデルのサポートを強化し、高品質なマスク予測を実現。
  - [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/en/PULC/PULC_person_attribute_en.md)および[VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/en/PULC/PULC_vehicle_attribute_en.md)モデルをサポート。
  - 新しいマルチラベル属性アノテーション機能を導入。
  - バージョン[1.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v1.1.0)をリリース。
  - ポーズ推定：[YOLOv8-Pose](https://github.com/ultralytics/ultralytics)をサポート。
  - オブジェクトレベルのタグ付け機能を追加。
  - Grounding-DINOを使用した任意の未知カテゴリのバッチラベリングの新機能を追加。
- 2023年10月:
  - バージョン[1.0.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v1.0.0)をリリース。
  - 回転ボックスの新機能を追加。
  - [YOLOv5-OBB](https://github.com/hukaixuan19970627/yolov5_obb)と[DroneVehicle](https://github.com/VisDrone/DroneVehicle)および[DOTA](https://captain-whu.github.io/DOTA/index.html)-v1.0/v1.5/v2.0モデルをサポート。
  - SOTAゼロショットオブジェクト検出 - [GroundingDINO](https://github.com/wenyi5608/GroundingDINO)がリリースされました。
  - SOTAイメージタギングモデル - [Recognize Anything](https://github.com/xinyu1205/Tag2Text)がリリースされました。
  - YOLOv5-SAMおよびYOLOv8-EfficientViT_SAMユニオンタスクをサポート。
  - YOLOv5およびYOLOv8セグメンテーションタスクをサポート。
  - [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)および[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)モデルをリリース。
  - MOTアルゴリズム：[OC_Sort](https://github.com/noahcao/OC_SORT)(**CVPR'23**)をリリース。
  - [SAHI](https://github.com/obss/sahi)を使用した小物体検出の新機能を追加。
- 2023年9月:
  - バージョン[0.2.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.4)をリリース。
  - [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)(**ICCV'23**)、[SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D)、[MedSAM](https://arxiv.org/abs/2304.12306)およびYOLOv5-SAMをリリース。
  - [ByteTrack](https://github.com/ifzhang/ByteTrack)(**ECCV'22**)をMOTタスクにサポート。
  - [PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR)モデルをサポート。
  - `video`アノテーション機能を追加。
  - `yolo`/`coco`/`voc`/`mot`/`dota`エクスポート機能を追加。
  - 一度にすべての画像を処理する機能を追加。
- 2023年8月:
  - バージョン[0.2.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.0)をリリース。
  - [LVMSAM](https://arxiv.org/abs/2306.11925)およびそのバリエーション[BUID](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/buid)、[ISIC](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/isic)、[Kvasir](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/kvasir)をリリース。
  - レーン検出アルゴリズム：[CLRNet](https://github.com/Turoad/CLRNet)(**CVPR'22**)をサポート。
  - 2D人体全身ポーズ推定：[DWPose](https://github.com/IDEA-Research/DWPose/tree/main)(**ICCV'23 Workshop**)をサポート。
- 2023年7月:
  - [label_converter.py](./tools/label_converter.py)スクリプトを追加。
  - [RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/rtdetr/README.md