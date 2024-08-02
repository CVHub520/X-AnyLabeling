<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img width="100%" src="https://user-images.githubusercontent.com/72010077/273420485-bdf4a930-8eca-4544-ae4b-0e15f3ebf095.png"></a>
  </p>

[日本語](README_ja-JP.md) | [English](README.md) | [简体中文](README_zh-CN.md)

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
- [👋 概要](#👋-概要-⏏️)
- [🔥 ハイライト](#🔥-ハイライト-⏏️)
  - [🗝️主な機能](#🗝️主な機能)
  - [⛏️モデルズー](#⛏️モデルズー)
- [📋 使用方法](#📋-使用方法-⏏️)
  - [📜 ドキュメント](#📜-ドキュメント-⏏️)
    - [🔜クイックスタート](#🔜-クイックスタート-⏏️)
    - [📋ユーザーガイド](#📋-ユーザーガイド-⏏️)
    - [🚀カスタムモデルの読み込み](#🚀-カスタムモデルの読み込み-⏏️)
  - [🧷ホットキー](#🧷-ホットキー-⏏️)
- [📧 連絡先](#📧-連絡先-⏏️)
- [✅ ライセンス](#✅-ライセンス-⏏️)
- [🙏🏻 謝辞](#🙏🏻-謝辞-⏏️)
- [🏷️ 引用](#🏷️-引用-⏏️)

## 🥳 新機能 [⏏️](#📄-目次)

- 2024年6月:
  - [YOLOv8-Pose](https://docs.ultralytics.com/tasks/pose/)モデルをサポート。
  - [yolo-pose](./docs/en/user_guide.md)のインポート/エクスポート機能を追加。
- 2024年5月：
  - ✨✨✨ [YOLOv8-World](https://docs.ultralytics.com/models/yolo-world)、[YOLOv8-oiv7](https://docs.ultralytics.com/models/yolov8)、[YOLOv10](https://github.com/THU-MIG/yolov10)モデルをサポート。
  - 🤗 最新バージョン[2.3.6](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.6)をリリース 🤗
  - 信頼度スコアの表示機能を追加。
- 2024年3月：
  - バージョン[2.3.5](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.5)をリリース。
- 2024年2月：
  - バージョン[2.3.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.4)をリリース。
  - ラベル表示機能を有効化。
  - バージョン[2.3.3](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.3)をリリース。
  - バージョン[2.3.2](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.2)をリリース。
  - [YOLOv9](https://github.com/WongKinYiu/yolov9)モデルをサポート。
  - 水平バウンディングボックスから回転バウンディングボックスへの変換をサポート。
  - ラベルの削除と名前変更をサポート。詳細は[ドキュメント](./docs/zh_cn/user_guide.md)を参照してください。
  - クイックタグ修正機能をサポート。詳細はこの[ドキュメント](./docs/en/user_guide.md)を参照してください。
  - バージョン[2.3.1](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.1)をリリース。
- 2024年1月：
  - CLIPとSAMモデルを組み合わせて、強化されたセマンティックおよび空間理解を実現。例は[こちら](./anylabeling/configs/auto_labeling/edge_sam_with_chinese_clip.yaml)。
  - 深度推定タスクで[Depth Anything](https://github.com/LiheYoung/Depth-Anything.git)モデルのサポートを追加。
  - バージョン[2.3.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.3.0)をリリース。
  - [YOLOv8-OBB](https://github.com/ultralytics/ultralytics)モデルをサポート。
  - [RTMDet](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet)および[RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)モデルをサポート。
  - YOLOv5に基づく[中国ナンバープレート](https://github.com/we0091234/Chinese_license_plate_detection_recognition)検出および認識モデルをリリース。
- 2023年12月：
  - バージョン[2.2.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.2.0)をリリース。
  - エッジデバイスでの効率的な実行を最適化するために[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)をサポート。
  - YOLOv5-ClsおよびYOLOv8-Clsモデルをサポート。
- 2023年11月：
  - バージョン[2.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.1.0)をリリース。
  - [InternImage](https://arxiv.org/abs/2211.05778)モデル（**CVPR'23**）をサポート。
  - バージョン[2.0.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.0.0)をリリース。
  - Grounding-SAMのサポートを追加し、[GroundingDINO](https://github.com/wenyi5608/GroundingDINO)と[HQ-SAM](https://github.com/SysCV/sam-hq)を組み合わせて、sotaゼロショット高品質予測を実現！
  - [HQ-SAM](https://github.com/SysCV/sam-hq)モデルのサポートを強化し、高品質のマスク予測を実現。
  - マルチラベル分類タスクのための[PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/en/PULC/PULC_person_attribute_en.md)および[VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/en/PULC/PULC_vehicle_attribute_en.md)モデルをサポート。
  - 新しいマルチラベル属性アノテーション機能を導入。
  - バージョン[1.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v1.1.0)をリリース。
  - 姿勢推定をサポート：[YOLOv8-Pose](https://github.com/ultralytics/ultralytics)。
  - yolov5_ramを使用したオブジェクトレベルのタグをサポート。
  - Grounding-DINOに基づいて、任意の未知のカテゴリのバッチラベリングを可能にする新機能を追加。
- 2023年10月：
  - バージョン[1.0.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v1.0.0)をリリース。
  - 回転ボックスの新機能を追加。
  - [YOLOv5-OBB](https://github.com/hukaixuan19970627/yolov5_obb)と[DroneVehicle](https://github.com/VisDrone/DroneVehicle)および[DOTA](https://captain-whu.github.io/DOTA/index.html)-v1.0/v1.5/v2.0モデルをサポート。
  - SOTAゼロショットオブジェクト検出 - [GroundingDINO](https://github.com/wenyi5608/GroundingDINO)をリリース。
  - SOTA画像タグ付けモデル - [Recognize Anything](https://github.com/xinyu1205/Tag2Text)をリリース。
  - YOLOv5-SAMおよびYOLOv8-EfficientViT_SAMの統合タスクをサポート。
  - YOLOv5およびYOLOv8のセグメンテーションタスクをサポート。
  - [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)および[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)モデルをリリース。
  - MOTアルゴリズムをリリース：[OC_Sort](https://github.com/noahcao/OC_SORT)（**CVPR'23**）。
  - [SAHI](https://github.com/obss/sahi)を使用した小物体検出の新機能を追加。
- 2023年9月：
  - バージョン[0.2.4](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.4)をリリース。
  - [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)（**ICCV'23**）、[SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D)、[MedSAM](https://arxiv.org/abs/2304.12306)およびYOLOv5-SAMをリリース。
  - MOTタスクのために[ByteTrack](https://github.com/ifzhang/ByteTrack)（**ECCV'22**）をサポート。
  - [PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR)モデルをサポート。
  - `ビデオ`アノテーション機能を追加。
  - `yolo`/`coco`/`voc`/`mot`/`dota`エクスポート機能を追加。
  - すべての画像を一度に処理する機能を追加。
- 2023年8月：
  - バージョン[0.2.0]((https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.2.0))をリリース。
  - [LVMSAM](https://arxiv.org/abs/2306.11925)およびそのバリアント[BUID](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/buid)、[ISIC](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/isic)、[Kvasir](https://github.com/CVHub520/X-AnyLabeling/tree/main/assets/examples/kvasir)をリリース。
  - レーン検出アルゴリズムをサポート：[CLRNet](https://github.com/Turoad/CLRNet)（**CVPR'22**）。
  - 2D全身姿勢推定をサポート：[DWPose](https://github.com/IDEA-Research/DWPose/tree/main)（**ICCV'23 Workshop**）。
- 2023年7月：
  - [label_converter.py](./tools/label_converter.py)スクリプトを追加。
  - [RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/rtdetr/README.md)モデルをリリース。
- 2023年6月：
  - [YOLO-NAS](https://github.com/Deci-AI/super-gradients/tree/master)モデルをリリース。
  - インスタンスセグメンテーションをサポート：[YOLOv8-seg](https://github.com/ultralytics/ultralytics)。
  - X-AnyLabelingの[README_zh-CN.md](README_zh-CN.md)を追加。
- 2023年5月：
  - バージョン[0.1.0](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.1.0)をリリース。
  - 顔検出および顔ランドマーク検出のための[YOLOv6-Face](https://github.com/meituan/YOLOv6/tree/yolov6-face)をリリース。
  - [SAM](https://arxiv.org/abs/2304.02643)およびその高速バージョン[MobileSAM](https://arxiv.org/abs/2306.14289)をリリース。
  - [YOLOv5](https://github.com/ultralytics/yolov5)、[YOLOv6](https://github.com/meituan/YOLOv6)、[YOLOv7](https://github.com/WongKinYiu/yolov7)、[YOLOv8](https://github.com/ultralytics/ultralytics)、[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)をリリース。


## 👋 概要 [⏏️](#📄-目次)

`X-AnyLabeling`は、AI推論エンジンと高度な機能をシームレスに統合した強力なアノテーションツールです。実用的なアプリケーションに特化しており、画像データエンジニアに包括的で産業グレードのソリューションを提供することを目指しています。このツールは、多様で複雑なタスクに対して迅速かつ自動的にアノテーションを実行する能力に優れています。


## 🔥 ハイライト [⏏️](#📄-目次)

### 🗝️主な機能

- `GPU`を使用した推論の高速化をサポート。
- `画像`および`ビデオ`の処理をサポート。
- すべてのタスクに対して単一フレームおよびバッチ予測をサポート。
- モデルのカスタマイズと二次開発設計をサポート。
- COCO、VOC、YOLO、DOTA、MOT、MASKなどの主流のラベル形式のワンクリックインポートおよびエクスポートを可能にします。
- `分類`、`検出`、`セグメンテーション`、`キャプション`、`回転`、`追跡`、`推定`、`OCR`などの視覚タスクをカバー。
- `ポリゴン`、`矩形`、`回転ボックス`、`円`、`線`、`点`、および`テキスト検出`、`認識`、`KIE`のアノテーションスタイルをサポート。


### ⛏️モデルズー

<div align="center">

| **オブジェクト検出** | **[SAHI](https://github.com/obss/sahi)を使用したSOD** | **顔ランドマーク検出** | **2D姿勢推定** |
| :---: | :---: | :---: | :---: |
| <img src='https://user-images.githubusercontent.com/72010077/273488633-fc31da5c-dfdd-434e-b5d0-874892807d95.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095892-934be83a-f869-4a31-8e52-1074184149d1.jpg' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206095684-72f42233-c9c7-4bd8-9195-e34859bd08bf.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100220-ab01d347-9ff9-4f17-9718-290ec14d4205.gif' height="126px" width="180px"> |
|  **2Dレーン検出** | **OCR** | **MOT** | **インスタンスセグメンテーション** |
| <img src='https://user-images.githubusercontent.com/72010077/273764641-65f456ed-27ce-4077-8fce-b30db093b988.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273421210-30d20e08-3b72-4f4d-8976-05b564e13d87.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206111753-836e7827-968e-4c80-92ef-7a78766892fc.gif' height="126px" width="180px"  > | <img src='https://user-images.githubusercontent.com/61035602/206095831-cc439557-1a23-4a99-b6b0-b6f2e97e8c57.jpg' height="126px" width="180px"> |
|  **画像タグ付け** | **Grounding DINO** | **認識** | **回転** |
| <img src='https://user-images.githubusercontent.com/72010077/277670825-8797ac7e-e593-45ea-be6a-65c3af17b12b.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277395884-4d500af3-3e4e-4fb3-aace-9a56a09c0595.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277396071-79daec2c-6b0a-4d42-97cf-69fd098b3400.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/277395955-aab54ea0-88f5-41af-ab0a-f4158a673f5e.png' height="126px" width="180px"> |
|  **[SAM](https://segment-anything.com/)** | **BC-SAM** | **Skin-SAM** | **Polyp-SAM** |
| <img src='https://user-images.githubusercontent.com/72010077/273421331-2c0858b5-0b92-405b-aae6-d061bc25aa3c.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764259-718dce97-d04d-4629-b6d2-95f17670ce2a.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764288-e26767d1-3c44-45cb-a72e-124efb4e8263.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/72010077/273764318-e8b6a197-e733-478e-a210-e4386bafa1e4.png' height="126px" width="180px"> |

詳細については、👉 [model_zoo](./docs/en/model_zoo.md) 👈を参照してください。

</div>

## 📋 使用方法 [⏏️](#📄-目次)

- ### 📜ドキュメント

  - ##### 🔜[クイックスタート](./docs/en/get_started.md)

  - ##### 📋[ユーザーガイド](./docs/en/user_guide.md)

  - ##### 🚀[カスタムモデルの読み込み](./docs/en/custom_model.md)

- ### 🧷ホットキー

<details>

<summary>クリックして展開/折りたたむ</summary>

| ショートカット          | 機能                                |
|-------------------|-----------------------------------------|
| d                 | 次のファイルを開く                          |
| a                 | 前のファイルを開く                      |
| p または [Ctrl+n]     | ポリゴンを作成                          |
| o                 | 回転を作成                         |
| r または [Crtl+r]     | 矩形を作成                        |
| i                 | モデルを実行                               |
| q                 | SAMモードの`正のポイント`            |
| e                 | SAMモードの`負のポイント`            |
| b                 | SAMモードのポイントをすばやくクリア        |
| g                 | 選択した形状をグループ化                   |
| u                 | 選択した形状のグループ化を解除                 |
| s                 | 選択した形状を非表示                    |
| w                 | 選択した形状を表示                    |
| Ctrl + q          | 終了                                    |
| Ctrl + i          | 画像ファイルを開く                         |
| Ctrl + o          | ビデオファイルを開く                         |
| Ctrl + u          | ディレクトリからすべての画像を読み込む        |
| Ctrl + e          | ラベルを編集                              |
| Ctrl + j          | ポリゴンを編集                            |
| Ctrl + c          | 選択した形状をコピー                    |
| Ctrl + v          | 選択した形状を貼り付け                   |
| Ctrl + d          | ポリゴンを複製                       |
| Ctrl + g          | 概要アノテーション統計を表示  |
| Ctrl + h          | 可視性形状を切り替え                |
| Ctrl + p          | 前のモードを保持するかどうかを切り替え               |
| Ctrl + y          | 最後のラベルを自動的に使用するかどうかを切り替え              |
| Ctrl + m          | すべての画像を一度に実行                  |
| Ctrl + a          | 自動アノテーションを有効にする                  |
| Ctrl + s          | 現在のアノテーションを保存                 |
| Ctrl + l          | ラベルの可視性を切り替え                |
| Ctrl + t          | テキストの可視性を切り替え                 |
| Ctrl + Shift + s  | 出力ディレクトリを変更                 |
| Ctrl -            | ズームアウト                                |
| Ctrl + 0          | 元のサイズにズーム                        |
| [Ctrl++, Ctrl+=]  | ズームイン                                 |
| Ctrl + f          | ウィンドウに合わせる                              |
| Ctrl + Shift + f  | 幅に合わせる                               |
| Ctrl + z          | 最後の操作を元に戻す                 |
| Ctrl + Delete     | ファイルを削除                             |
| Delete            | ポリゴンを削除                          |
| Esc               | 選択したオブジェクトをキャンセル              |
| Backspace         | 選択したポイントを削除                   |
| ↑→↓←              | キーボードの矢印で選択したオブジェクトを移動 |
| zxcv              | キーボードで選択した矩形ボックスを回転    |


</details>


## 📧 連絡先 [⏏️](#📄-目次)

<p align="center">
🤗 このプロジェクトを楽しんでいますか？ ぜひスターを付けてください！ 🤗
</p>

このプロジェクトが役立つまたは興味深いと感じた場合は、スターを付けてサポートを示してください。また、このプロジェクトの使用中に質問や問題が発生した場合は、以下の方法で支援を求めてください：

- [問題を作成](https://github.com/CVHub520/X-AnyLabeling/issues)
- メール: cv_hub@163.com


## ✅ ライセンス [⏏️](#📄-目次)

このプロジェクトは[GPL-3.0ライセンス](./LICENSE)の下でリリースされています。

## 🙏🏻 謝辞 [⏏️](#📄-目次)

私は、[LabelMe](https://github.com/wkentaro/labelme)、[LabelImg](https://github.com/tzutalin/labelIm)、[roLabelImg](https://github.com/cgvict/roLabelImg)、[AnyLabeling](https://github.com/vietanhdev/anylabeling)、および[Computer Vision Annotation Tool](https://github.com/opencv/cvat)の開発者および貢献者に心から感謝します。彼らの献身と貢献が、このプロジェクトの成功に重要な役割を果たしました。

## 🏷️ 引用 [⏏️](#📄-目次)

### BibTeX

このソフトウェアを研究に使用する場合は、以下のように引用してください：

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
