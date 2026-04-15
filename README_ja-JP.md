<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img alt="X-AnyLabeling" height="200px" src="https://github.com/user-attachments/assets/0714a182-92bd-4b47-b48d-1c5d7c225176"></a>
  </p>

[日本語](README_ja-JP.md) | [简体中文](README_zh-CN.md) | [English](README.md)

</div>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-LGPL%20v3-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/github/v/release/CVHub520/X-AnyLabeling?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.11+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/github/downloads/CVHub520/X-AnyLabeling/total?label=downloads"></a>
    <a href="https://modelscope.cn/collections/X-AnyLabeling-7b0e1798bcda43"><img src="https://img.shields.io/badge/modelscope-X--AnyLabeling-6750FF?link=https%3A%2F%2Fmodelscope.cn%2Fcollections%2FX-AnyLabeling-7b0e1798bcda43"></a>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

<img src="https://github.com/user-attachments/assets/8b5f290a-dddf-410c-a004-21e5a7bcd1cc" width="100%" />

<details>
<summary><strong>自動トレーニング</strong></summary>

<video src="https://github.com/user-attachments/assets/c0ab2056-2743-4a2c-ba93-13f478d3481e" width="100%" controls>
</video>
</details>

<details>
<summary><strong>自動ラベル付け</strong></summary>

<video src="https://github.com/user-attachments/assets/f517fa94-c49c-4f05-864e-96b34f592079" width="100%" controls>
</video>
</details>

<details>
<summary><strong>すべて検出</strong></summary>

<img src="https://github.com/user-attachments/assets/7f43bcec-96fd-48d1-bd36-9e5a440a66f6" width="100%" />
</details>

<details>
<summary><strong>すべて分割</strong></summary>

<img src="https://github.com/user-attachments/assets/208dc9ed-b8c9-4127-9e5b-e76f53892f03" width="100%" />
</details>

<details>
<summary><strong>プロンプト可能な概念的根拠</strong></summary>

<video src="https://github.com/user-attachments/assets/52cbdb5d-cc60-4be5-826f-903ea4330ca8" width="100%" controls>
</video>
</details>

<details>
<summary><strong>VQA</strong></summary>

<video src="https://github.com/user-attachments/assets/53adcff4-b962-41b7-a408-3afecd8d8c82" width="100%" controls>
</video>
</details>

<details>
<summary><strong>チャットボット</strong></summary>

<img src="https://github.com/user-attachments/assets/56c9a20b-c836-47aa-8b54-bad5bb99b735" width="100%" />
</details>

<details>
<summary><strong>画像分類器</strong></summary>

<video src="https://github.com/user-attachments/assets/0652adfb-48a4-4219-9b18-16ff5ce31be0" width="100%" controls>
</video>
</details>

<details>
<summary><strong>OCR</strong></summary>

<video src="https://github.com/user-attachments/assets/493183fd-6cbe-45fb-9808-ec2b0af7a0f9" width="100%" controls>
</video>
</details>

## 🥳 新機能

<video src="https://github.com/user-attachments/assets/4a676ebf-d2ae-4327-b078-8e63a5323793" width="100%" controls>
</video>

- `2026-04-01`: 日本語および韓国語のUI言語に対応しました（`ja_JP`、`ko_KR`）。
- `2026-03-22`: GUIに組み込みの設定機能を追加し、一般的なオプションを直接調整できるようにしました。
- `2026-03-10`: 長方形から3D直方体の形状アノテーションを作成できる機能を追加しました。
- `2026-03-01`: PyQt5からPyQt6へのアップグレードに伴うリファクタリング（ベータ版）を完了し、機能の修正や最適化も実施しました。
- 詳細については、以下の [更新履歴](./CHANGELOG.md)

## X-AnyLabeling

**X-AnyLabeling** は、高速かつ自動的なラベリングを実現するAIエンジンを統合した、強力なアノテーションツールです。マルチモーダルデータエンジニア向けに設計されており、複雑なタスクに対応する産業レベルのソリューションを提供します。

<img src="https://github.com/user-attachments/assets/632e629b-0dec-407b-95a6-728052e1dd7b" width="100%" />

また、X-AnyLabelingのリモート推論機能を実現する、シンプルで軽量、かつ拡張性の高いフレームワークである[X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server)をぜひお試しください。

## 特徴

<img src="https://github.com/user-attachments/assets/c65db18f-167b-49e8-bea3-fcf4b43a8ffd" width="100%" />

- リモート推論サービスに対応しています。
- `画像`と`動画`の両方を処理します。
- `GPU`および`FFmpeg`のサポートにより、推論を高速化します。
- `英語`、`中国語`、`日本語`、`韓国語`でのUIローカライズに対応しています。
- カスタムモデルの利用や二次開発が可能です。
- 現在のタスク内のすべての画像に対して、ワンクリックでの推論が可能です。
- 以下の形式のインポート/エクスポートに対応しています。COCO\VOC\YOLO\DOTA\MOT\MASK\PPOCR\MMGF\VLM-R1など。
- 以下のようなタスクを処理します。 `分類`、`検出`、`セグメンテーション`、`キャプション`、`回転`、`追跡`、`推定`、`OCR`、`VQA`、`グラウンディング`など。
- さまざまな注釈スタイルに対応：`polygons`、`rectangles`、`cuboids`、`rotated boxes`、`quadrilaterals`、`circles`、`lines`、`line strips`、`points`、および`text detection`、`recognition`、`KIE`用の注釈。

### モデルライブラリ

<img src="https://github.com/user-attachments/assets/7da2da2e-f182-4a1b-85f6-bfd0dfcc6a1b" width="100%" />

| **タスクカテゴリ** | **対応モデル** |
| :--- | :--- |
| 🖼️ **画像分類** | YOLOv5-Cls, YOLOv8-Cls, YOLO11-Cls, InternImage, PULC |
| 🎯 **物体検出** | YOLOv5/6/7/8/9/10, YOLO11/12/26, YOLOX, YOLO-NAS, D-FINE, DAMO-YOLO, Gold_YOLO, RT-DETR, RF-DETR, DEIMv2 |
| 🖌️ **インスタンスセグメンテーション** | YOLOv5-Seg, YOLOv8-Seg, YOLO11-Seg, YOLO26-Seg, Hyper-YOLO-Seg, RF-DETR-Seg |
| 🏃 **姿勢推定** | YOLOv8-Pose, YOLO11-Pose, YOLO26-Pose, DWPose, RTMO |
| 👣 **トラッキング** | Bot-SORT, ByteTrack, SAM2/3-Video |
| 🔄 **回転オブジェクト検出** | YOLOv5-Obb, YOLOv8-Obb, YOLO11-Obb, YOLO26-Obb |
| 📏 **深度推定** | Depth Anything |
| 🧩 **セグメンテーション** | SAM 1/2/3, SAM-HQ, SAM-Med2D, EdgeSAM, EfficientViT-SAM, MobileSAM |
| ✂️ **画像マスキング** | RMBG 1.4/2.0 |
| 💡 **プロポーザル** | UPN |
| 🏷️ **タグ付け** | RAM, RAM++ |
| 📄 **OCR** | PP-OCRv4, PP-OCRv5, PP-DocLayoutV3, PaddleOCR-VL-1.5 |
| 🗣️ **ビジョン基盤モデル** | Rex-Omni, Florence2 |
| 👁️ **ビジョン言語モデル** | Qwen3-VL, Gemini, ChatGPT, GLM |
| 🛣️ **車線検出** | CLRNet |
| 📍 **グランディング** | CountGD, GeCO, Grounding DINO, YOLO-World, YOLOE |
| 📚 **その他** | 👉 [model_zoo](./docs/en/model_zoo.md) 👈 |

## ドキュメント

0. [リモート推論サービス](https://github.com/CVHub520/X-AnyLabeling-Server)
1. [インストールとクイックスタート](./docs/en/get_started.md)
2. [使用方法](./docs/en/user_guide.md)
3. [コマンドラインインターフェース](./docs/en/cli.md)
4. [モデルのカスタマイズ](./docs/en/custom_model.md)
5. [チャットボット](./docs/en/chatbot.md)
6. [VQA](./docs/en/vqa.md)
7. [多クラス画像分類器](./docs/en/image_classifier.md)

<img src="https://github.com/user-attachments/assets/0d67311c-f441-44b6-9ee0-932f25f51b1c" width="100%" />

## 例

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
  - [Rex-Omni](./examples/vision_language/rexomni/README.md)
  - [Florence 2](./examples/vision_language/florence2/README.md)
- [Counting](./examples/counting/)
  - [GeCo](./examples/counting/geco/README.md)
- [Grounding](./examples/grounding/)
  - [YOLOE](./examples/grounding/yoloe/README.md)
  - [SAM 3](./examples/grounding/sam3/README.md)
- [Training](./examples/training/)
  - [Ultralytics](./examples/training/ultralytics/README.md)


## 貢献について

私たちはオープンな協力を大切にしています！**X‑AnyLabeling**は、コミュニティのサポートによって成長し続けています。バグの修正、ドキュメントの改善、新機能の追加など、どのような貢献でも、大きな影響をもたらします。

まずは、[貢献ガイド](./CONTRIBUTING.md)をお読みいただき、プルリクエストを送信する前に[貢献者ライセンス契約 (CLA)](./CLA.md)に同意してください。

このプロジェクトが役に立ったと感じられたら、ぜひ⭐️スターを付けてください！質問や提案がある場合は、[イシュー](https://github.com/CVHub520/X-AnyLabeling/issues)を開くか、cv_hub@163.com までメールでお問い合わせください。

X‑AnyLabelingをより良いものにするために協力してくださる皆様に、心から感謝申し上げます 🙏。

## ライセンス

本プロジェクトは[GPL-3.0ライセンス](./LICENSE)の下で提供されており、完全にオープンソースかつ無料です。当初の目的は、より多くの開発者、研究者、企業がこのAIアプリケーションプラットフォームを便利に利用できるようにし、業界全体の発展を促進することにあります。皆様には（商用利用を含め）自由に利用していただくことを推奨しており、本プロジェクトをベースに機能を追加して商用化することも可能ですが、ブランドアイデンティティを維持し、ソースプロジェクトのアドレスを明記する必要があります。

また、X-AnyLabelingのエコシステムや利用状況を把握するため、学術、研究、教育、または企業目的で本プロジェクトをご利用になる場合は、[登録フォーム](https://forms.gle/MZCKhU7UJ4TRSWxR7)にご記入ください。この登録は統計目的のみであり、費用は一切かかりません。すべての情報は厳重に機密保持いたします。

X-AnyLabelingは個人によって独自に開発・維持管理されています。本プロジェクトがお役に立った場合は、プロジェクトの継続的な開発を支えるため、以下の寄付リンクを通じてご支援いただければ幸いです。皆様のご支援は、私たちにとって最大の励みとなります！プロジェクトに関するご質問やコラボレーションのご希望がございましたら、WeChat（ww10874）または上記のメールアドレスまでお気軽にお問い合わせください。

## スポンサー

- [buy-me-a-coffee](https://ko-fi.com/cvhub520)
- [Wechat/Alipay](https://github.com/CVHub520/X-AnyLabeling/blob/main/README_zh-CN.md#%E8%B5%9E%E5%8A%A9)

## 引用について

研究において本ソフトウェアをご利用になる場合は、以下の形式で引用してください：

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

<div align="center"><a href="#top">🔝 トップへ戻る</a></div>
