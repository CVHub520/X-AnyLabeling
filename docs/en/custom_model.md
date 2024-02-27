## 1. Loading Built-in Models

Currently, the built-in models in `X-AnyLabeling` are hosted on the GitHub release repository. Therefore, if you want to import a model directly from the GUI interface, it is essential to ensure a smooth internet connection; otherwise, you may encounter download failures.

For users who, due to network issues, are unable to download weight files smoothly from the interface, please refer to the [model_zoo.md](./model_zoo.md) to locate the corresponding weight file for the model you wish to load. Follow the step-by-step instructions provided in [#23](https://github.com/CVHub520/X-AnyLabeling/issues/23) for a demonstration.


## 2. Loading Custom Models

### 2.1 Adapted Models

**Adapted Models** refer to network models that have already been adapted within X-AnyLabeling. Similarly, you can refer to the [model_zoo.md](./model_zoo.md) document. Taking the [yolov5s](https://github.com/ultralytics/yolov5) model as an example, assuming a user has trained a custom `yolov5` detection model locally, you can download the corresponding [configuration file](../../anylabeling/configs/auto_labeling/yolov5s.yaml) as shown below:

```YAML
type: yolov5
name: yolov5s-r20230520
display_name: YOLOv5s Ultralytics
model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov5s.onnx
nms_threshold: 0.45
confidence_threshold: 0.25
classes:
  - person
  - bicycle
  - car
  - ...
```

A brief explanation of each field:

- `type`: Identifies the network type, providing a unique identifier for each model. This identifier is not user-modifiable.
- `name`: Defines the configuration file index marker for the current built-in model. If loading a user-defined model, this field may be left unset. For more details, refer to the [models.yaml](../../anylabeling/configs/auto_labeling/models.yaml) file.
- `display_name`: Used for the displayed name in the interface, customizable based on the specific task, such as `Fruits (YOLOv5s)`.
- `model_path`: Specifies the path to load the model weights. Note that this path is relative to the current configuration file. If necessary, you can also provide an absolute path. Ensure the file format is `*.onnx`.
- `nms_threshold`, `confidence_threshold`, and `classes` fields can be set according to the actual requirements. Additionally, `X-AnyLabeling` offers some auxiliary features, such as supporting `agnostic` and `filter_classes`. For example:

```YAML
type: yolov5
name: yolov5s-r20230520
display_name: YOLOv5s Ultralytics
model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov5s.onnx
nms_threshold: 0.45
confidence_threshold: 0.25
agnostic: True
filter_classes:
  - person
classes:
  - person
  - bicycle
  - car
  - ...
```

With the `filter_classes` setting, you can make the model detect only the `person` category, and the `agnostic` parameter is used to treat all objects as one category for NMS. Note that the current built-in model supports `yolov5` versions v6.0 and above. If you are using an older version like `v5.0`, be sure to specify the `anchors` and `stride` fields in the configuration file, as shown below:

```YAML
type: yolov5
...
stride: 32
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

Now, we understand the role of configuration files in the entire `X-AnyLabeling`. Suppose a user has trained a `yolov5s` detection model for detecting `apple`, `banana`, and `orange` categories. First, convert the `*.pt` file to `*.onnx` format, obtaining a model weight file named `fruits.onnx`. 

> Note: After the transformation, make sure to open the model using [Netron](https://netron.app/) tool to ensure alignment between the input and output nodes of the model and the built-in model within X-AnyLabeling. This step is crucial for compatibility and seamless integration.

Next, copy the configuration file corresponding to the current model from [model_zoo.md](./model_zoo.md) to the local machine and modify the corresponding hyperparameter fields as needed, such as detection threshold and categories. For example:

```YAML
type: yolov5
name: yolov5s-r20230520
display_name: Fruits (YOLOv5s)
model_path: fruits.onnx
input_width: 640
input_height: 640
nms_threshold: 0.45
confidence_threshold: 0.45
classes:
  - apple
  - banana
  - orange
```

Then, create a new folder and place the weight file and the corresponding configuration file in the same directory:

```
|- custom_model
|   |- fruits.onnx
|   |- fruits_yolov5s.yaml
```

Finally, open the GUI interface, click the `Model Icon` button, select `...Load Custom Model` and choose the `fruits_yolov5s.yaml` configuration file to complete the loading of the custom model.

### 2.2 Unadapted Models

**Unadapted Models** refer to models not built into the X-AnyLabeling tool. Follow the steps below to integrate them quickly:

- Define the configuration file

Take `yolov5` as an example, referring to this [configuration file](../../anylabeling/configs/auto_labeling/yolov5s.yaml).

- Add to task management

Refer to this [file](../../anylabeling/configs/auto_labeling/models.yaml) for adding according to a unified format.

- Define the model file

The core of this step is to inherit the [models](../../anylabeling/services/auto_labeling/model.py) class, implementing the corresponding pre-processing, model inference, and post-processing logic. Check an example for details.

- Add to model management

In the [model management file](../../anylabeling/services/auto_labeling/model_manager.py), add the newly defined custom model class.


## 3. Model Export

This section provides tutorials on converting PyTorch (`pt`) weight files to ONNX (`onnx`) to assist users in completing this conversion process quickly.

- [YOLOv5_OBB](https://github.com/hukaixuan19970627/yolov5_obb)

> Author: Kaixuan Hu

Refer to this [tutorial](https://github.com/CVHub520/yolov5_obb/tree/main).

- [YOLOv7](https://github.com/WongKinYiu/yolov7)

> Paper: YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors</br>
> Affiliation: Institute of Information Science, Academia Sinica, Taiwan

```bash
python export.py --weights yolov7.pt --img-size 640 --grid
```

- [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)

> Paper: Efficient object detectors including Gold-YOLO</br>
> Affiliation: huawei-noah</br>
> Published: NeurIPS23</br>

```bash
git clone https://github.com/huawei-noah/Efficient-Computing.git
cd Detection/Gold-YOLO
python deploy/ONNX/export_onnx.py --weights Gold_n_dist.pt --simplify --ort
                                            Gold_s_pre_dist.pt                     
                                            Gold_m_pre_dist.pt
                                            Gold_l_pre_dist.pt
```

- [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)

`DAMO-YOLO` is a fast and accurate object detection method developed by the TinyML Team from Alibaba DAMO Data Analytics and Intelligence Lab. It achieves higher performance than state-of-the-art YOLO series, extending YOLO with new technologies, including Neural Architecture Search (NAS) backbones, efficient Reparameterized Generalized-FPN (RepGFPN), a lightweight head with AlignedOTA label assignment, and distillation enhancement. For more details, refer to the Arxiv Report. Here you can find not only powerful models but also highly efficient training strategies and complete tools from training to deployment.

> Paper: DAMO-YOLO: A Report on Real-Time Object Detection</br>
> Affiliation: Alibaba Group</br>
> Published: Arxiv22</br>

```bash
git clone https://github.com/tinyvision/DAMO-YOLO.git
cd DAMO-YOLO
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640
```

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

Real-Time DEtection TRansformer (`RT-DETR`, aka RTDETR) is the first real-time end-to-end object detector known to the authors. RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS.

> Paper: RT-DETR: DETRs Beat YOLOs on Real-time Object Detection</br>
> Affiliation: Baidu</br>
> Published: Arxiv22</br>

Refer to this [article](https://zhuanlan.zhihu.com/p/628660998).

- [SAM](https://github.com/vietanhdev/samexporter)

The Segment Anything Model (`SAM`) produces high-quality object masks from input prompts such as points or boxes. It can be used to generate masks for all objects in an image, trained on a dataset of 11 million images and 1.1 billion masks. SAM has strong zero-shot performance on various segmentation tasks.

> Paper: Segment Anything</br>
> Affiliation: Meta AI Research, FAIR</br>
> Published: ICCV23</br>

Refer to these [steps](https://github.com/vietanhdev/samexporter#sam-exporter).

- [Efficient-SAM](https://github.com/CVHub520/efficientvit)

`EfficientViT` is a new family of vision models for efficient high-resolution dense prediction. It uses a new lightweight multi-scale linear attention module as the core building block. This module achieves a global receptive field and multi-scale learning with only hardware-efficient operations.

> Paper: EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction</br>
> Affiliation: MIT</br>
> Published: ICCV23</br>

Refer to these [steps](https://github.com/CVHub520/efficientvit#benchmarking-with-onnxruntime).

- [SAM-Med2D](https://github.com/CVHub520/SAM-Med2D)

`SAM-Med2D` is a specialized model developed to address the challenge of applying state-of-the-art image segmentation techniques to medical images.

> Paper: SAM-Med2D</br>
> Affiliation: OpenGVLab</br>
> Published: Arxiv23</br>

Refer to these [steps](https://github.com/CVHub520/SAM-Med2D#-deploy).

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 

`GroundingDINO` is a state-of-the-art (SOTA) zero-shot object detection model excelling in detecting objects beyond the predefined training classes. Its unique capability allows adaptation to new objects and scenarios, making it highly versatile for real-world applications. It also performs well in Referring Expression Comprehension (REC), identifying and localizing specific objects or regions within an image based on textual descriptions. Grounding DINO simplifies object detection by eliminating hand-designed components like Non-Maximum Suppression (NMS), streamlining the model architecture, and enhancing efficiency and performance.

> Paper: Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection</br>
> Affiliation: IDEA-CVR, IDEA-Research</br>
> Published: Arxiv23</br>

Refer to this [tutorial](../../tools/export_grounding_dino_onnx.py).

- [Recognize Anything](https://github.com/xinyu1205/Tag2Text) 

`RAM` is a robust image tagging model known for its exceptional capabilities in image recognition. RAM stands out for its strong and versatile performance, excelling in zero-shot generalization. It offers the advantages of being both cost-effective and reproducible, relying on open-source and annotation-free datasets. RAM's flexibility makes it suitable for a wide range of application scenarios, making it a valuable tool for various image recognition tasks.

> Paper: Recognize Anything: A Strong Image Tagging Model</br>
> Affiliation: OPPO Research Institute, IDEA-Research, AI Robotics</br>
> Published: Arxiv23</br>

Refer to this [tutorial](../../tools/export_recognize_anything_model_onnx.py).

- [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_person_attribute.md)

This tutorial provides a way for users to quickly build a lightweight, high-precision, and practical classification model of person attributes using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in pedestrian analysis scenarios, pedestrian tracking scenarios, etc.

Refer to this [tutorial](../../tools/export_pulc_attribute_model_onnx.py).

- [VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md)

This tutorial provides a way for users to quickly build a lightweight, high-precision, and practical classification model of vehicle attributes using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in vehicle identification, road monitoring, and other scenarios.

Refer to this [tutorial](../../tools/export_pulc_attribute_model_onnx.py).

- [HQ-SAM](https://github.com/SysCV/sam-hq)

The recent Segment Anything Model (SAM) represents a big leap in scaling up segmentation models, allowing for powerful zero-shot capabilities and flexible prompting. Despite being trained with 1.1 billion masks, SAM's mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures. We propose HQ-SAM, equipping SAM with the ability to accurately segment any object, while maintaining SAM's original promptable design, efficiency, and zero-shot generalizability. Our careful design reuses and preserves the pre-trained model weights of SAM, while only introducing minimal additional parameters and computation. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with early and final ViT features for improved mask details. To train our introduced learnable parameters, we compose a dataset of 44K fine-grained masks from several sources. HQ-SAM is only trained on the introduced dataset of 44k masks, which takes only 4 hours on 8 GPUs. We show the efficacy of HQ-SAM in a suite of 9 diverse segmentation datasets across different downstream tasks, where 7 out of them are evaluated in a zero-shot transfer protocol.

> Paper: Segment Anything in High Quality</br>
> Affiliation: ETH Zurich & HKUST</br>
> Published: NeurIPS 2023</br>

Refer to this [tutorial](https://github.com/CVHub520/sam-hq).

- [InternImage](https://github.com/OpenGVLab/InternImage)

InternImage introduces a large-scale convolutional neural network (CNN) model, leveraging deformable convolution as the core operator to achieve a large effective receptive field, adaptive spatial aggregation, and reduced inductive bias, leading to stronger and more robust pattern learning from massive data. It outperforms current CNNs and vision transformers on benchmarks

> Paper: InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions</br>
> Affiliation: Shanghai AI Laboratory, Tsinghua University, Nanjing University, etc.</br>
> Published: CVPR 2023</br>

Refer to this [tutorial](../../tools/export_internimage_model_onnx.py).

- [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)

`EdgeSAM` is an accelerated variant of the Segment Anything Model (SAM), optimized for efficient execution on edge devices with minimal compromise in performance. It achieves a 40-fold speed increase compared to the original SAM, and outperforms MobileSAM, being 14 times as fast when deployed on edge devices while enhancing the mIoUs on COCO and LVIS by 2.3 and 3.2 respectively. EdgeSAM is also the first SAM variant that can run at over 30 FPS on an iPhone 14.

> Paper: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM</br>
> Affiliation: S-Lab, Nanyang Technological University, Shanghai Artificial Intelligence Laboratory.</br>
> Published: Arxiv 2023</br>

Refer to this [tutorial](https://github.com/chongzhou96/EdgeSAM/blob/master/scripts/export_onnx_model.py).

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

`YOLO-World` enhances the YOLO series by incorporating vision-language modeling, achieving efficient open-scenario object detection with impressive performance on various tasks.

> Paper: Real-Time Open-Vocabulary Object Detection</br>
> Affiliation: Tencent AI Lab, ARC Lab, Tencent PCG, Huazhong University of Science and Technology.</br>
> Published: Arxiv 2024</br>

Refer to this [tutorial](../../tools/export_yolow_onnx.py).
