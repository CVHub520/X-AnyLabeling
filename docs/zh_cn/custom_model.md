## 一、如何加载内置模型

目前 `X-AnyLabeling` 内置模型默认是托管到 github 的 release 仓库，因此想要从 GUI 界面直接导入模型的同学需要自行配置科学上网条件，并保持当前网络畅通，否则大概会遇到下载失败的情况。

对由于网络问题未能够从界面顺利下载权重文件至本地的小伙伴，请先从 [model_zoo.md](./model_zoo.md) 中找到您期望加载模型对应的权重文件（没配置科学上网的请自觉走百度网盘链接），最后按照 [#23](https://github.com/CVHub520/X-AnyLabeling/issues/23) 提供的演示步骤一步步操作即可。


## 二、如何加载自定义模型？

### 2.1 已适配模型

**已适配模型**指的是 X-AnyLabeling 中已经适配过的网络模型，同样地可参考 [model_zoo.md](./model_zoo.md) 文档。以 [yolov5s](https://github.com/ultralytics/yolov5) 模型为例，现假设用户在本地自己训练了一个 `yolov5` 检测模型，大家可以把相应的 [配置文件](../../anylabeling/configs/auto_labeling/yolov5s.yaml) 下载下来，如下所示：

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

简单解释下每个字段：

- `type`: 用于定义网络类型的标识符，以唯一标识每个模型。该标识符用户不可更改。
- `name`: 用于定义当前内置模型对应的配置文件索引标记，如果是加载用户自定义模型，此字段可不用设置。更多详情可参见 [models.yaml](../../anylabeling/configs/auto_labeling/models.yaml) 文件。
- `display_name`: 用于在界面上展示的名称，可根据自定义任务进行命名，例如 `Fruits (YOLOv5s)`。
`model_path`: 用于指定加载模型权重的路径。请注意，该路径是相对于当前配置文件的相对路径。如果需要，也可以直接填写绝对路径。同时，确保文件格式为 `*.onnx`。
- `nms_threshold`、`confidence_threshold`、`classes`字段可根据实际情况自行设置。此外，`X-AnyLabeling` 还为用户提供了一些辅助功能，例如支持 `agnostic` 和 `filter_classes` 功能，示例如下：

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

通过 `filter_classes` 的设置我们便可以让模型仅检测 `person` 这个类别，而 `agnostic` 参数则用于将所有目标都视为一个类别做一次 NMS。需要注意的是，目前内置模型支持的 `yolov5` 版本为 v6.0+，如果你是 `v5.0` 等旧版本，务必在配置文件中指定 `anchors` 和 `stride` 字段，示例如下：

```YAML
type: yolov5
...
stride: 32
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

至此，我们便了解了整个 `X-AnyLabeling` 中配置文件的作用。假如用户手头上训练了一个检测 `apple`、`banana` 以及 `orange` 三类别的 `yolov5s` 检测模型，首先需要先将 `*.pt` 文件转换为 `*.onnx` 格式，得到一个 `fruits.onnx` 的模型权重。

> 请注意，转换后的模型务必使用[netron](https://netron.app/)工具打开，确保输入输出节点与 X-AnyLabeling 内置模型的输入输出节点一致。

其次，我们可以在 [model_zoo.md](./model_zoo.md) 中将当前模型对应的配置文件赋值一份至本地，并根据需要修改对应的超参数字段，如检测阈值和类别等，示例如下：

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

接下来，新建一个文件夹，将权重文件和对应的配置文件放置到同一个目录下：

```
|- custom_model
|   |- fruits.onnx
|   |- fruits_yolov5s.yaml
```

最后，打开 `GUI` 界面，点击 `模型图标` 按钮，选择 `...加载自定义模型`（中文版） 或者 `...Load Custom Model`（英文版），然后选择 `fruits_yolov5s.yaml` 配置文件即可完成自定义模型加载。


### 2.2 未适配模型

**未适配模型**指的是非 `X-AnyLabeling` 工具内置的模型，可参考以下步骤快速集成到框架中：

- 定义配置文件

以 `yolov5` 为例，具体可参考此[配置文件](../../anylabeling/configs/auto_labeling/yolov5s.yaml)。

- 添加至任务管理

具体可参考此[文件](../../anylabeling/configs/auto_labeling/models.yaml)，按照统一范式进行添加。

- 定义模型文件

这一步核心是继承 [models](../../anylabeling/services/auto_labeling/model.py) 类，实现相应地预处理、模型推理和后处理逻辑，具体可以找个示例查看。

- 添加至模型管理

在[模型管理文件](../../anylabeling/services/auto_labeling/model_manager.py)中新增上述自定义的模型类。


## 三、如何导出模型

> 本小节主要提供将 PyTorch (`pt`) 权重文件转换为 ONNX (`onnx`) 的教程，旨在协助用户快速完成此转换过程。

- [YOLOv5_OBB](https://github.com/hukaixuan19970627/yolov5_obb)

> 作者：胡凯旋

参考此[教程](https://github.com/CVHub520/yolov5_obb/tree/main).


- [YOLOv7](https://github.com/WongKinYiu/yolov7)

> 论文：YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors</br>
> 单位：Institute of Information Science, Academia Sinica, Taiwan

```bash
python export.py --weights yolov7.pt --img-size 640 --grid
```

- [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)

> 论文：Efficient object detectors including Gold-YOLO</br>
> 单位：huawei-noah</br>
> 发表：NeurIPS23</br>

```bash
git clone https://github.com/huawei-noah/Efficient-Computing.git
cd Detection/Gold-YOLO
python deploy/ONNX/export_onnx.py --weights Gold_n_dist.pt --simplify --ort
                                            Gold_s_pre_dist.pt                     
                                            Gold_m_pre_dist.pt
                                            Gold_l_pre_dist.pt
```

- [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)

`DAMO-YOLO` is a fast and accurate object detection method, which is developed by TinyML Team from Alibaba DAMO Data Analytics and Intelligence Lab. And it achieves a higher performance than state-of-the-art YOLO series. DAMO-YOLO is extend from YOLO but with some new techs, including Neural Architecture Search (NAS) backbones, efficient Reparameterized Generalized-FPN (RepGFPN), a lightweight head with AlignedOTA label assignment, and distillation enhancement. For more details, please refer to our Arxiv Report. Moreover, here you can find not only powerful models, but also highly efficient training strategies and complete tools from training to deployment.

> 论文：DAMO-YOLO: A Report on Real-Time Object Detection Design</br>
> 单位：Alibaba Group</br>
> 发表：Arxiv22</br>

```bash
git clone https://github.com/tinyvision/DAMO-YOLO.git
cd DAMO-YOLO
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640
```

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

Real-Time DEtection TRansformer (`RT-DETR`, aka RTDETR), the first real-time end-to-end object detector to our best knowledge. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS.

> 论文：RT-DETR: DETRs Beat YOLOs on Real-time Object Detection</br>
> 单位：Baidu</br>
> 发表：Arxiv22</br>

参考此[文章](https://zhuanlan.zhihu.com/p/628660998).

- [SAM](https://github.com/vietanhdev/samexporter)

The Segment Anything Model (`SAM`) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

> 论文：Segment Anything</br>
> 单位：Meta AI Research, FAIR</br>
> 发表：ICCV23</br>

参考此[步骤](https://github.com/vietanhdev/samexporter#sam-exporter).

- [Efficient-SAM](https://github.com/CVHub520/efficientvit)

`EfficientViT` is a new family of vision models for efficient high-resolution dense prediction. The core building block of EfficientViT is a new lightweight multi-scale linear attention module that achieves global receptive field and multi-scale learning with only hardware-efficient operations.

> 论文：EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction</br>
> 单位：MIT</br>
> 发表：ICCV23</br>

参考此[步骤](https://github.com/CVHub520/efficientvit#benchmarking-with-onnxruntime).

- [SAM-Med2D](https://github.com/CVHub520/SAM-Med2D)

`SAM-Med2D` is a specialized model developed to address the challenge of applying state-of-the-art image segmentation techniques to medical images.

> 论文：SAM-Med2D</br>
> 单位：OpenGVLab</br>
> 发表：Arxiv23</br>

参考此[步骤](https://github.com/CVHub520/SAM-Med2D#-deploy).

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 

`GroundingDINO` is a state-of-the-art (SOTA) zero-shot object detection model that excels in detecting objects beyond the predefined training classes. This unique capability allows the model to adapt to new objects and scenarios, making it highly versatile for real-world applications. It also excels in Referring Expression Comprehension (REC), where it can identify and localize specific objects or regions within an image based on textual descriptions. What sets it apart is its deep understanding of language and visual content, enabling it to associate words or phrases with corresponding visual elements. Moreover, Grounding DINO simplifies object detection by eliminating hand-designed components like Non-Maximum Suppression (NMS), streamlining the model architecture, and enhancing efficiency and performance.

> 论文：Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection</br>
> 单位：IDEA-CVR, IDEA-Research</br>
> 发表：Arxiv23</br>

参考此[教程](../../tools/export_grounding_dino_onnx.py).

- [Recognize Anything](https://github.com/xinyu1205/Tag2Text) 

`RAM` is a robust image tagging model known for its exceptional capabilities in image recognition. RAM stands out for its strong and versatile performance, excelling in zero-shot generalization. It offers the advantages of being both cost-effective and reproducible, thanks to its reliance on open-source and annotation-free datasets. RAM's flexibility makes it suitable for a wide range of application scenarios, making it a valuable tool for various image recognition tasks.

> 论文：Recognize Anything: A Strong Image Tagging Model</br>
> 单位：OPPO Research Institute, IDEA-Research, AI Robotics</br>
> 发表：Arxiv23</br>

参考此[教程](../../tools/export_recognize_anything_model_onnx.py).

- [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_person_attribute.md)

This case provides a way for users to quickly build a lightweight, high-precision and practical classification model of person attribute using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in Pedestrian analysis scenarios, pedestrian tracking scenarios, etc.

参考此[教程](../../tools/export_pulc_attribute_model_onnx.py).

- [VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md)

This case provides a way for users to quickly build a lightweight, high-precision and practical classification model of vehicle attribute using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in Vehicle identification, road monitoring and other scenarios.

参考此[教程](../../tools/export_pulc_attribute_model_onnx.py).

- [HQ-SAM](https://github.com/SysCV/sam-hq)

The recent Segment Anything Model (SAM) represents a big leap in scaling up segmentation models, allowing for powerful zero-shot capabilities and flexible prompting. Despite being trained with 1.1 billion masks, SAM's mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures. We propose HQ-SAM, equipping SAM with the ability to accurately segment any object, while maintaining SAM's original promptable design, efficiency, and zero-shot generalizability. Our careful design reuses and preserves the pre-trained model weights of SAM, while only introducing minimal additional parameters and computation. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with early and final ViT features for improved mask details. To train our introduced learnable parameters, we compose a dataset of 44K fine-grained masks from several sources. HQ-SAM is only trained on the introduced detaset of 44k masks, which takes only 4 hours on 8 GPUs. We show the efficacy of HQ-SAM in a suite of 9 diverse segmentation datasets across different downstream tasks, where 7 out of them are evaluated in a zero-shot transfer protocol.

> 论文：Segment Anything in High Quality</br>
> 单位：ETH Zurich & HKUST</br>
> 发表：NeurIPS 2023</br>

参考此[教程](https://github.com/CVHub520/sam-hq).

- [InternImage](https://github.com/OpenGVLab/InternImage)

InternImage introduces a large-scale convolutional neural network (CNN) model, leveraging deformable convolution as the core operator to achieve a large effective receptive field, adaptive spatial aggregation, and reduced inductive bias, leading to stronger and more robust pattern learning from massive data, outperforming current CNNs and vision transformers on benchmarks like COCO and ADE20K.

> 论文：InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions</br>
> 单位：Shanghai AI Laboratory, Tsinghua University, Nanjing University, etc.</br>
> 发表：CVPR 2023</br>

参考此[教程](../../tools/export_internimage_model_onnx.py).

- [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)

`EdgeSAM` is an accelerated variant of the Segment Anything Model (SAM), optimized for efficient execution on edge devices with minimal compromise in performance. It achieves a 40-fold speed increase compared to the original SAM, and outperforms MobileSAM, being 14 times as fast when deployed on edge devices while enhancing the mIoUs on COCO and LVIS by 2.3 and 3.2 respectively. EdgeSAM is also the first SAM variant that can run at over 30 FPS on an iPhone 14.

> 论文：Prompt-In-the-Loop Distillation for On-Device Deployment of SAM</br>
> 单位：S-Lab, Nanyang Technological University, Shanghai Artificial Intelligence Laboratory.</br>
> 发表：Arxiv 2023</br>

参考此[教程](https://github.com/chongzhou96/EdgeSAM/blob/master/scripts/export_onnx_model.py).

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

`YOLO-World` enhances the YOLO series by incorporating vision-language modeling, achieving efficient open-scenario object detection with impressive performance on various tasks.

> 论文：Real-Time Open-Vocabulary Object Detection</br>
> 单位：Tencent AI Lab, ARC Lab, Tencent PCG, Huazhong University of Science and Technology.</br>
> 发表：Arxiv 2024</br>

参考此[教程](../../tools/export_yolow_onnx.py).
