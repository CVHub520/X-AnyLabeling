## 一、如何加载已有模型

对由于网络问题未能够从界面下载权重文件至本地的小伙伴，首先请先从 [models_list.md](./models_list.md) 找到并下载预加载模型对应的权重文件（没有科学上网的请自觉走百度网盘链接），随后按照 [#23](https://github.com/CVHub520/X-AnyLabeling/issues/23) 提供的步骤一步步操作即可。


## 二、如何加载自定义模型？

> 本文档将为大家详细阐述，如何在 `X-AnyLabeling` 框架上加载自定义的模型。

### 2.1 已适配模型

此处所述的**已适配模型**大家可以参考下 [models_list.md](./models_list.md)，如果是框架中已经支持的网络结构，那么可以按照以下步骤进行适配，这里以 [yolov5s](https://github.com/ultralytics/yolov5) 模型为例，其它类似。

首先，在正式开始之前，我们可以看下该模型对应的[配置文件](../anylabeling/configs/auto_labeling/yolov5s.yaml)：

```YAML
type: yolov5
name: yolov5s-r20230520
display_name: YOLOv5s Ultralytics
model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov5s.onnx
input_width: 640
input_height: 640
stride: 32
nms_threshold: 0.45
confidence_threshold: 0.45
classes:
  - person
  - bicycle
  - car
  ...
```

这里详细解释下每个字段：

- `type`: 网络类型定义，不可更改，目前已适配的网络类型定义可参见 [model_manager.py](../anylabeling/services/auto_labeling/model_manager.py) 文件中的 `load_custom_model()` 函数；
- `name`: 该字段为当前模型对应的配置文件索引标记，如果是加载用户自定义模型，此字段可忽略，详情可参见 [models.yaml](../anylabeling/configs/auto_labeling/models.yaml) 文件；
- `display_name`: 即展示到界面上显示的名称，可根据自定义任务自行命名，如 `Fruits (YOLOv5s)`;
- `model_path`: 即相对于自定义配置文件 `*.yaml` 所对应的模型权重路径，要求是 `*.onnx` 文件格式；

注：剩余的均为当前模型所依赖的相关超参数设置，可根据任务自行设置，具体的实现可参考 [yolov5s.py](../anylabeling/services/auto_labeling/yolov5.py) 文件。

好了，了解完前置知识后，假设现在我们手头上训练了一个可检测 `apple`、`banana` 以及 `orange` 三类别的 `yolov5s` 检测模型，我们需要先将 `*.pt` 文件转换为 `*.onnx` 文件，具体的转换方法可参考每个框架给出的转换指令，如 `yolov5` 官方提供的 [Tutorial](https://docs.ultralytics.com/yolov5/tutorials/model_export) 文档。

其次，得到 `onnx` 权重文件（假设命名为 `fruits.onnx`）之后，我们可以复制一份 `X-AnyLabeling` 中提供的对应模型的配置文件，如上述提到的 [yolov5s.yaml](../anylabeling/configs/auto_labeling/yolov5s.yaml)，随后根据自己需要修改下对应的超参数字段，如检测阈值，类别名称等，示例如下：

```YAML
type: yolov5
name: yolov5s-r20230520
display_name: Fruits (YOLOv5s)
model_path: fruits.onnx
input_width: 640
input_height: 640
stride: 32
nms_threshold: 0.45
confidence_threshold: 0.45
classes:
  - apple
  - banana
  - orange
```

可以看出，这里 `model_path` 字段建议直接填写模型权重名称，随后我们只需在任意新建一个文件夹，将上述权重文件和对应的配置文件放置到同一个文件夹下存放即可，组织目录如下：

```
|- custom_model
|   |- fruits.onnx
|   |- yolov5s.yaml
```

最后，我们打开 `GUI` 界面，点击 `模型图标` 按钮，选择 `...加载自定义模型`（中文版） 或者 `...Load Custom Model`（英文版），然后选择 `yolov5s.yaml` 配置文件即可完成自定义模型加载。此外，我们同样可以设置 `agnostic`, `filter_classes` 参数。如果你是使用 `yolov5-5.0` 版本的，可以加入 `anchors` 参数，例如：

```YAML
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

通过上面的简单教程大家也可以看出，其实这与整个 `yolo` 框架是无缝对接的。

> 注：如果按照上述教程加载后提示报错，请参考[帮助文档](./Q&A.md)中的**问题反馈**章节。

### 2.2 未适配模型

对于目前 [models_list.md](./models_list.md) 中未适配过的模型，我们也可以参考上述资料，并按照以下快速完成适配工作：

- [X-AnyLabeling/anylabeling/configs/auto_labeling](../anylabeling/configs/auto_labeling): 定义配置文件；
- [X-AnyLabeling/anylabeling/configs/auto_labeling/models.yaml](../anylabeling/configs/auto_labeling/models.yaml): 添加配置文件；
- [X-AnyLabeling/anylabeling/services/auto_labeling](../anylabeling/services/auto_labeling)：模型推理逻辑实现；
- [X-AnyLabeling/anylabeling/services/auto_labeling/model_manager.py](../anylabeling/services/auto_labeling/model_manager.py)：添加到模型管理。

> 注：这部分需要少量的代码编程功底，如果遇到困难可直接在 [issue](https://github.com/CVHub520/X-AnyLabeling/issues) 区提交，尽量补充详细的上下文信息。

## 三、其它模型部署教程

> 此章节仅提供从 `pt` 转 `onnx` 格式的步骤，剩余操作请参考上述教程。

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

参考此[教程](../tools/export_grounding_dino_onnx.py).

- [Recognize Anything](https://github.com/xinyu1205/Tag2Text) 

`RAM` is a robust image tagging model known for its exceptional capabilities in image recognition. RAM stands out for its strong and versatile performance, excelling in zero-shot generalization. It offers the advantages of being both cost-effective and reproducible, thanks to its reliance on open-source and annotation-free datasets. RAM's flexibility makes it suitable for a wide range of application scenarios, making it a valuable tool for various image recognition tasks.

> 论文：Recognize Anything: A Strong Image Tagging Model</br>
> 单位：OPPO Research Institute, IDEA-Research, AI Robotics</br>
> 发表：Arxiv23</br>

参考此[教程](../tools/export_recognize_anything_model_onnx.py).

- [PersonAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_person_attribute.md)

This case provides a way for users to quickly build a lightweight, high-precision and practical classification model of person attribute using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in Pedestrian analysis scenarios, pedestrian tracking scenarios, etc.

参考此[教程](../tools/export_pulc_attribute_model_onnx.py).

- [VehicleAttribute](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md)

This case provides a way for users to quickly build a lightweight, high-precision and practical classification model of vehicle attribute using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in Vehicle identification, road monitoring and other scenarios.

参考此[教程](../tools/export_pulc_attribute_model_onnx.py).

- [HQ-SAM](https://github.com/SysCV/sam-hq)

The recent Segment Anything Model (SAM) represents a big leap in scaling up segmentation models, allowing for powerful zero-shot capabilities and flexible prompting. Despite being trained with 1.1 billion masks, SAM's mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures. We propose HQ-SAM, equipping SAM with the ability to accurately segment any object, while maintaining SAM's original promptable design, efficiency, and zero-shot generalizability. Our careful design reuses and preserves the pre-trained model weights of SAM, while only introducing minimal additional parameters and computation. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with early and final ViT features for improved mask details. To train our introduced learnable parameters, we compose a dataset of 44K fine-grained masks from several sources. HQ-SAM is only trained on the introduced detaset of 44k masks, which takes only 4 hours on 8 GPUs. We show the efficacy of HQ-SAM in a suite of 9 diverse segmentation datasets across different downstream tasks, where 7 out of them are evaluated in a zero-shot transfer protocol.

> 论文：Segment Anything in High Quality</br>
> 单位：ETH Zurich & HKUST</br>
> 发表：NeurIPS 2023</br>

参考此[教程](https://github.com/CVHub520/sam-hq).
