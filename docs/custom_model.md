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
- `name`: 该字段为当前模型对应的配置文件索引标记，同样不可更改，详情可参见 [models.yaml](../anylabeling/configs/auto_labeling/models.yaml) 文件；
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

最后，我们打开 `GUI` 界面，点击 `模型图标` 按钮，选择 `...加载自定义模型`（中文版） 或者 `...Load Custom Model`（英文版），然后选择 `yolov5s.yaml` 配置文件即可完成自定义模型加载。

> 注：如果按照上述教程加载后提示报错，请参考[帮助文档](./Q&A.md)中的**问题反馈**章节。

### 2.2 未适配模型

对于目前 [models_list.md](./models_list.md) 中未适配过的模型，我们也可以参考上述资料，并按照以下快速完成适配工作：

- [X-AnyLabeling/anylabeling/configs/auto_labeling](../anylabeling/configs/auto_labeling): 定义配置文件；
- [X-AnyLabeling/anylabeling/configs/auto_labeling/models.yaml](../anylabeling/configs/auto_labeling/models.yaml): 添加配置文件；
- [X-AnyLabeling/anylabeling/services/auto_labeling](../anylabeling/services/auto_labeling)：模型推理逻辑实现；
- [X-AnyLabeling/anylabeling/services/auto_labeling/model_manager.py](../anylabeling/services/auto_labeling/model_manager.py)：添加到模型管理。

> 注：这部分需要少量的代码编程功底，如果遇到困难可直接在 [issue](https://github.com/CVHub520/X-AnyLabeling/issues) 区提交，尽量补充详细的上下文信息。