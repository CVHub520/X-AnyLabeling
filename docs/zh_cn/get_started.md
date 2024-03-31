## 快速入门指南

### 运行模式

目前 `X-AnyLabeling` 支持两种运行方式，一种是下载源码直接运行，另一种是直接下载编译好的 `GUI` 版本运行。需要注意的时，为了保证用户能使用到最新的功能特性和最稳定的性能体验，强烈建议从源码运行。

### 源码运行

1. 下载源码

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
```

2. 安装依赖

目前，`X-AnyLabeling` 针对不同的运行环境提供了多份依赖文件：

| 依赖文件                | 系统环境           | 运行环境 | 是否支持打包 |
|----------------------|------------------|--------|-----|
| requirements.txt     | Windows/Linux    | CPU    | 否   |
| requirements-dev.txt | Windows/Linux    | CPU    | 是   |
| requirements-gpu.txt | Windows/Linux    | GPU    | 否   |
| requirements-gpu-dev.txt | Windows/Linux | GPU    | 是   |
| requirements-macos.txt | MacOS           | CPU    | 否   |
| requirements-macos-dev.txt | MacOS       | CPU    | 是   |

由于当前工具内置的模型推理后端为 `OnnxRuntime`，因此，如果您希望利用 GPU 进行模型推理加速，请务必确保本地 CUDA 版本与 `onnxruntime-gpu` 版本兼容，以确保顺利调用显卡。有关详细信息，请参考[官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。同时，请务必将[app_info.py](../../anylabeling/app_info.py)配置文件中的`__preferred_device__`字段设置为`GPU`。

3. 启动工具

> 设置当前工作环境变量可参考以下步骤：</br>
> - Linux/MasOS
>   - export PYTHONPATH=/path/to/X-AnyLabeling
> - Windows
>   - set PYTHONPATH=C:\path\to\X-AnyLabeling

在 `X-AnyLabeling` 工程目录下执行以下命令进行启动：

```bash
python anylabeling/app.py
```

**可选参数:**

* `filename`: 图像或标签文件名；如果传入目录路径，则会自动加载该文件夹

* `--help`,`-h`: 显示帮助消息并退出

- `--reset-config`: 重置 Qt 配置，清除所有设置。
- `--logger-level`: 设置日志级别，可选值包括 "debug", "info", "warning", "fatal", "error"。
- `--output`, `-O`, `-o`: 指定输出文件或目录。如果以 `.json` 结尾，则被识别为文件，否则被识别为目录。
- `--config`: 指定配置文件或者以 YAML 格式提供配置信息的字符串
  默认为 `~/.anylabelingrc`(Linux)      `C:\Users\{user}\.anylabelingrc`(Windows)。
- `--nodata`: 停止将图像数据存储到 JSON 文件中。
- `--autosave`: 自动保存标注数据。
- `--nosortlabels`: 停止对标签进行排序。
- `--flags`: 逗号分隔的标志列表或包含标志的文件。
- `--labelflags`: 包含标签特定标志的 YAML 字符串或包含 JSON 字符串的文件。
- `--labels`: 逗号分隔的标签列表或包含标签的文件。
- `--validatelabel`: 标签验证类型。
- `--keep-prev`: 保留前一帧的注释。
- `--epsilon`: 在画布上找到最近顶点的 epsilon。

### GUI 环境运行

在使用 `X-AnyLabeling` 自身提供的 `GUI` 环境运行时，相较于源码运行，最大的优势在于其方便快捷，用户无需深入关注底层实现细节，只需下载完成即可立即使用，省去了繁琐的环境配置和依赖安装步骤。然而，这种便捷方式也存在一些明显的弊端，主要包括：

1. **不易排查问题：** 当出现闪退或报错问题时，由于用户无法直接查看源码，难以快速定位具体原因，使问题排查变得相对困难。

2. **GPU加速限制：** 对于希望通过调用GPU进行加速推理的用户，存在较大限制。当前提供的编译版本基于CUDA 11.6和onnxruntime 1.16.0版本进行打包编译，可能无法满足某些用户对于最新硬件或库版本的需求。

3. **功能特性滞后：** 由于无法及时更新编译版本，GUI环境运行可能无法享受到最新的功能特性，并且一些潜在的bug可能未能及时修复，影响了用户的整体体验。

为了在选择运行方式时能够更好地权衡利弊，建议用户根据具体需求和偏好，灵活选择源码运行或GUI环境运行，以达到最佳的使用体验。

下载链接：[Release](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v5) | [百度网盘](https://pan.baidu.com/s/1n3vfXZpBUG9s12NZjMxhlw?pwd=odyi)


### 文件导入

`X-AnyLabeling` 目前提供了三种便捷的导入方式，如下所示：

| 导入方式  | 快捷键    |
|----------|-----------|
| 图像文件  | Ctrl+I    |
| 图像目录  | Ctrl+U    |
| 视频文件  | Ctrl+O    |

需要注意的是，默认的标注文件保存路径为导入文件路径，如果需要存放到其它目录，可点击左上角 `文件` -> `另存为`，选择保存目录即可。

### 快速绘制

当前 `X-AnyLabeling` 中支持**多边形**、**矩形框**、**旋转框**、**圆形**、**线段**、**多线段**和**点**等多种标注样式，可供用户灵活地选取。部分绘制模式的快捷键设置如下：

| 标注样式  | 快捷键    | 应用场景 |
|----------|-----------|-----------|
| 多边形  |  P   | 图像分割 |
| 矩形框  |  R   | 水平目标检测 |
| 旋转框  |  O   | 旋转目标检测 |
| 圆形 | - | 特定场景 |
| 线段 | - | 车道线检测 |
| 多线段 | - | 血管分割 |
| 点 | - | 关键点检测 |

`X-AnyLabeling` 交互模式目前主要有两种：

- 编辑模式：此状态下用户可移动、复制、黏贴、修改对象等；
- 绘制模式：此状态下仅支持绘制相应地标注样式；

目前在 **矩形框**、**旋转框**、**圆形**、**线段**、**点**五种标注样式下，当图案绘制完成后，会自动切换到编辑模式。对于其它两种样式，用户可通过快捷键 `Ctrl+J` 完成快速切换。

### 辅助推理

对于想要使用 `X-AnyLabeling` 工具提供的 AI 算法功能库，可点击左侧菜单栏带 `AI` 字样的图标或直接按下快捷键 `Ctrl+A` 调出模型列表，点击下拉框选择自己需要的模型即可。如遇下载失败情况，请参考[custom_model.md](./custom_model.md)文档。

### 一键运行

`X-AnyLabeling` 工具中提供了实用的 `一键运行` 功能给予用户快速完成对当前批次任务的标注工作，用户可直接点击左侧菜单栏带 `播放` 图案的图标或直接按下快捷键 `Ctrl+M` 唤醒该功能，自动完成从当前图片到最后一张图片的标注。

> 需要注意的是，此项功能需要在给定模型被激活的状态下使用。此外一经开启便需要跑完整个任务，因此在启动之前笔者强烈建议先在小批量图片上进行测试，确保无误后再调用此功能。

### 打包编译

> 请注意，以下步骤是非必要的，本小节内容仅为可能需要自定义和编译软件以在特定环境中分发的用户提供的。如果您只是单纯使用本软件，请跳过这一步骤。

<details>
<summary>展开/折叠</summary>

为了方便用户在不同平台上运行 `X-AnyLabeling`，工具提供了打包编译的指令和相关注意事项。在执行以下打包指令之前，请根据您的环境和需求，修改 [app_info.py](../../anylabeling/app_info.py) 文件中的 `__preferred_device__` 参数，以选择相应的 GPU 或 CPU 版本进行构建。

注意事项：

1. 在编译前，请确保已经根据所需的 GPU/CPU 版本修改了 `anylabeling/app_info.py` 文件中的 `__preferred_device__` 参数。

2. 如果需要编译 GPU 版本，请先激活相应地 `GPU` 运行环境，执行 `pip install | grep onnxruntime-gpu` 确保被正确安装。

3. 对于 Windows-GPU 版本的编译，需要手动修改 `anylabeling-win-gpu.spec` 文件中的 `datas` 列表参数，将本地的 `onnxruntime-gpu` 相关动态库 `*.dll` 添加进列表中。

4. 对于 Linux-GPU 版本的编译，需要手动修改 `anylabeling-linux-gpu.spec` 文件中的 `datas` 列表参数，将本地的 `onnxruntime-gpu` 相关动态库 `*.so` 添加进列表中。此外，请注意根据您的 CUDA 版本下载匹配的 `onnxruntime-gpu` 包，详细匹配表可参考[官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

参考指令：

```bash
# Windows-CPU
bash scripts/build_executable.sh win-cpu

# Windows-GPU
bash scripts/build_executable.sh win-gpu

# Linux-CPU
bash scripts/build_executable.sh linux-cpu

# Linux-GPU
bash scripts/build_executable.sh linux-gpu
```

注：如果您在 Windows 环境下执行以上指令出现权限问题的话，可在确保上述准备工作完成之后，直接根据需要执行以下指令：

> pyinstaller --noconfirm anylabeling-win-cpu.spec</br>
> pyinstaller --noconfirm anylabeling-win-gpu.spec

</details>



