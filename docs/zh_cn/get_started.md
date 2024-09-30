# 快速入门指南

## 1. 准备工作

### 1.1 从源码运行

#### 1.1.1 前置条件

> [!NOTE]
> 如果你需要使用 Segment-Anything-2 的视频追踪功能, 请先移步至此[文档](../../examples/interactive_video_object_segmentation/README.md)安装相关依赖。

在开始之前，请确保您已安装以下前置条件：

**步骤 0.** 从[官方网站](https://docs.anaconda.com/miniconda/)下载并安装 Miniconda。

**步骤 1.** 创建一个 Python 版本为 3.8 或更高版本的 conda 环境，并激活它。

```bash
conda create --name x-anylabeling python=3.9 -y
conda activate x-anylabeling
```

#### 1.1.2 安装

**步骤 0.** 安装 [ONNX Runtime](https://onnxruntime.ai/)。

```bash
# Install ONNX Runtime CPU
pip install onnxruntime

# Install ONNX Runtime GPU (CUDA 11.x)
pip install onnxruntime-gpu==x.x.x

# Install ONNX Runtime GPU (CUDA 12.x)
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

> [!Important]
> 对于 GPU 加速，请按照以下说明，确保您本地的 CUDA 和 cuDNN 版本与 ONNX Runtime 版本兼容，并安装需要依赖库，以确保 GPU 加速推理正常：</br>
> Ⅰ. [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)</br>
> Ⅱ. [Get started with ONNX Runtime in Python](https://onnxruntime.ai/docs/get-started/with-python.html)</br>
> Ⅲ. ONNX Runtime 版本需大于等于 1.16.0.

**步骤 1.** 克隆代码仓库。

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
```

**步骤 2:** 安装 `requirements.txt` 文件。

对于不同的配置，X-AnyLabeling 提供了以下依赖文件：

| 依赖文件                   | 操作系统        | 运行环境 | 可编译   |
|----------------------------|-----------------|----------|----------|
| requirements.txt           | Windows/Linux   | CPU      | 否       |
| requirements-dev.txt       | Windows/Linux   | CPU      | 是       |
| requirements-gpu.txt       | Windows/Linux   | GPU      | 否       |
| requirements-gpu-dev.txt   | Windows/Linux   | GPU      | 是       |
| requirements-macos.txt     | MacOS           | CPU      | 否       |
| requirements-macos-dev.txt | MacOS           | CPU      | 是       |

- 对于开发者，您应选择带有 `*-dev.txt` 后缀的选项进行安装。
- 如需启用 GPU 加速，您应选择带有 `*-gpu.txt` 后缀的选项进行安装。

使用以下命令安装必要的包，将 [xxx] 替换为适合您需求的安装包名称：

```bash
pip install -r requirements-[xxx].txt
```

此外，对于 macOS 用户，你需要额外运行以下命令从 conda-forge 源安装特定版本的版本：

```bash
conda install -c conda-forge pyqt=5.15.9
```

#### 1.1.3 启动

完成必要步骤后，使用以下命令生成资源：

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

> [!CAUTION]
> 为避免冲突，请执行以下命令卸载第三方相关包。

```bash
pip uninstall anylabeling -y
```

设置环境变量：

```bash
# Linux 或 macOS
export PYTHONPATH=/path/to/X-AnyLabeling
# Windows
set PYTHONPATH=C:\path\to\X-AnyLabeling
```

要运行应用程序，请执行以下命令：

> [!TIP]
> 您可以通过传递 `--help` 参数随时查看可用的选项。

```python
python anylabeling/app.py
```

**参数**:

| 选项                       | 描述                                                                                               |
|----------------------------|----------------------------------------------------------------------------------------------------|
| `filename`                 | 指定图像或标签文件名。如果提供目录路径，则加载文件夹中的所有文件。                                |
| `--help`, `-h`             | 显示帮助信息并退出。                                                                               |
| `--reset-config`           | 重置 Qt 配置，清除所有设置。                                                                       |
| `--logger-level`           | 设置日志记录级别：“debug”、“info”、“warning”、“fatal”、“error”。                              |
| `--output`, `-O`, `-o`     | 指定输出文件或目录。以 `.json` 结尾的路径被视为文件。                                               |
| `--config`                 | 指定配置文件或 YAML 格式的配置字符串。默认为用户特定路径。                                          |
| `--nodata`                 | 防止在 JSON 文件中存储图像数据。                                                                   |
| `--autosave`               | 启用自动保存注释数据。                                                                             |
| `--nosortlabels`           | 禁用标签排序。                                                                                     |
| `--flags`                  | 逗号分隔的标志列表或包含标志的文件路径。                                                           |
| `--labelflags`             | 用于标签特定标志的 YAML 格式字符串或包含 JSON 格式字符串的文件。                                    |
| `--labels`                 | 逗号分隔的标签列表或包含标签的文件路径。                                                           |
| `--validatelabel`          | 指定标签验证的类型。                                                                               |
| `--keep-prev`              | 保留上一帧的注释。                                                                                 |
| `--epsilon`                | 确定在画布上找到最近顶点的 epsilon 值。                                                             |

⚠️ 请注意，如果您需要 GPU 加速，应在 [app_info.py](../../anylabeling/app_info.py) 配置文件中将 `__preferred_device__` 字段设置为 'GPU'。

### 1.2 从 GUI 运行

> 下载链接: [Github](https://github.com/CVHub520/X-AnyLabeling/releases) | [百度网盘](https://pan.baidu.com/s/1gzle9K1_84j9z1YkDuOpmA?pwd=ucqe)

相比于从源代码运行，GUI 运行环境提供了更便捷的体验，用户无需深入了解底层实现，只需解压便可直接使用。然而，其也存在一些问题，包括：
- **故障排除困难:** 如果发生崩溃或错误，可能难以快速定位具体原因，从而增加了故障排除的难度。
- **功能滞后:** GUI 版本在功能上可能落后于源代码版本，可能会导致缺少功能和兼容性问题。
- **GPU 加速限制:** 鉴于硬件和操作系统环境的多样性，当前的 GPU 推理加速服务需要用户根据需要从源代码编译。

因此，建议根据具体需求和偏好，在从源代码运行和使用 GUI 环境之间做出选择，以优化用户体验。

## 2. 使用方法

有关如何使用 X-AnyLabeling 的详细说明，请参考相应的[用户手册](./user_guide.md)。

## 3. 开发

> 请注意，以下步骤是可选的。此部分针对可能需要定制和编译软件以适应特定部署场景的用户。如果您使用软件时没有此类需求，可以跳过此部分。

<details>
<summary>展开/收起</summary>

为了方便用户在不同平台上运行 `X-AnyLabeling`，该工具提供了打包和编译的说明以及相关注意事项。在执行以下打包命令之前，根据您的环境和要求修改 [app_info.py](../../anylabeling/app_info.py) 文件中的 `__preferred_device__` 参数，以选择适当的 GPU 或 CPU 版本进行构建。

注意事项：

1. 在编译之前，请确保 `anylabeling/app_info.py` 文件中的 `__preferred_device__` 参数已根据所需的 GPU/CPU 版本进行修改。

2. 如果编译 GPU 版本，请先激活相应的 GPU 运行环境，并执行 `pip install | grep onnxruntime-gpu` 以确保其正确安装。

3. 对于编译 Windows-GPU 版本，手动修改 `x-anylabeling-win-gpu.spec` 文件中的 `datas` 列表参数，以将本地 `onnxruntime-gpu` 动态库的相关 `*.dll` 文件添加到列表中。

4. 对于编译 Linux-GPU 版本，手动修改 `x-anylabeling-linux-gpu.spec` 文件中的 `datas` 列表参数，以将本地 `onnxruntime-gpu` 动态库的相关 `*.so` 文件添加到列表中。此外，请确保根据您的 CUDA 版本下载匹配的 `onnxruntime-gpu` 包。有关详细的兼容性信息，请参阅 [官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

参考命令：

```bash
# Windows-CPU
bash scripts/build_executable.sh win-cpu

# Windows-GPU
bash scripts/build_executable.sh win-gpu

# Linux-CPU
bash scripts/build_executable.sh linux-cpu

# Linux-GPU
bash scripts/build_executable.sh linux-gpu

# macOS
bash scripts/build_executable.sh macos
```

注意：如果在 Windows 上执行上述命令时遇到权限问题，在确保完成上述准备步骤后，可以根据需要直接执行以下命令：

> pyinstaller --noconfirm anylabeling-win-cpu.spec</br>
> pyinstaller --noconfirm anylabeling-win-gpu.spec

</details>
