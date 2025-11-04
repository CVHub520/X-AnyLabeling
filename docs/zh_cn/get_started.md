# 快速入门指南

## 1. 安装部署

X-AnyLabeling 提供了多种安装方法，您可以通过 `pip` 直接安装官方软件包获取最新的稳定版本，或通过克隆官方 GitHub 仓库进行源码安装，GUI 安装包也是一种便捷的选择。

> [!NOTE]
> **高级功能说明**：以下高级功能仅适用于 Git 克隆方式，如需使用，请先参考对应的文档进行配置。
>
> 0. **远程推理服务指南**：基于 X-AnyLabeling-Server 的远程推理服务 - [安装指南](https://github.com/CVHub520/X-AnyLabeling-Server)
> 1. **视频目标追踪**：基于 Segment-Anything-2 的视频目标追踪 - [安装指南](../../examples/interactive_video_object_segmentation/README.md)
> 2. **目标候选框生成**：基于 UPN 的目标候选框生成 - [安装指南](../../examples/detection/hbb/README.md)
> 3. **交互式检测分割**：基于视觉和文本提示的交互式目标检测和分割 - [安装指南](../../examples/detection/hbb/README.md)
> 4. **智能检测分割**：基于视觉和文本提示及免提示的目标检测和分割 - [安装指南](../../examples/grounding/yoloe/README.md)
> 5. **一键训练平台**：基于 Ultralytics 框架的一键训练平台 - [安装指南](../../examples/training/ultralytics/README.md)

### 1.1 前置条件

#### 1.1.1 Miniconda

**步骤 0.** 从 [官方网站](https://docs.anaconda.com/miniconda/) 下载并安装 Miniconda。

**步骤 1.** 创建一个 Python 3.10 ~ 3.12 版本的 conda 环境，并激活它。

> [!NOTE]
> 其他 Python 版本需要自行验证兼容性。

```bash
# CPU 环境 [Windows/Linux/macOS]
conda create --name x-anylabeling-cpu python=3.10 -y
conda activate x-anylabeling-cpu

# CUDA 11.x 环境 [Windows/Linux]
conda create --name x-anylabeling-cu11 python=3.11 -y
conda activate x-anylabeling-cu11

# CUDA 12.x 环境 [Windows/Linux]
conda create --name x-anylabeling-cu12 python=3.12 -y
conda activate x-anylabeling-cu12
```

#### 1.1.2 Venv

除了 Miniconda，您也可以使用 Python 内置的 `venv` 模块创建虚拟环境。以下是不同配置下的环境创建和激活命令：

```bash
# CPU [Windows/Linux/macOS]
python3.10 -m venv venv-cpu
source venv-cpu/bin/activate  # Linux/macOS
# venv-cpu\Scripts\activate    # Windows

# CUDA 12.x [Windows/Linux]
python3.12 -m venv venv-cu12
source venv-cu12/bin/activate  # Linux
# venv-cu12\Scripts\activate    # Windows

# CUDA 11.x [Windows/Linux]
python3.11 -m venv venv-cu11
source venv-cu11/bin/activate  # Linux
# venv-cu11\Scripts\activate    # Windows
```

> [!TIP]
> 对于追求更快的依赖安装速度和更现代化的 Python 包管理体验，强烈推荐使用 [uv](https://github.com/astral-sh/uv) 作为包管理器。uv 提供了显著更快的安装速度和更好的依赖解析能力。

### 1.2 安装

#### 1.2.1 Pip 安装

您可以通过以下命令轻松安装 X-AnyLabeling 的最新稳定版本：

```bash
# CPU [Windows/Linux/macOS]
pip install x-anylabeling-cvhub[cpu]

# CUDA 12.x 是 GPU 版本的默认选项 [Windows/Linux]
pip install x-anylabeling-cvhub[gpu]

# CUDA 11.x [Windows/Linux]
pip install x-anylabeling-cvhub[gpu-cu11]
```

#### 1.2.2 Git 克隆

**步骤 a.** 克隆代码仓库。

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
cd X-AnyLabeling
```

克隆完仓库以后，您可以根据需要自行选择开发者模式或常规模式安装相应的依赖。

**步骤 b.1.** 开发者模式

```bash
# CPU [Windows/Linux/macOS]
pip install -e .[cpu]

# CUDA 12.x 是 GPU 版本的默认选项 [Windows/Linux]
pip install -e .[gpu]

# CUDA 11.x [Windows/Linux]
pip install -e .[gpu-cu11]
```

如果您需要进行二次开发或打包编译，可同步安装 `dev` 依赖，例如：

```bash
pip install -e .[cpu,dev]
```

安装完成后，可执行以下命令进行验证：

```bash
xanylabeling checks   # 显示系统及版本信息
```

您也可以运行以下命令获取其他信息：

```bash
xanylabeling help     # 显示帮助信息
xanylabeling version  # 显示版本号
xanylabeling config   # 显示配置文件路径
```

验证无误后，可直接运行应用程序：

```bash
xanylabeling
```

> [!TIP]
> 您可以通过 `xanylabeling --help` 查看所有可用的命令行选项。完整的参数说明请参考下方的**命令行参数**表格。

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
| `--no-auto-update-check`   | 禁用启动时的自动更新检查。                                                                         |

> [!NOTE]
> 请参阅 X-AnyLabeling [pyproject.toml](../../pyproject.toml) 文件以获取依赖项列表。请注意，以上所有示例都安装了所有必需的依赖项。

此外，还支持多种标签格式之间的批量转换功能：

```bash
xanylabeling convert         # 列出所有支持的转换任务
xanylabeling convert <task>  # 查看特定转换任务的详细帮助和使用示例，例如：xlabel2yolo
```

**步骤 b.2.** 常规模式

对于不同的配置，X-AnyLabeling 提供了以下依赖文件：

| 依赖文件                   | 操作系统        | 运行环境 | 可编译   |
|----------------------------|-----------------|----------|----------|
| requirements.txt           | Windows/Linux   | CPU      | 否       |
| requirements-dev.txt       | Windows/Linux   | CPU      | 是       |
| requirements-gpu.txt       | Windows/Linux   | GPU      | 否       |
| requirements-gpu-dev.txt   | Windows/Linux   | GPU      | 是       |
| requirements-macos.txt     | MacOS           | CPU      | 否       |
| requirements-macos-dev.txt | MacOS           | CPU      | 是       |

**说明**：

- 如果您需要进行二次开发或打包编译，请选择带有 `*-dev.txt` 后缀的依赖文件。
- 如需启用 GPU 加速，请选择带有 `*-gpu.txt` 后缀的依赖文件。

使用以下命令安装依赖包，将 `[xxx]` 替换为适合您需求的配置名称：

```bash
pip install -r requirements-[xxx].txt
```

> [!NOTE]
> **macOS 用户特别说明**：需要额外从 conda-forge 源安装特定版本的软件包：
> ```bash
> conda install -c conda-forge pyqt==5.15.9 pyqtwebengine
> ```

> [!IMPORTANT]
> 对于 GPU 加速，请按照以下说明，确保您本地的 CUDA 和 cuDNN 版本与 ONNX Runtime 版本兼容，并安装所需依赖库，以确保 GPU 加速推理正常运行：
> 
> - Ⅰ. [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
> - Ⅱ. [Get started with ONNX Runtime in Python](https://onnxruntime.ai/docs/get-started/with-python.html)
> - Ⅲ. [ONNX Runtime Compatibility](https://onnxruntime.ai/docs/reference/compatibility.html)

> [!WARNING]
> 对于 `CUDA 11.x` 环境，请务必确保版本满足以下要求：
> - `onnx >= 1.15.0, < 1.16.1`
> - `onnxruntime-gpu >= 1.15.0, < 1.19.0`

**可选步骤**：生成资源文件

完成必要步骤后，可使用以下命令生成资源文件：

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

**可选步骤**：设置环境变量

```bash
# Linux 或 macOS
export PYTHONPATH=/path/to/X-AnyLabeling

# Windows
set PYTHONPATH=C:\path\to\X-AnyLabeling
```

> [!CAUTION]
> **避免依赖冲突**：为避免与第三方包冲突，请先执行以下命令卸载旧版本：
> ```bash
> pip uninstall anylabeling -y
> ```

**运行应用程序**

```bash
python anylabeling/app.py
```

> [!NOTE]
> **Fedora KDE 用户特别说明**：如果遇到鼠标移动缓慢或响应延迟的问题，可以尝试使用 `--qt-platform xcb` 参数来提升性能：
> ```bash
> python anylabeling/app.py --qt-platform xcb
> ```

#### 1.2.3 GUI 安装包

> **下载链接**: [GitHub Releases](https://github.com/CVHub520/X-AnyLabeling/releases)

相比于从源代码运行，GUI 安装包提供了更便捷的使用体验，用户无需深入了解底层实现，只需解压即可直接使用。然而，GUI 安装包也存在一些局限性：

- **故障排除困难**：如果发生崩溃或错误，可能难以快速定位具体原因，从而增加了故障排除的难度。
- **功能滞后**：GUI 版本在功能上可能落后于源代码版本，可能会导致功能缺失和兼容性问题。
- **GPU 加速限制**：鉴于硬件和操作系统环境的多样性，当前的 GPU 推理加速服务需要用户根据需要从源代码编译。

因此，建议根据具体需求和使用场景，在从源代码运行和使用 GUI 安装包之间做出选择，以优化使用体验。

## 2. 使用方法

有关如何使用 X-AnyLabeling 的详细说明，请参考相应的[用户手册](./user_guide.md)。

## 3. 打包编译

> [!NOTE]
> 请注意，以下步骤是可选的。此部分针对可能需要定制和编译软件以适应特定部署场景的用户。如果您使用软件时没有此类需求，可以跳过此部分。

<details>
<summary>展开/收起</summary>

为了方便用户在不同平台上运行 `X-AnyLabeling`，该工具提供了打包和编译的说明以及相关注意事项。在执行以下打包命令之前，请根据您的环境和要求修改 [app_info.py](../../anylabeling/app_info.py) 文件中的 `__preferred_device__` 参数，以选择适当的 GPU 或 CPU 版本进行构建。

### 3.1 注意事项

- **修改设备配置**：在编译之前，请确保 `anylabeling/app_info.py` 文件中的 `__preferred_device__` 参数已根据所需的 GPU/CPU 版本进行修改。

- **验证 GPU 环境**：如果编译 GPU 版本，请先激活相应的 GPU 运行环境，并执行 `pip list | grep onnxruntime-gpu` 以确保其正确安装。

- **Windows-GPU 编译**：手动修改 `x-anylabeling-win-gpu.spec` 文件中的 `datas` 列表参数，以将本地 `onnxruntime-gpu` 动态库的相关 `*.dll` 文件添加到列表中。

- **Linux-GPU 编译**：手动修改 `x-anylabeling-linux-gpu.spec` 文件中的 `datas` 列表参数，以将本地 `onnxruntime-gpu` 动态库的相关 `*.so` 文件添加到列表中。此外，请确保根据您的 CUDA 版本下载匹配的 `onnxruntime-gpu` 包。有关详细的兼容性信息，请参阅[官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

### 3.2 编译命令

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

> [!TIP]
> 如果在 Windows 上执行上述命令时遇到权限问题，在确保完成上述准备步骤后，可以直接执行以下命令：
> ```bash
> pyinstaller --noconfirm anylabeling-win-cpu.spec
> pyinstaller --noconfirm anylabeling-win-gpu.spec
> ```

</details>
