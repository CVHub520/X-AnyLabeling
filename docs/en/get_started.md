# Quick Start Guide

## 1. Installation and Deployment

X-AnyLabeling provides multiple installation methods. You can install the official package directly via `pip` to get the latest stable version, install from source by cloning the official GitHub repository, or use the convenient GUI installer package.

> [!NOTE]
> **Advanced Features**: The following advanced features are only available through Git clone installation. Please refer to the corresponding documentation for configuration instructions.
>
> 0. **Remote Inference Service**: X-AnyLabeling-Server based remote inference service - [Installation Guide](https://github.com/CVHub520/X-AnyLabeling-Server)
> 1. **Video Object Tracking**: Segment-Anything-2 based video object tracking - [Installation Guide](../../examples/interactive_video_object_segmentation/README.md)
> 2. **Bounding Box Generation**: UPN-based bounding box generation - [Installation Guide](../../examples/detection/hbb/README.md)
> 3. **Interactive Detection & Segmentation**: Interactive object detection and segmentation with visual and text prompts - [Installation Guide](../../examples/detection/hbb/README.md)
> 4. **Smart Detection & Segmentation**: Object detection and segmentation with visual prompts, text prompts, and prompt-free modes - [Installation Guide](../../examples/grounding/yoloe/README.md)
> 5. **One-Click Training Platform**: Ultralytics framework-based training platform - [Installation Guide](../../examples/training/ultralytics/README.md)

### 1.1 Prerequisites

#### 1.1.1 Miniconda

**Step 0.** Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 1.** Create a conda environment with Python 3.10 ~ 3.12 and activate it.

> [!NOTE]
> Other Python versions require compatibility verification on your own.

```bash
# CPU Environment [Windows/Linux/macOS]
conda create --name x-anylabeling-cpu python=3.10 -y
conda activate x-anylabeling-cpu

# CUDA 11.x Environment [Windows/Linux]
conda create --name x-anylabeling-cu11 python=3.11 -y
conda activate x-anylabeling-cu11

# CUDA 12.x Environment [Windows/Linux]
conda create --name x-anylabeling-cu12 python=3.12 -y
conda activate x-anylabeling-cu12
```

#### 1.1.2 Venv

In addition to Miniconda, you can also use Python's built-in `venv` module to create virtual environments. Here are the commands for creating and activating environments under different configurations:

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
> For faster dependency installation and a more modern Python package management experience, we strongly recommend using [uv](https://github.com/astral-sh/uv) as your package manager. uv provides significantly faster installation speeds and better dependency resolution capabilities.

### 1.2 Installation

#### 1.2.1 Pip Installation

You can easily install the latest stable version of X-AnyLabeling with the following commands:

```bash
# CPU [Windows/Linux/macOS]
pip install x-anylabeling-cvhub[cpu]

# CUDA 12.x is the default GPU option [Windows/Linux]
pip install x-anylabeling-cvhub[gpu]

# CUDA 11.x [Windows/Linux]
pip install x-anylabeling-cvhub[gpu-cu11]
```

#### 1.2.2 Git Clone

**Step a.** Clone the repository.

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
cd X-AnyLabeling
```

After cloning the repository, you can choose to install the dependencies in either developer mode or regular mode according to your needs.

**Step b.1.** Developer Mode

```bash
# CPU [Windows/Linux/macOS]
pip install -e .[cpu]

# CUDA 12.x is the default GPU option [Windows/Linux]
pip install -e .[gpu]

# CUDA 11.x [Windows/Linux]
pip install -e .[gpu-cu11]
```

If you need to perform secondary development or package compilation, you can install the `dev` dependencies simultaneously, for example:

```bash
pip install -e .[cpu,dev]
```

After installation, you can verify it by running the following command:

```bash
xanylabeling checks   # Display system and version information
```

You can also run the following commands to get other information:

```bash
xanylabeling help     # Display help information
xanylabeling version  # Display version number
xanylabeling config   # Display configuration file path
```

After verification, you can run the application directly:

```bash
xanylabeling
```

> [!TIP]
> You can use `xanylabeling --help` to view all available command line options. Please refer to the **Command Line Parameters** table below for complete parameter descriptions.

| Option                     | Description                                                                                                   |
|----------------------------|---------------------------------------------------------------------------------------------------------------|
| `filename`                 | Specify the image or label filename. If a directory path is provided, all files in the folder will be loaded. |
| `--help`, `-h`             | Display help information and exit.                                                                            |
| `--reset-config`           | Reset Qt configuration, clearing all settings.                                                                |
| `--logger-level`           | Set the logging level: "debug", "info", "warning", "fatal", "error".                                          |
| `--output`, `-O`, `-o`     | Specify the output file or directory. Paths ending with `.json` are treated as files.                         |
| `--config`                 | Specify a configuration file or YAML-formatted configuration string. Defaults to a user-specific path.        |
| `--nodata`                 | Prevent storing image data in JSON files.                                                                     |
| `--autosave`               | Enable automatic saving of annotation data.                                                                   |
| `--nosortlabels`           | Disable label sorting.                                                                                        |
| `--flags`                  | Comma-separated list of flags or path to a file containing flags.                                             |
| `--labelflags`             | YAML-formatted string for label-specific flags or a file containing a JSON-formatted string.                  |
| `--labels`                 | Comma-separated list of labels or path to a file containing labels.                                           |
| `--validatelabel`          | Specify the type of label validation.                                                                         |
| `--keep-prev`              | Keep annotations from the previous frame.                                                                     |
| `--epsilon`                | Determine the epsilon value for finding the nearest vertex on the canvas.                                     |
| `--no-auto-update-check`   | Disable automatic update checks on startup.                                                                   |

> [!NOTE]
> Please refer to the X-AnyLabeling [pyproject.toml](../../pyproject.toml) file for a list of dependencies. Note that all the examples above install all required dependencies.

We also supports batch conversion between multiple annotation formats:

```bash
xanylabeling convert         # List all supported conversion tasks
xanylabeling convert <task>  # Show detailed help and examples for a specific task, i.e., xlabel2yolo
```

**Step b.2.** Regular Mode

For different configurations, X-AnyLabeling provides the following dependency files:

| Dependency File            | Operating System | Runtime | Compilable |
|----------------------------|------------------|---------|------------|
| requirements.txt           | Windows/Linux    | CPU     | No         |
| requirements-dev.txt       | Windows/Linux    | CPU     | Yes        |
| requirements-gpu.txt       | Windows/Linux    | GPU     | No         |
| requirements-gpu-dev.txt   | Windows/Linux    | GPU     | Yes        |
| requirements-macos.txt     | MacOS            | CPU     | No         |
| requirements-macos-dev.txt | MacOS            | CPU     | Yes        |

**Description**:

- If you need to perform secondary development or package compilation, please select dependency files with the `*-dev.txt` suffix.
- If you need to enable GPU acceleration, please select dependency files with the `*-gpu.txt` suffix.

Use the following command to install the necessary packages, replacing `[xxx]` with the configuration name that suits your needs:

```bash
pip install -r requirements-[xxx].txt
```

> [!NOTE]
> **Special Note for macOS Users**: You need to additionally install specific versions of packages from the conda-forge source:
> ```bash
> conda install -c conda-forge pyqt==5.15.9 pyqtwebengine
> ```

> [!IMPORTANT]
> For GPU acceleration, please follow the instructions below to ensure that your local CUDA and cuDNN versions are compatible with the ONNX Runtime version, and install the required dependencies to ensure GPU-accelerated inference works properly:
> 
> - Ⅰ. [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
> - Ⅱ. [Get started with ONNX Runtime in Python](https://onnxruntime.ai/docs/get-started/with-python.html)
> - Ⅲ. [ONNX Runtime Compatibility](https://onnxruntime.ai/docs/reference/compatibility.html)

> [!WARNING]
> For `CUDA 11.x` environments, please ensure that the versions meet the following requirements:
> - `onnx >= 1.15.0, < 1.16.1`
> - `onnxruntime-gpu >= 1.15.0, < 1.19.0`

**Optional Step**: Generate Resource Files

After completing the necessary steps, you can generate resource files using the following command:

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

**Optional Step**: Set Environment Variables

```bash
# Linux or macOS
export PYTHONPATH=/path/to/X-AnyLabeling

# Windows
set PYTHONPATH=C:\path\to\X-AnyLabeling
```

> [!CAUTION]
> **Avoid Dependency Conflicts**: To avoid conflicts with third-party packages, please uninstall the old version first:
> ```bash
> pip uninstall anylabeling -y
> ```

**Run the Application**

```bash
python anylabeling/app.py
```

> [!NOTE]
> **Special Note for Fedora KDE Users**: If you encounter slow mouse movement or response lag, try using the `--qt-platform xcb` parameter to improve performance:
> ```bash
> python anylabeling/app.py --qt-platform xcb
> ```

#### 1.2.3 GUI Installer Package

> **Download Link**: [GitHub Releases](https://github.com/CVHub520/X-AnyLabeling/releases)

Compared to running from source code, the GUI installer package provides a more convenient user experience. Users don't need to understand the underlying implementation and can use it directly after extraction. However, the GUI installer package also has some limitations:

- **Difficult Troubleshooting**: If crashes or errors occur, it may be difficult to quickly identify the specific cause, increasing the difficulty of troubleshooting.
- **Feature Lag**: The GUI version may lag behind the source code version in functionality, potentially leading to missing features and compatibility issues.
- **GPU Acceleration Limitations**: Given the diversity of hardware and operating system environments, current GPU inference acceleration services require users to compile from source code as needed.

Therefore, it is recommended to choose between running from source code and using the GUI installer package based on your specific needs and usage scenarios to optimize the user experience.

## 2. Usage

For detailed instructions on how to use X-AnyLabeling, please refer to the corresponding [User Guide](./user_guide.md).

## 3. Packaging and Compilation

> [!NOTE]
> Please note that the following steps are optional. This section is intended for users who may need to customize and compile the software to adapt to specific deployment scenarios. If you use the software without such requirements, you can skip this section.

<details>
<summary>Expand/Collapse</summary>

To facilitate users running `X-AnyLabeling` on different platforms, this tool provides packaging and compilation instructions along with relevant notes. Before executing the following packaging commands, please modify the `__preferred_device__` parameter in the [app_info.py](../../anylabeling/app_info.py) file according to your environment and requirements to select the appropriate GPU or CPU version for building.

### 3.1 Notes

- **Modify Device Configuration**: Before compiling, ensure that the `__preferred_device__` parameter in the `anylabeling/app_info.py` file has been modified according to the required GPU/CPU version.

- **Verify GPU Environment**: If compiling the GPU version, please activate the corresponding GPU runtime environment first and execute `pip list | grep onnxruntime-gpu` to ensure it is properly installed.

- **Windows-GPU Compilation**: Manually modify the `datas` list parameter in the `x-anylabeling-win-gpu.spec` file to add the relevant `*.dll` files of the local `onnxruntime-gpu` dynamic library to the list.

- **Linux-GPU Compilation**: Manually modify the `datas` list parameter in the `x-anylabeling-linux-gpu.spec` file to add the relevant `*.so` files of the local `onnxruntime-gpu` dynamic library to the list. Additionally, ensure that you download a matching `onnxruntime-gpu` package according to your CUDA version. For detailed compatibility information, please refer to the [official documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

### 3.2 Build Commands

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
> If you encounter permission issues when executing the above commands on Windows, after ensuring the above preparation steps are completed, you can directly execute the following commands:
> ```bash
> pyinstaller --noconfirm anylabeling-win-cpu.spec
> pyinstaller --noconfirm anylabeling-win-gpu.spec
> ```

</details>

