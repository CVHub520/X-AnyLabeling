# QuickStart Guides

## 1. Quick Start

### 1.1 Running from Source

#### 1.1.1 Prerequisites

> [!NOTE]
> If you need to use the video tracking feature of Segment-Anything-2, please first visit [Docs](../../examples/interactive_video_object_segmentation/README.md) to install the relevant dependencies.

Before you start, ensure that you have the following prerequisites installed:

**Step 0.** Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 1.** Create a new conda environment with Python version 3.8 or higher, and activate it.

```bash
conda create --name x-anylabeling python=3.9 -y
conda activate x-anylabeling
```

#### 1.1.2 Installation

**Step 0.** Install [ONNX Runtime](https://onnxruntime.ai/).

```bash
# Install ONNX Runtime CPU
pip install onnxruntime

# Install ONNX Runtime GPU (CUDA 11.x)
pip install onnxruntime-gpu==x.x.x

# Install ONNX Runtime GPU (CUDA 12.x)
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

> [!Important]
> For GPU acceleration, please follow the instructions below to ensure that your local CUDA and cuDNN versions are compatible with your ONNX Runtime version. Additionally, install the required dependency libraries to ensure normal GPU-accelerated inference:</br>
> Ⅰ. [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)</br>
> Ⅱ. [Get started with ONNX Runtime in Python](https://onnxruntime.ai/docs/get-started/with-python.html)</br>
> Ⅲ. The ONNX Runtime version must be greater than or equal to 1.16.0.

**Step 1.** Git clone repository.

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
```

**Step 2:** Install the `requirements.txt` file.

For different configurations, X-AnyLabeling provides the following dependency files:

| Dependency File            | Operating System | Runtime Environment | Compilable |
|----------------------------|------------------|---------------------|------------|
| requirements.txt           | Windows/Linux    | CPU                 | No         |
| requirements-dev.txt       | Windows/Linux    | CPU                 | Yes        |
| requirements-gpu.txt       | Windows/Linux    | GPU                 | No         |
| requirements-gpu-dev.txt   | Windows/Linux    | GPU                 | Yes        |
| requirements-macos.txt     | MacOS            | CPU                 | No         |
| requirements-macos-dev.txt | MacOS            | CPU                 | Yes        |

- For development purposes, you should select the option with the `*-dev.txt` suffix for installation.
- To enable GPU acceleration, you should choose the option with the `*-gpu.txt` suffix for installation.

To install the necessary packages, use the following command, replacing [xxx] with the appropriate suffix for your requirements:

```bash
pip install -r requirements-[xxx].txt
```

Moreover, for macOS, you’ll need to execute an additional command to install a specific version of PyQt from the conda-forge repository:

```bash
conda install -c conda-forge pyqt=5.15.9
```

#### 1.1.3 Launch

Once you have completed the necessary steps, generate the resources with the following command:

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

To avoid potential conflicts, uninstall any existing installations of AnyLabeling with the following command:

```bash
pip uninstall anylabeling -y
```

Set the environment variable:

```bash
# linux or macos
export PYTHONPATH=/path/to/X-AnyLabeling
# windows
set PYTHONPATH=C:\path\to\X-AnyLabeling
```

To run the application, execute the following command:

```python
python anylabeling/app.py
```

**Arguments**:

| Option                     | Description                                                                                       |
|----------------------------|---------------------------------------------------------------------------------------------------|
| `filename`                 | Specifies the image or label filename. If a directory path is provided, loads all files in the folder. |
| `--help`, `-h`             | Displays the help message and exits.                                                              |
| `--reset-config`           | Resets the Qt configuration, clearing all settings.                                                |
| `--logger-level`           | Sets the logging level: "debug", "info", "warning", "fatal", "error".                             |
| `--output`, `-O`, `-o`     | Specifies the output file or directory. Paths ending with `.json` are treated as files.            |
| `--config`                 | Specifies a configuration file or YAML-formatted configuration string. Defaults to user-specific paths. |
| `--nodata`                 | Prevents storing image data in JSON files.                                                         |
| `--autosave`               | Enables automatic saving of annotation data.                                                       |
| `--nosortlabels`           | Disables sorting of labels.                                                                       |
| `--flags`                  | Comma-separated list of flags or file path containing flags.                                       |
| `--labelflags`             | YAML-formatted string or file with JSON-formatted string for label-specific flags.                 |
| `--labels`                 | Comma-separated list of labels or file path containing labels.                                     |
| `--validatelabel`          | Specifies the type of label validation.                                                           |
| `--keep-prev`              | Retains annotations from the previous frame.                                                       |
| `--epsilon`                | Determines the epsilon value for finding the nearest vertex on the canvas.                         |

⚠️Please note that if you require GPU acceleration, you should set the `__preferred_device__` field to 'GPU' in the [app_info.py](../../anylabeling/app_info.py) configuration file.

### 1.2 Running from GUI

> Download link: [Release](https://github.com/CVHub520/X-AnyLabeling/releases)

Compared to running from source code, the GUI runtime environment offers a more convenient experience. Users do not need to delve into the underlying implementation; simply extract and it's ready to use. However, there are some issues associated with it, including:
- **Difficulty in Troubleshooting:** In the event of a crash or error, it may be challenging to quickly pinpoint the exact cause, thereby increasing the difficulty of troubleshooting.
- **Feature Lag:** The GUI version may lag behind the source code version in terms of features, which could result in missing features and compatibility issues.
- **GPU Acceleration Limitations:** Given the diversity of hardware and operating system environments, the current GPU inference acceleration service requires users to compile from source code as needed.

Therefore, it is recommended to choose between running from source code or using the GUI environment based on specific needs and preferences to optimize the user experience.

## 2. Usage

For detailed instructions on how to use X-AnyLabeling, please refer to the corresponding [User Manual](./user_guide.md).

## 3. Development

> Please be aware that the subsequent procedures are optional. This part is intended for users who might require tailoring and compiling the software to suit particular deployment scenarios. Should you be utilizing the software without such needs, you may proceed to bypass this section.

<details>
<summary>Expand/Collapse</summary>

To facilitate users running `X-AnyLabeling` on different platforms, the tool provides instructions for packaging and compilation, along with relevant considerations. Before executing the packaging commands below, modify the `__preferred_device__` parameter in the [app_info.py](../../anylabeling/app_info.py) file according to your environment and requirements to select the appropriate GPU or CPU version for building.

Considerations:

1. Before compiling, ensure that the `__preferred_device__` parameter in the `anylabeling/app_info.py` file has been modified according to the desired GPU/CPU version.

2. If compiling the GPU version, activate the corresponding GPU runtime environment first, and execute `pip install | grep onnxruntime-gpu` to ensure it is correctly installed.

3. For compiling the Windows-GPU version, manually modify the `datas` list parameter in the `x-anylabeling-win-gpu.spec` file to add the relevant `*.dll` files of the local `onnxruntime-gpu` dynamic library to the list.

4. For compiling the Linux-GPU version, manually modify the `datas` list parameter in the `x-anylabeling-linux-gpu.spec` file to add the relevant `*.so` files of the local `onnxruntime-gpu` dynamic library to the list. Additionally, ensure that you download the matching `onnxruntime-gpu` package based on your CUDA version. Refer to the [official documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for detailed compatibility information.

Reference commands:

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

Note: If you encounter permission issues when executing the above commands on Windows, after ensuring the preparation steps above are completed, you can directly execute the following commands as needed:

> pyinstaller --noconfirm anylabeling-win-cpu.spec</br>
> pyinstaller --noconfirm anylabeling-win-gpu.spec

</details>
