## Quick Start Guide

### Running Modes

`X-AnyLabeling` currently supports two modes of operation: running the source code, or downloading the precompiled GUI version directly. 

To ensure access to the latest features and stable performance, it is highly recommended to run from the source code.

#### Running from Source Code

1. **Download Source Code:**
   ```bash
   git clone https://github.com/CVHub520/X-AnyLabeling.git
   ```

2. **Install Dependencies:**
   - Choose the appropriate dependency file based on the operating environment and runtime (CPU or GPU).
   - Example:
     ```bash
     # upgrade pip to its latest version
     pip install -U pip

     pip install -r requirements-dev.txt
     ```

3. **Launch the Tool:**
   Execute the following command in the `X-AnyLabeling` project directory:
   ```bash
   python anylabeling/app.py
   ```

   > Set the environment variable:
   > - Linux/MacOS: `export PYTHONPATH=/path/to/X-AnyLabeling`
   > - Windows: `set PYTHONPATH=C:\path\to\X-AnyLabeling`

#### Running in GUI Environment

Running in the GUI environment is convenient but may have limitations compared to running from the source code. Consider the pros and cons based on your specific needs and preferences.

Download Link: [Release](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v2.2.0) | [Baidu Disk](https://pan.baidu.com/s/1TK2-cBN-sLI84dqHUe7A2w?pwd=oevw)

Note:
- For MacOS:
  - After installation, go to the Applications folder.
  - Right-click on the application and choose Open.
  - From the second time onwards, you can open the application normally using Launchpad.

- Due to the lack of necessary hardware, the current tool is only available in executable versions for `Windows` and `Linux`. If you require executable programs for other operating systems, e.g., `MacOS`, please refer to the following steps for self-compilation.

### File Import

`X-AnyLabeling` supports importing images or videos through shortcuts (Ctrl+I, Ctrl+U, Ctrl+O). Note that the default annotation file is saved in the import file path.

If you need to save to a different directory, you can click on the top-left `File` -> `Save As`, and then choose the destination directory for saving.

### Quick Annotation

The tool supports various annotation styles (Polygon, Rectangle, Rotated Box, Circle, Line, Linestrip, Point). Use shortcut keys (e.g., P for Polygon, R for Rectangle, O for Rotation) for quick drawing.

| Annotation Style | Shortcut | Application |
|-------------------|----------|--------------|
| Polygon           | P        | Image Segmentation |
| Rectangle         | R        | Horizontal Object Detection |
| Rotated Box       | O        | Rotational Object Detection |
| Circle            | -        | Specific Scenarios |
| Line              | -        | Lane Detection |
| Polyline          | -        | Vessel Segmentation |
| Point             | -        | Key Point Detection |

Currently, the tools has two main interaction modes:

- **Edit Mode:** In this state, users can move, copy, paste, and modify objects.
- **Draw Mode:** In this state, only drawing of the corresponding annotation style is supported.

Note: for the annotation styles of **Rectangle**, **Rotated Box**, **Circle**, **Line**, and **Point**, when the drawing is completed, the tool will automatically switch to edit mode. For the other two styles, you can quickly switch by using the shortcut `Ctrl+J`.

### Auxiliary Inference

For AI algorithm functions, click the AI icon or use the shortcut Ctrl+A to access the model list. Choose the desired model for use.

### One-Click Run

The "One-Click Run" feature automates annotation tasks for the current batch. Click the Play icon or use the shortcut Ctrl+M. Note that this feature requires an activated model and should be tested on a small batch before full deployment.

### Packaging and Compilation

> Please note that the following steps are not mandatory. This section is provided for users who may need to customize and compile the software for distribution in specific environments. If you are simply using the software, you can skip this step.

<details>
<summary>Expand/Collapse</summary>

To facilitate users running `X-AnyLabeling` on different platforms, the tool provides instructions for packaging and compilation, along with relevant considerations. Before executing the packaging commands below, modify the `__preferred_device__` parameter in the [app_info.py](../../anylabeling/app_info.py) file according to your environment and requirements to select the appropriate GPU or CPU version for building.

Considerations:

1. Before compiling, ensure that the `__preferred_device__` parameter in the `anylabeling/app_info.py` file has been modified according to the desired GPU/CPU version.

2. If compiling the GPU version, activate the corresponding GPU runtime environment first, and execute `pip install | grep onnxruntime-gpu` to ensure it is correctly installed.

3. For compiling the Windows-GPU version, manually modify the `datas` list parameter in the `anylabeling-win-gpu.spec` file to add the relevant `*.dll` files of the local `onnxruntime-gpu` dynamic library to the list.

4. For compiling the Linux-GPU version, manually modify the `datas` list parameter in the `anylabeling-linux-gpu.spec` file to add the relevant `*.so` files of the local `onnxruntime-gpu` dynamic library to the list. Additionally, ensure that you download the matching `onnxruntime-gpu` package based on your CUDA version. Refer to the [official documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for detailed compatibility information.

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
```

Note: If you encounter permission issues when executing the above commands on Windows, after ensuring the preparation steps above are completed, you can directly execute the following commands as needed:

> pyinstaller --noconfirm anylabeling-win-cpu.spec</br>
> pyinstaller --noconfirm anylabeling-win-gpu.spec

</details>
