# 常见问题解答 (FAQ)


### 安装与运行相关问题

<details>
<summary>Q: 启动时报错：ImportError: No module named expat; use SimpleXMLTreeBuilder instead</summary>

`Conda` 会自动更新系统库（如 expat）到最新版本，导致与预编译的 Python 模块不兼容。可尝试下载低版本 expat 解决，如：
```bash
conda install expat=2.5.0 -y
```
</details>

<details>
<summary>Q: 在 Fedora KDE 等环境下鼠标移动缓慢，响应延迟</summary>

在 Fedora KDE 桌面环境（包括 Wayland 和 X11）下，可能会遇到画布上鼠标指针移动缓慢、响应延迟的性能问题。

- 方法1：使用命令行参数强制使用 XCB 平台
```bash
python -m anylabeling --qt-platform xcb
```

- 方法2：设置环境变量
```bash
export QT_QPA_PLATFORM=xcb
python -m anylabeling
```

详情可参考[#1145](https://github.com/CVHub520/X-AnyLabeling/issues/1145)。
</details>

<details>
<summary>Q: 启动时报错：Failed to execute script 'app' due to unhandled exception: 'str' object does not support item assignment</summary>

删除用户目录下的配置文件后再重启。
可参考[#996](https://github.com/CVHub520/X-AnyLabeling/issues/996)。
</details>

<details>
<summary>Q: 启动时报错：OPENSSL_Uplink(00007FFE6B47AC88,08): not OPENSSL_Applink</summary>

可参考[#941](https://github.com/CVHub520/X-AnyLabeling/issues/941)。
</details>

<details>
<summary>Q: 程序运行时崩溃，出现 `Qt5Core.dll` 依赖库错误</summary>

可参考[#907](https://github.com/CVHub520/X-AnyLabeling/issues/907)。
</details>

<details>
<summary>Q: 启动时报错：Gtk-WARNING **: 17:40:30.674: Could not load a pixbuf from icon theme </summary>

可参考[#893](https://github.com/CVHub520/X-AnyLabeling/issues/893)。
</details>

<details>
<summary>Q: 启动时报错：AttributeError: 'NoneType' object has no attribute 'items'</summary>

可删除用户目录下的 `.xanylabelingrc` 文件再尝试重启。可参考[#877](https://github.com/CVHub520/X-AnyLabeling/issues/877)。
</details>

<details>
<summary>Q: 启动时报错：`Could not locate cublasLt64_12.dll. Please make sure it is in your library path!` </summary>

方案1：OnnxRunTime 库与 CUDA 版本不兼容，可参考[#844](https://github.com/CVHub520/X-AnyLabeling/issues/844)；
方案2：针对于没有全局安装 CUDA、CUDNN 的方法，可参考[#1014](https://github.com/CVHub520/X-AnyLabeling/issues/1014)。
</details>

<details>
<summary>Q: 启动时报错：`qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found` </summary>

可参考[#541](https://github.com/CVHub520/X-AnyLabeling/issues/541)、[#496](https://github.com/CVHub520/X-AnyLabeling/issues/496)。
</details>

<details>
<summary>Q: 启动时报错：`qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/usr/lib/qt/plugins/platforms/"` </summary>

可参考[#761](https://github.com/CVHub520/X-AnyLabeling/issues/761)。
</details>

<details>
<summary>Q: GPU版本运行时闪退？</summary>

可参考[#500](https://github.com/CVHub520/X-AnyLabeling/issues/500)。
</details>

<details>
<summary>Q: QStandardPaths: wrong permissions on runtime directory /run/user/1000/, 0755 instead of 0700</summary>

添加 `chmod 0700 /run/user/1000/` 到 `.bashrc` 文件中激活并重新启动应用即可。
</details>

<details>
<summary>Q: 启动软件时出现 `AttributeError: module 'importlib.resources' has no attribute 'files'` </summary>

请将 Python 版本升级至 3.9 及以上。
</details>


### 界面交互相关问题

<details>
<summary>Q: 如何快速绘制小目标物体？</summary>

可参考[#1000](https://github.com/CVHub520/X-AnyLabeling/issues/1000)。 
</details>

<details>
<summary>Q: 如何标注镂空目标用于图像分割任务？</summary>

可参考[#991](https://github.com/CVHub520/X-AnyLabeling/issues/991)。 
</details>

<details>
<summary>Q: 如何快速进行多目标关键点标签与分组标注？</summary>

可参考[#982](https://github.com/CVHub520/X-AnyLabeling/issues/982)。 
</details>

<details>
<summary>Q: 绘制矩形等目标框时，如何开启连续绘制模式？</summary>

可以打开电脑用户目录下的 .xanylabelingrc 配置文件，修改 auto_highlight_shape 和 auto_switch_to_edit_mode 为 False。
可参考[#887](https://github.com/CVHub520/X-AnyLabeling/issues/887)。 
</details>

<details>
<summary>Q: 在高分辨率显示器上界面显示模糊</summary>

可参考[#811](https://github.com/CVHub520/X-AnyLabeling/issues/811)。
</details>

<details>
<summary>Q: 标注完成后，没有弹出标签管理器自动分配标签名称？</summary>

取消勾选 `Auto Use Last Label`，可参考[#805](https://github.com/CVHub520/X-AnyLabeling/issues/805)。
</details>

<details>
<summary>Q: 复制粘贴标注对象时出现重叠？(Duplicate Polygons vs Copy & Paste Objects)</summary>

这个问题通常是因为混淆了两种复制功能的使用场景。X-AnyLabeling 其实提供了两套复制机制，它们的设计目的完全不同。

先说第一种，也就是快速复制功能，快捷键是 `Ctrl+D`。这个功能主要是为了在同一张图片里快速复制标注框。它的特别之处在于，复制出来的标注框会自动往旁边偏移几个像素（默认为2），这样就不会和原来的框完全重叠在一起。这个设计很实用，特别是当你需要标注一堆相似的目标物体时，按一下 `Ctrl+D` 就能快速复制，而且新旧标注框会自动错开，不会重叠。

第二种是传统的剪贴板复制粘贴，也就是 `Ctrl+C` 加 `Ctrl+V` 的组合。这个功能的设计初衷是用来跨图片复制标注的。比如你标注了第一张图片里某个位置的物体，想把这个标注直接复制到第二张、...、第n张图片的相同位置上。所以这个功能在粘贴时会严格按照原来的坐标位置来放置标注框，不会做任何偏移。这也是为什么如果你在同一张图片上用 `Ctrl+C` 和 `Ctrl+V`，会发现新旧标注框完全重叠在一起，看起来像是"多个标注框对应同一个对象"——其实这就是它的设计行为，并不是 bug。

另外还有个系统剪贴板的设置需要注意。默认情况下，这个功能是关闭的，也就是配置文件里的 system_clipboard 是 false。这时候你用 `Ctrl+C` 复制的标注数据会存在软件内部的一个变量里，不会影响系统剪贴板的其他内容。这样设计的好处是，即使你切换到其他图片，之前复制的标注内容还在，随时可以粘贴。有些用户可能会觉得"快捷键有概率无法正常触发复制，导致粘贴时依旧沿用上一次的复制结果"，其实这正是这个机制的体现。

如果你想启用系统剪贴板，可以在菜单里找到 Edit -> Use System Clipboard。启用后，复制的标注数据会以 JSON 格式存到系统剪贴板里，你甚至可以在记事本里看到和编辑这些数据。但要注意，这样一来每次复制都会覆盖系统剪贴板的内容。

所以总结一下使用建议：如果你只是在当前图片里复制标注框，建议用 `Ctrl+D`，它会自动帮你错开位置，避免重叠。如果你需要把标注复制到其他图片上，或者就是想让标注框精确地粘贴在原位置，那就用 `Ctrl+C` 和 `Ctrl+V`。

</details>

<details>
<summary>Q: 应用启动时，首次点击无效？</summary>

此问题暂时无解。 
</details>

<details>
<summary>Q: 无法打开 *.jpg、*.png 等格式的图片？</summary>

可参考[#823](https://github.com/CVHub520/X-AnyLabeling/issues/823)。
</details>


### 模型相关问题

<details>
<summary>Q: 在 venv 环境中运行时出现 "Error loading tokenizer: No such file or directory (os error 2)"</summary>

这个问题的根源在于 Python 的资源加载机制。当我们用 `importlib.resources.files()` 获取包内资源时，它返回的实际上是一个 `Traversable` 对象。这个对象在不同环境下的行为会有差异：在某些情况下（比如通过 `pip install -e .` 可编辑模式安装时），直接把 `Traversable` 对象当作文件路径传给需要真实路径的函数（如 `Tokenizer.from_file()`）就会出现文件找不到的错误。

我们将会在 v3.3.4+ 版本中修复了此问题，统一将资源加载方式改为 `read_text()` 方法，直接读取资源内容而不是依赖文件路径，这样就能确保在任何安装方式和运行环境下都能正常工作。如果你遇到此问题，建议更新到合适的版本。

</details>

<details>
<summary>Q: Error in model predict_shapes: ModelManager.new_auto_labeling_result[AutolabelingResult].emit(): argument 1 has unexpected type 'list'</summary>

通常是加载了损坏或无效的图像文件，或者当前图像格式不支持等原因所造成。
</details>

<details>
<summary>Q: Error in loading model: YOLOE model will be not be available.</summary>

可参考[#997](https://github.com/CVHub520/X-AnyLabeling/issues/997)。
</details>

<details>
<summary>Q: Error in loading model: yoloe with error: [WinError, 1314] 客户端没有所需的特权。mobileclip_blt.pt</summary>

可参考[#992](https://github.com/CVHub520/X-AnyLabeling/issues/992)。
</details>

<details>
<summary>Q: Error in loading custom model: Invalid config file format.</summary>

可参考[#986](https://github.com/CVHub520/X-AnyLabeling/issues/986)。
</details>

<details>
<summary>Q: Error in predict_shapes: cannot access local variable 'p' where it is not associated with a value</summary>

可参考[#983](https://github.com/CVHub520/X-AnyLabeling/issues/983)。
</details>

<details>
<summary>Q: Error in loading model :Could not download or initialize encoder data</summary>

可参考[#961](https://github.com/CVHub520/X-AnyLabeling/issues/961)。
</details>

<details>
<summary>Q: 下载的模型默认保存在什么位置？</summary>

可参考[#943](https://github.com/CVHub520/X-AnyLabeling/issues/943)。
</details>

<details>
<summary>Q: 模型预测的类别跟训练的类别对应不上</summary>

检查自定义模型配置文件（*.yaml）是不是格式没写对：
1. 默认2个字符缩进；
2. key 不能有空格；
可参考[#923](https://github.com/CVHub520/X-AnyLabeling/issues/923)。
</details>

<details>
<summary>Q: Chatbot 中如何访问 'Google Gemini' 等需要外网访问的模型？</summary>

可在当前终端或者系统配置文件中设置代理协议，其中 ip 和 port 替换为自己的网络协议地址和端口号：

```bash
export http_proxy=http://ip:port
export https_proxy=http://ip:port
```
</details>

<details>
<summary>Q: ERROR | model_manager:predict_shapes:2031 - Error in predict_shapes: '<=' not supported between instances of 'int' and 'str'</summary>

请检查模型配置文件（*.yaml）是否正确，具体可参考此[#902](https://github.com/CVHub520/X-AnyLabeling/issues/902)。
</details>

<details>
<summary>Q: Error in model prediction:openCV(4.7.0)xxx\opencv-python\opencv-python\opencv\modulesl\imgproc\src\resize.cpp:4065: error: (-215:Assertion failed) inv_scale_x >0 in function'cv::resize'</summary>

可尝试以下解决方案：
- 配置文件中置顶图片宽高信息，参考[#885](https://github.com/CVHub520/X-AnyLabeling/issues/885)。
- 检查导出模型时是否设置了动态 batch？参考[#784](https://github.com/CVHub520/X-AnyLabeling/issues/784)。
</details>

<details>
<summary>Q: 运行 yolo-pose 模型出现 `Error in loading model: 'list' object has no attribute 'items'`</summary>

未按照官方模板编写配置文件，可参考[#880](https://github.com/CVHub520/X-AnyLabeling/issues/880)。
</details>

<details>
<summary>Q: Error in predict_shapes or Error in model prediction: list index out of range. Please check the model.</summary>

可参考以下解决方案：
    - 检查标签名称是否为纯数字，若是，请务必将其加上单引号；
    - 检查配置文件中 `type` 字段是否正确定义，可参考[#837](https://github.com/CVHub520/X-AnyLabeling/issues/837)、[#878](https://github.com/CVHub520/X-AnyLabeling/issues/878)；
</details>

<details>
<summary>Q: Error with Error in loading model: exceptions must derive from BaseException</summary>

1. 请确保配置文件中，模型路径格式正确且存在。可参考[#868](https://github.com/CVHub520/X-AnyLabeling/issues/868)、[#441](https://github.com/CVHub520/X-AnyLabeling/issues/441)。
2. 检查是否是当前 onnx 模型 ir_version 版本与对应的 onnx 版本之间不匹配。可参考[#1120](https://github.com/CVHub520/X-AnyLabeling/issues/1120#issuecomment-3303864917)
</details>

<details>
<summary>Q: Error in model prediction: ‘int’ object is not subscriptable. Please check the model.</summary>

如果是非官方内置的自定义模型，请检查模型预处理、推理和后处理部分，可参考[#828](https://github.com/CVHub520/X-AnyLabeling/issues/828).
</details>

<details>
<summary>Q: 安装 SAM2 出现的报错问题：`from sam2 import _C`</summary>

可参考[#719](https://github.com/CVHub520/X-AnyLabeling/issues/719)、[#842](https://github.com/CVHub520/X-AnyLabeling/issues/842)、[#843](https://github.com/CVHub520/X-AnyLabeling/issues/843)、[#864](https://github.com/CVHub520/X-AnyLabeling/issues/865)、[#865](https://github.com/CVHub520/X-AnyLabeling/issues/865)。
</details>

<details>
<summary>Q: 下载完的模型每次重新启动应用时都被自动删除重新下载</summary>

- 注意模型路径不得有中文字符，否则会有异常。（[#600](https://github.com/CVHub520/X-AnyLabeling/issues/600)）
</details>

<details>
<summary>Q: AI模型推理时如何识别特定的类别？</summary>

当前仅支持部分模型设置此选项。具体地，以 yolo 系列模型为例，用户可通过在配置文件中添加 `filter_classes`，具体可参考此[文档](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/zh_cn/custom_model.md#%E5%8A%A0%E8%BD%BD%E5%B7%B2%E9%80%82%E9%85%8D%E7%9A%84%E7%94%A8%E6%88%B7%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B)。
</details>

<details>
<summary>Q: in paint assert len(self.points) in [1, 2, 4] AssertionError.</summary>

可参考[#491](https://github.com/CVHub520/X-AnyLabeling/issues/491)。
</details>

<details>
<summary>Q: 加载自定义模型报错？</summary>

可参考以下步骤解决：

1. 检查配置文件中的定义的类别与模型支持的类别列表是否一致。
2. 使用 [netron](https://netron.app/) 工具比较自定义模型与官方对应的内置模型输入输出节点维度是否一致。
3. 检查模型配置文件各个字段是否正确，可参考[#888](https://github.com/CVHub520/X-AnyLabeling/issues/888)。
4. 检查是否擅自修改了某个内置字段名称，可参考[#983](https://github.com/CVHub520/X-AnyLabeling/issues/983)。
</details>

<details>
<summary>Q: 运行完没有输出结果？</summary>

可参考[#536](https://github.com/CVHub520/X-AnyLabeling/issues/536)。
</details>

<details>
<summary>Q: 使用 Grounding DINO 模型进行 GPU 推理时报错：

```shell
Error in predict_shapes: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Expand node. Name:'Expand_33526' Status Message: Expand_33526: left operand cannot broadcast on dim 2 LeftShape: {1,900,4}, RightShape: {1,900,256}
```
</summary>

可参考[#389](https://github.com/CVHub520/X-AnyLabeling/issues/389)。
</details>

<details>
<summary>Q: Missing config : num_masks</summary>

可参考[#515](https://github.com/CVHub520/X-AnyLabeling/issues/515)。
</details>

<details>
<summary>Q: AttributeError: 'int' object has no attribute 'replace'</summary>
查看配置文件是否有定义纯数字标签。请注意在定义以**纯数字**命名的标签名称时，请务必将其加上单引号 `''`
</details>

<details>
<summary>Q: Unsupported model IR version: 11, max supported IR version: 10</summary>

可参考以下解决方案：

```bash
import onnx
onnx_model = onnx.load("/path/to/your/onnx_model")
onnx_model.ir_version = 10  # 修改为合适的版本号
onnx.save(onnx_model, "/path/to/onnx_model")
```
</details>

<details>
<summary>Q: Your model ir_version is higher than the checker's.</summary>
onnx 版本过低，请更新：

```shell
pip install --upgrade onnx
```
</details>

<details>
<summary>Q: 如何在 MacOS 上使用 Segment-Anthing Video 功能？</summary>

可参考[#865](https://github.com/CVHub520/X-AnyLabeling/issues/865)。
</details>

<details>
<summary>Q: 运行 GPU 版本时出现 `FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file: No such file or directory` 错误</summary>

可参考[此教程](../zh_cn/get_started.md)中 "步骤 0. 安装 ONNX Runtime" 安装匹配版本。
此外，可查看[#834](https://github.com/CVHub520/X-AnyLabeling/issues/834)。
</details>

<details>
<summary>Q: 运行 GPU 版本时出现 `ImportError: DLL load failed while importing onnx_cpp2py_export: 动态链接库(DLL)初始化例程失败` 错误</summary>

onnx 和 onnxruntime 库版本不兼容，具体可参考[#886](https://github.com/CVHub520/X-AnyLabeling/issues/886)
</details>

<details>
<summary>Q: 运行编译好的 GPU 版本时发现模型长时间处于加载状态</summary>

这通常是由于 onnx 和 onnxruntime-gpu 版本不匹配导致的。请参考[ONNX Runtime 官方兼容性文档](https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support)确保安装了匹配的 `onnx` 版本。
</details>


### 文件相关问题

<details>
<summary>Q: 导出 Mask 图为空</summary>

请按照以下步骤依次检查下：
1. 按下 `ctrl+g` 查看当前任务的标注类别和标注类型，确保所有标注类型为`polygon`对象；
2. 检查导出配置文件的格式正确，且类别名称与上述步骤所示的标签类别名称相符；
3. 升级到最新版本或通过源码运行获取更全面的日志信息。

详情可参考此[#1153](https://github.com/CVHub520/X-AnyLabeling/issues/1153)。
</details>

<details>
<summary>Q: 预编译版本加载视频文件没有成功解析处图片</summary>

可参考此[#1136](https://github.com/CVHub520/X-AnyLabeling/issues/1136)。
</details>

<details>
<summary>Q: 在 Labelme 中导入 X-AnyLabeling 的标签会提示 `AssertionError` 错误</summary>

可参考此[#1007](https://github.com/CVHub520/X-AnyLabeling/issues/1007)。
</details>

<details>
<summary>Q: 导出标签时发生错误：Error occurred while exporting annotations. 'xxx' is not in list</summary>

请按照以下步骤依次检查下：
1. 按下 `ctrl+g` 查看当前任务的标注类别；
2. 检查导出时上传的标签类别文件（`classes.txt`）中定义的类别是否与标注类别名称一一对应；
3. 检查是否选择了错误的导出类型，例如标注的是矩形框，导出时却选择了分割或旋转框等选项；
</details>

<details>
<summary>Q: 加载图片目录时，出现 `segmentation fault` 段错误</summary>

可参考此[#906](https://github.com/CVHub520/X-AnyLabeling/issues/906)。
</details>

<details>
<summary>Q: 上传标签文件时发生错误，出现 `cannot identify image file xxx`</summary>

请检查图片文件和标签文件有没有分目录存放。可参考此[#911](https://github.com/CVHub520/X-AnyLabeling/issues/869)。
</details>

<details>
<summary>Q: 加载文件时，出现 `a bytes-like object is required, not 'NoneType'`，确保 xxx.json 是一个有效的标签文件</summary>

请检查 *.json 文件中，`imagePath` 字段值是否与图像文件名一致。具体可参考此[#869](https://github.com/CVHub520/X-AnyLabeling/issues/869)。
</details>

<details>
<summary>Q: 导入的标签文件为空</summary>

请检查是否存在以下情况：
    - 标注类型与导出类型不一致，例如标注的的是 `rectangle` 矩形框，导出时选择 `Polygon` 选项；
    - 导入的图像文件夹存在多级嵌套的子文件夹，可参考[#839](https://github.com/CVHub520/X-AnyLabeling/issues/839)；
</details>

<details>
<summary>Q: 导入和导出 yolo 关键点标签时闪退</summary>

请检查模型配置文件（*.yaml）是否正确，具体可参考此[#898](https://github.com/CVHub520/X-AnyLabeling/issues/898)。
</details>

<details>
<summary>Q: 导入标签时出现 `invalid literal for int() with base 10`</summary>

可参考[#782](https://github.com/CVHub520/X-AnyLabeling/issues/782)。
</details>

<details>
<summary>Q: Export mask error: "imageWidth"</summary>

可参考[#477](https://github.com/CVHub520/X-AnyLabeling/issues/477)。
</details>

<details>
<summary>Q: operands could not be broadcast together with shapes (0,) (1,2)</summary>

可参考[#492](https://github.com/CVHub520/X-AnyLabeling/issues/492)。
</details>

<details>
<summary>Q: 导出关键点标签时报错：int0 argument must be a string, a byteslike object or a number, not 'NoneType'</summary>

`group_id`字段缺失，请确保每个矩形框和关键点都有对应的群组编号。
</details>

<details>
<summary>Q: 'NoneType' object has no attribute 'shape'</summary>

检查文件路径是否包含**中文字符**。
</details>
