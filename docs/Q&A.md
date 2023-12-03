## 文档说明

本文档旨在针对大家在使用 `X-AnyLabeling` 项目中碰到的一些常见问题进行梳理，以更好地帮助大家使用该工具，后续会不间断地更新此文档。</br>

## 功能支持

Q: **`X-AnyLabeling` 目前支持哪些标注样式？**</br>
A: 当前支持**多边形**、**矩形框**、**旋转框**、**圆形**、**直线**、**线段**和**点**。</br>

Q: **`X-AnyLabeling` 目前提供哪些内置模型？**</br>
A: 详情可移步至 [models_list](./models_list.md) 文档查看。

## 标签转换

Q: **如何将打标完的 `*.json` 标签文件转换为 `YOLO`/`VOC`/`COCO` 等主流格式？**</br>
A: X-AnyLabeling 工具目前内置了多种主流数据格式的导出，包括但不仅限于 YOLO/VOC/COCO。</br>
为了满足更多元化的需求，针对自定义(`custom`)格式，额外向大家提供了 `tools/label_converter.py` 一键转换脚本。</br>

```bash
#=============================================================================== Usage ================================================================================#
#
#---------------------------------------------------------------------------- custom2voc  -----------------------------------------------------------------------------#
# python tools/label_converter.py --task rectangle --src_path custom_folder --dst_path voc_folder --mode custom2voc
#
#---------------------------------------------------------------------------- voc2custom  -----------------------------------------------------------------------------#
# python tools/label_converter.py --task rectangle --src_path voc_folder --img_path img_folder --mode voc2custom
#
#---------------------------------------------------------------------------- custom2yolo  ----------------------------------------------------------------------------#
# python tools/label_converter.py --task rectangle --src_path custom_folder --dst_path yolo_folder --classes xxx.txt --mode custom2yolo
# python tools/label_converter.py --task polygon --src_path custom_folder --dst_path yolo_folder --classes xxx.txt --mode custom2yolo
#
#---------------------------------------------------------------------------- yolo2custom  ----------------------------------------------------------------------------#
# python tools/label_converter.py --task rectangle --src_path yolo_folder --img_path img_folder --classes xxx.txt --mode yolo2custom
# python tools/label_converter.py --task polygon --src_path yolo_folder --img_path img_folder --classes xxx.txt --mode yolo2custom
#
#---------------------------------------------------------------------------- custom2coco  ----------------------------------------------------------------------------#
# python tools/label_converter.py --task rectangle --src_path custom_folder --dst_path coco_folder --classes xxx.txt --mode custom2coco
# python tools/label_converter.py --task polygon --src_path custom_folder --dst_path coco_folder --classes xxx.txt --mode custom2coco
#
#---------------------------------------------------------------------------- coco2custom  ----------------------------------------------------------------------------#
# python tools/label_converter.py --task rectangle --src_path coco.json --img_path img_folder --mode coco2custom
# python tools/label_converter.py --task polygon --src_path coco.json --img_path img_folder --mode coco2custom
#
#---------------------------------------------------------------------------- custom2dota  ----------------------------------------------------------------------------#
# python tools/label_converter.py --task rotation --src_path dota_image_folder --dst_path save_folder --mode custom2dota
#
#---------------------------------------------------------------------------- dota2custom  ----------------------------------------------------------------------------#
# python tools/label_converter.py --task rotation --src_path dota_label_folder --img_path dota_image_folder --mode dota2custom
#
#---------------------------------------------------------------------------- dota2dcoco   ----------------------------------------------------------------------------#
# python tools/label_converter.py --task rotation --src_path dota_label_folder --dst_path xxx.json --img_path dota_image_folder --classes xxx.txt --mode dota2dcoco
#
#---------------------------------------------------------------------------- dcoco2dota   ----------------------------------------------------------------------------#
# python tools/label_converter.py --task rotation --src_path xxx.json --dst_path dota_folder --mode dcoco2dota
#
#=============================================================================== Usage ================================================================================#
```

注意：
1. 目前 `--task` 支持的任务有 ['rectangle', 'polygon', 'rotation'] 即矩形框、多边形框和旋转框三种，方便大家快速接入训练框架。至于其它的任务如关键点等，可参考脚本自行修改下。</br>
2. 此处 `--classes` 参数指定的  `*.txt` 文件是用户预定义的类别文件，每一行代表一个类别，类别编号按从上到下的顺序编排，可参考此文件[classes.txt](../assets/classes.txt)。</br>

Q: **语义分割任务如何将输出的标签文件转换为 \*.png 格式输出？**</br>
A: 当前 `X-AnyLabeling` 中同样支持了 png 掩码图的导出，仅需在标注之前准备好一份自定义颜色映射表文件（具体可参考[mask_color_map.json](../assets/mask_color_map.json)和[mask_grayscale_map.json](../assets/mask_grayscale_map.json)文件，分别用于将当前分割结果映射为相应地RGB格式活灰度图格式掩码图，可按需选取），并将当前导出格式设置为 `MASK` 格式并导入事先准备好的文件即可，掩码图默认保存在与图像文件同级目录下。</br>
当然，针对工具本身自定义(`custom`)的格式，我们可以使用工程目录下的 `tools/polygon_mask_conversion.py` 脚本 （仅支持二分类转换）轻松转换，以下是参考的转换指令：

```bash
python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --mode poly2mask
# [option] 如果标签和图像不在同一目录下，请使用以下命令：
python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --json_path xxx_folder --mode poly2mask
```

此外，也支持将掩码图一键转换为自定义格式导入 `X-AnyLabeling` 中进行修正，输出的 `*.json` 文件默认保存至 'img_path' 目录：

```bash
python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --mode mask2poly
```

## 工具使用

Q: **如何修改自定义快捷键？**
</br>
A: 可通过修改当前设备用户根目录下的 `.anylabelingrc` 文件：

```bash
#Linux
cd ~/.anylabelingrc

#Windows
cd C:\\Users\\xxx\\.anylabelingrc
```

Q: **如何使用X-AnyLabeling自动标注功能？**
A: 可参考此篇[文章](https://zhuanlan.zhihu.com/p/667668033)

Q：**如何使用GPU加速推理？**</br>
A：由于 `X-AnyLabeling` 现阶段的 IR 是基于 `OnnxRuntime` 库实现的。因此，如需使用 GPU 推理，需安装 `onnxruntime-gpu` 版本并确保与机器的 CUDA 版本相匹配，具体参照表可参考[官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。此外，当确定当前环境可用时，只需将此[文件](../anylabeling/app_info.py)中的 `__preferred_device__` 字段设置为 `GPU` 即可。

Q: **如何进行多分类属性标注**</br>
A: 可参考以下步骤：
1. 准备好一个自定义属性标签文件，具体示例可参考 [attributes.json](../assets/attributes.json)；
2. 运行`X-AnyLabeling`，点击`Import`-`Import Attributes File`导入上一步的标签文件；
3. 加载图片，绘制目标矩形框，标签必须与自定义类别标签一致；
4. 右键或者快捷键`Ctrl+J`打开编辑模式；
5. 点击选中目标，右上角即可进行标签属性的标注。
注：如果你打算使用AI模型进行预打标，可先加载对应的模型，选择一键运行所有图像功能，再进行微调即可。

Q: **如何进行旋转目标标注**</br>
A: 可参考此篇[博客](https://blog.csdn.net/CVHub/article/details/134216999)

Q: **如何使用 SAM 系列模型？**</br>
A: 可参考以下步骤实施：
1. 点击菜单栏左侧的`Brain`标志按钮以激活AI功能选项；
2. 从下拉菜单`Model`中选择`Segment Anything Models`系列模型；
> 说明：模型精度和速度因模型而异，其中</br>
> - `Segment Anything Model (ViT-B)` 是最快的但精度不高；</br>
> - `Segment Anything Model (ViT-H)` 是最慢和最准确的；</br>
> - `Quant` 表示量化过的模型；</br>
3. 使用自动分割标记工具标记对象：
- `+Point`：添加一个属于对象的点；
- `-Point`：移除一个你想从对象中排除的点；
- `+Rect`：绘制一个包含对象的矩形。Segment Anything 将自动分割对象。
4. 清除：清除所有自动分割标记；
5. 完成对象(f)：当完成当前标记后，我们可以及时按下快捷键f，输入标签名称并保存当前对象。

> 注意</br>
> 1. `X-AnyLabeling` 在第一次运行任何模型时，需要从服务器下载模型，需要一段时间，这具体取决于本地的网络速度；</br>
> 2. 第一次进行 AI 推理也需要时间，请耐心等待，因此后台任务会运行以缓存`SAM`模型的“编码器”，在接下来的图像中自动分割模型需要时间会缩短。</br>

Q: **如何支持自定义模型？**</br>
A: 请参考 [custom_model.md](./custom_model.md) 文档。

Q: **如何编译打包成可执行文件？**</br>
A：可参考以下打包指令：</br>

```bash
#Windows-CPU
bash scripts/build_executable.sh win-cpu

#Windows-GPU
bash scripts/build_executable.sh win-gpu

#Linux-CPU
bash scripts/build_executable.sh linux-cpu

#Linux-GPU
bash scripts/build_executable.sh linux-gpu
```

> 注意事项：</br>
> 1. 编译前请针对相应的 GPU/CPU 版本修改 `anylabeling/app_info.py` 文件中的 `__preferred_device__` 参数；</br>
> 2. 如果需要编译`GPU`版本，请通过`pip install -r requirements-gpu-dev.txt`安装对应的环境；
>    - 对于 `Windows-GPU` 版本的编译，请自行修改 `anylabeling-win-gpu.spec` 的 `datas` 列表参数，将您本地的`onnxruntime-gpu`的相关动态库`*.dll`添加进列表中；</br>
>    - 对于 `Linux-GPU` 版本的编译，请自行修改 `anylabeling-linux-gpu.spec` 的 `datas` 列表参数，将您本地的`onnxruntime-gpu`的相关动态库`*.so`添加进列表中；</br>
>    - 此外，下载 `onnxruntime-gpu` 包是需要根据 `CUDA` 版本进行适配，具体匹配表可参考[官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)说明。</br>
> 3. 对于 `macos` 版本可自行参考 `anylabeling-win-*.spec` 脚本进行修改。

Q: **使用YOLO系列模型如何只标注自己需要的类别？**</br>
A：可以通过添加 `filter_classes` 字段来实现，具体方法与步骤可参考 [#121](https://github.com/CVHub520/X-AnyLabeling/issues/121).

## 问题反馈

Q: **初始化加载时下载模型失败如何处理？**</br>
A: 由于当前模型权重暂存在 `github` 上，下载前请先开启科学上网，否则大概率会因为网络问题下载失败；如果不具备条件的可考虑手动下载方式，将下载好的 `*.onnx` 文件放置到当前系统用户目录下的 `anylabeling_data/models/xxx` 下。</br>
这里 `xxx` 为对应的模型的名称，具体地可参考 [`X-anylabeling/anylabeling/configs/auto_labeling/models.yaml`](../anylabeling/configs/auto_labeling/models.yaml) 文件中的对应 `model_name` 字段，示例如下：

```bash
(x-anylabeling) cvhub@CVHub:~/anylabeling_data$ tree
.
└── models
    ├── mobile_sam_vit_h-r20230810
    │   ├── mobile_sam.encoder.onnx
    │   └── sam_vit_h_4b8939.decoder.onnx
    ├── yolov5s-r20230520
    │   └── yolov5s.onnx
    ├── yolov6lite_s_face-r20230520
    └── yolox_l_dwpose_ucoco-r20230820
        ├── dw-ll_ucoco_384.onnx
        └── yolox_l.onnx
```

Q: 如何修改自定义标签颜色？
A: 您可以通过以下步骤来修改自定义标签的颜色：

1. 打开您的用户目录下的配置文件(.anylabelingrc)，您可以使用文本编辑器或命令行工具进行编辑。

2. 在配置文件中找到字段 `shape_color`，确保其值为 "manual"，这表示您将手动设置标签的颜色。

3. 定位到 `label_colors` 字段，这是一个包含各个标签及其对应颜色的部分。

4. 在 `label_colors` 中，找到您想要修改颜色的标签，比如 "person"、"car"、"bicycle" 等。

5. 使用 RGB 值来表示颜色，例如 [255, 0, 0] 表示红色，[0, 255, 0] 表示绿色，[0, 0, 255] 表示蓝色。

6. 将您想要设置的颜色值替换到相应标签的值中，保存文件并关闭编辑器。

具体示例如下：
```YAML
...
default_shape_color: [0, 255, 0]
shape_color: manual  # null, 'auto', 'manual'
shift_auto_shape_color: 0
label_colors:
  person: [255, 0, 0]
  car: [0, 255, 0]
  bicycle: [0, 0, 255]
  ...
...
```

这样，您已成功修改了自定义标签的颜色。下次在标注过程中使用这些标签时，它们将显示您所设置的颜色。

Q: **自定义快捷键不生效？**</br>
A: 可参考 [#100](https://github.com/CVHub520/X-AnyLabeling/issues/100).

Q: **pip安装包时遇到lap库安装失败怎么办？**</br>
A: 可参考 [#124](https://github.com/CVHub520/X-AnyLabeling/issues/124).

Q: **能正常加载模型，但推理没结果？**</br>
A: 请先下载源码，在终端运行，查看具体的报错信息再尝试解决，如解决不了可在 [issue](https://github.com/CVHub520/X-AnyLabeling/issues) 提交反馈。

Q: **Linux 环境下 GUI 界面中文显示乱码？**</br>
A: 这是由于当前系统环境缺乏中文字体，可以参考以下步骤配置。

```
cd /usr/share/fonts
sudo mkdir myfonts
cd myfonts
sudo cp /mnt/c/Windows/Fonts/simsun.ttc .
sudo mkfontscale
sudo mkfontdir
sudo fc-cache -fv
```

> 注：此处 `/mnt/c/Windows/Fonts/simsun.ttc` 为新宋体文件，大家可以将自己喜欢的支持中文的 *.ttc 文件拷贝到 `myfonts` 目录下即可。