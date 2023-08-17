<p align="center">
  <img alt="X-AnyLabeling" style="width: 128px; max-width: 100%; height: auto;" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/logo.png"/>
  <h1 align="center"> 💫 X-AnyLabeling 💫</h1>
  <p align="center">轻松进行数据标注，借助<b>Segment Anything</b>和其他强大的模型提供AI支持！</p>
  <p align="center"><b>X-AnyLabeling：具备增强功能的高级自动标注解决方案</b></p>
</p>

<div align="center">


[中文文档](https://mp.weixin.qq.com/s/Fi7i4kw0n_QsA7AgmtP-JQ)

简体中文 | [English](README.md)

</div>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)


**使用Segment Anything进行自动标注**

<a href="https://b23.tv/AcwX0Gx">
  <img style="width: 800px; margin-left: auto; margin-right: auto; display: block;" alt="AnyLabeling-SegmentAnything" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/demo.gif"/>
</a>


**功能特点：**

- [x] 支持多边形、矩形、圆形、直线和点的图像标注。
- [x] 借助YOLOv5和Segment Anything进行自动标注。
- [x] 文本检测、识别和KIE（关键信息提取）标注。
- [x] 支持多种语言：英语、中文。

**亮点：**

- [x] 基于检测的细粒度分类。
- [x] 提供人脸检测和关键点检测。
- [x] 提供先进的检测器，包括YOLOX、YOLOv6、YOLOv7、YOLOv8 和 DETR 系列。
- [x] 支持转换成标准的COCO-JSON、VOC-XML 以及 YOLOv5-TXT 文件格式。

**🚀 新特性：**

- [x] 支持 YOLO-NAS [2023-06-15]
- [x] 支持 YOLOv8-Segmentation [2023-06-20]
- [x] 支持 MobileSAM, MedSAM 以及 LVM-Med [2023-08-10]

## I. 安装和运行

### 1. 下载和运行可执行文件

- 从[百度网盘(提取码: xnqx)](https://pan.baidu.com/s/12nETv3CTTcitGFfnWmcefA)下载并运行最新版本。

- 对于MacOS：
  - 安装完成后，转到Applications文件夹。
  - 右键单击应用程序并选择打开。
  - 从第二次开始，您可以使用Launchpad正常打开应用程序。

注意：目前我们仅为Windows操作系统提供带有图形用户界面（GUI）的可执行程序。对于其他操作系统的用户，您可以按照[步骤Ⅲ](#build)的说明自行编译程序。

### 2. 从Pypi安装

暂未准备好，即将推出...


## II. 开发

- 安装依赖包

```bash
pip install -r requirements.txt
```

- 生成资源：

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- 运行应用程序：

```bash
python anylabeling/app.py
```

## III. 构建可执行文件 <span id="build">编译</span>

- 安装PyInstaller：

```bash
pip install -r requirements-dev.txt
```

- 构建：

请注意，在运行之前，请根据本地conda环境在anylabeling.spec文件中替换'pathex'。

```bash
bash scripts/build_executable.sh
```

- 移步至目录 `dist/` 下检查输出。


## IV. 参考资料

- 本项目继承自 [Anylabeling](https://github.com/vietanhdev/anylabeling) 并在此基础上扩展了丰富的功能，非常感谢 @[vietanhdev](https://github.com/vietanhdev) 开源如此出色的工具。
- 支持 [MMPreTrain](https://github.com/open-mmlab/mmpretrain), [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [timm](https://github.com/huggingface/pytorch-image-models) 等主流框架。
- 使用 [YOLOv5](https://github.com/ultralytics/yolov5)、[YOLOv6](https://github.com/meituan/YOLOv6)、[YOLOv7](https://github.com/WongKinYiu/yolov7)、[YOLOv8](https://github.com/ultralytics/ultralytics)、[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)、[YOLO-NAS](https://github.com/Deci-AI/super-gradients)和[Segment Anything Models](https://segment-anything.com/) 进行自动标注。


## 联系我们 👋

如果您在使用本项目的过程中有任何的疑问或碰到什么问题，请及时扫描以下二维码，备注“X-Anylabeing”添加微信好友，我们将给予力所能及的帮助！


![](https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/Wechat.jpg)
