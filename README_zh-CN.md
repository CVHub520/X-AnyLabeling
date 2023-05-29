<p align="center">
  <img alt="X-AnyLabeling" style="width: 128px; max-width: 100%; height: auto;" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/logo.png"/>
  <h1 align="center"> 💫 X-AnyLabeling 💫</h1>
  <p align="center">轻松进行数据标注，借助<b>Segment Anything</b>和其他强大的模型提供AI支持！</p>
  <p align="center"><b>X-AnyLabeling：具备增强功能的高级自动标注解决方案</b></p>
</p>

<div align="center">

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
- [x] 提供先进的检测器，包括YOLOv6、YOLOv7、YOLOv8和DETR系列。


## I. 安装和运行

### 1. 下载和运行可执行文件

- 从[Releases](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.1.1)下载并运行最新版本。
- 对于MacOS：
  - 安装完成后，转到Applications文件夹。
  - 右键单击应用程序并选择打开。
  - 从第二次开始，您可以使用Launchpad正常打开应用程序。

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

## III. 构建可执行文件

- 安装PyInstaller：

```bash
pip install -r requirements-dev.txt
```

- 构建：

请注意，在运行之前，请根据本地conda环境在anylabeling.spec文件中替换'pathex'。

```bash
bash build_executable.sh
```

- 移步至目录 `dist/` 下检查输出。


## IV. 参考资料

- 标注用户界面的构建借鉴了LabelImg、LabelMe和Anylabeling的思想和组件。
- 使用Segment Anything Models进行自动标注。
- 使用YOLOv5、YOLOv6、YOLOv7、YOLOv8和YOLOX进行自动标注。


## 联系我们 👋

欢迎关注 CVHub，一个有爱、有趣、有料的计算机视觉专业知识分享平台，每日为您提供原创、多领域、有深度的前沿AI科技论文解读及成熟的工业级应用解决方案，学术 | 技术 | 就业一站式服务！


| Platform | Account |
| --- | --- |
| Wechat 💬 | cv_huber |
| Zhihu  🧠 | [CVHub](https://www.zhihu.com/people/cvhub-40) |
| CSDN   📚 | [CVHub](https://blog.csdn.net/CVHub?spm=1010.2135.3001.5343) |
| Github 🐱 | [CVHub](https://github.com/CVHub520) |


如果您在使用本项目的过程中有任何的疑问或碰到什么问题，请及时扫描以下二维码，备注“X-Anylabeing”添加微信好友！


![](https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/Wechat.jpg)
