<p align="center">
  <img alt="X-AnyLabeling" style="width: 128px; max-width: 100%; height: auto;" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/logo.png"/>
  <h1 align="center"> 💫 X-AnyLabeling 💫</h1>
  <p align="center">Effortless data labeling with AI support from <b>Segment Anything</b> and other awesome models!</p>
  <p align="center"><b>X-AnyLabeling: Advanced Auto Labeling Solution with Added Features</b></p>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)


**Auto Labeling with Segment Anything**

<a href="https://b23.tv/AcwX0Gx">
  <img style="width: 800px; margin-left: auto; margin-right: auto; display: block;" alt="AnyLabeling-SegmentAnything" src="https://github.com/CVHub520/Resources/blob/main/X-Anylabeling/demo.gif"/>
</a>


**Features:**

- [x] Image annotation for polygon, rectangle, circle, line and point.
- [x] Auto-labeling with YOLOv5 and Segment Anything.
- [x] Text detection, recognition and KIE (Key Information Extraction) labeling.
- [x] Multiple languages availables: English, Vietnamese, Chinese.

**Highlight:**

- [x] Detection-Guided Fine-grained Classification.
- [x] Offer face detection with keypoint detection.
- [x] Provide advanced detectors, including YOLOv6, YOLOv7, YOLOv8, and DETR series.


## I. Install and run

### 1. Download and run executable

- Download and run newest version from [Releases](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v0.1.1).
- For MacOS:
  - After installing, go to Applications folder
  - Right click on the app and select Open
  - From the second time, you can open the app normally using Launchpad

### 2. Install from Pypi

Not ready yet, coming soon...

## II. Development

- Install packages

```bash
pip install -r requirements.txt
```

- Generate resources:

```bash
pyrcc5 -o anylabeling/resources/resources.py anylabeling/resources/resources.qrc
```

- Run app:

```bash
python anylabeling/app.py
```

## III. Build executable

- Install PyInstaller:

```bash
pip install -r requirements-dev.txt
```

- Build:

Note: Please replace the 'pathex' in the anylabeling.spec file according to the local conda environment before running.

```bash
bash build_executable.sh
```

- Check the outputs in: `dist/`.


## IV. References

- Labeling UI built with ideas and components from [LabelImg](https://github.com/heartexlabs/labelImg), [LabelMe](https://github.com/wkentaro/labelme), [Anylabeling](https://github.com/vietanhdev/anylabeling).
- Auto-labeling with [Segment Anything Models](https://segment-anything.com/).
- Auto-labeling with [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
