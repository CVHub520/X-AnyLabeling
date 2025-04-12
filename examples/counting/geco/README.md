# GeCo Example

## Introduction

[GeCo](https://github.com/jerpelhan/GeCo) (NeurIPS'24) is a novel low-shot counter that uses a unified architecture for accurate object detection, segmentation, and count estimation, robustly generalizing object prototypes and employing a novel counting loss for direct optimization of the detection task, significantly outperforming existing methods.

<img src=".data/architecture.jpg" width="100%" />


## Usage 🚀

<img src=".data/GeCo.gif" width="100%" />

1. 📁 Load your media into X-AnyLabeling:
   - 🖼️ Images: Press `Ctrl+I` for a single image or `Ctrl+U` for a folder
   - 🎥 Videos: Press `Ctrl+O`

2. 🤖 Select and load the `GECO (ViT-huge Quant)` model. You can also download it offline - see [documentation](../../../docs/en/custom_model.md) for details.

3. ✏️ Drawing and labeling:
   - 📦 Click the `Rect` tool to start
   - 🎯 Draw one or more bounding boxes around objects of interest
   - ✅ Press `F` or click `Finish` when done
   - 🏷️ Enter the class name for the labeled object
   - ↩️ Made a mistake? Press `B` or click `Clear` to undo

