# SAM3 Interactive Video Object Segmentation Example

## Introduction

SAM3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor SAM2, SAM3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase. SAM3 Video performs Promptable Concept Segmentation (PCS) on videos, taking text as prompts and automatically detecting and tracking all matching object instances across video frames.

<img src=".data/model_diagram.png" width="100%" />

In this tutorial, you'll learn how to leverage the video tracking feature of [SAM3](https://github.com/facebookresearch/sam3) on X-AnyLabeling to accomplish iVOS tasks. Let's get started!

## Installation

Please refer to [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server) for download, installation, and server setup instructions. After installation, make sure to enable `segment_anything_3_video` in the `configs/models.yaml` file (see [example](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/configs/models.yaml)).

## Usage

Launch the X-AnyLabeling client, press `Ctrl+A` or click the `AI` button in the left menu bar to open the auto-labeling panel. In the model dropdown list, select `Remote-Server`, then choose `Segment Anything 3 Video`.

### Text Prompting

<video src="https://github.com/user-attachments/assets/4cc2755b-1c79-445c-97f0-7fcf93045b6a" width="100%" controls>
</video>

1. Enter object names in the text field (e.g., `person`, `car`, `bicycle`)
2. Click **Send** to initiate detection
3. After verification, click the `Auto Run` button in the left menu bar or press `Ctrl+M` to start forward propagation

> [!NOTE]
> Each session supports one category only. Propagation starts from the current frame and runs to the end of the video, with a brief warm-up period of approximately 15 frames before processing begins.

> [!WARNING]
> If you cancel the propagation task midway, all processed frame results will be lost. Please wait for the task to complete or ensure you have saved the results before canceling.

### Visual Prompting

We currently don't support this feature—please stay tuned.