# Interactive Video Object Segmentation Example

## Introduction

**Interactive Video Object Segmentation (iVOS)** has become an essential task for efficiently obtaining object segmentations in videos, often guided by user inputs like scribbles, clicks, or bounding boxes. In this tutorial, you'll learn how to leverage the video tracking feature of [SAM2](https://github.com/facebookresearch/segment-anything-2) on X-AnyLabeling to accomplish iVOS tasks.

<img src="https://github.com/user-attachments/assets/27922366-2b29-49ff-ad91-a807a6f8c20e" width="100%" />

Let's get started!

## Installation

Before you begin, make sure you have the following prerequisites installed:

**Step 0:** Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 1:** Create a new Conda environment with Python version `3.10` or higher, and activate it:

```bash
conda create -n x-anylabeling-sam2 python=3.10 -y
conda activate x-anylabeling-sam2
```

You'll need to install SAM2 first. The code requires `torch>=2.3.1` and `torchvision>=0.18.1`. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.

Afterward, you can install SAM2 on a GPU-enabled machine using:

```bash
git clone https://github.com/CVHub520/segment-anything-2
cd segment-anything-2
pip install -e .
```

Finally, install the necessary dependencies for X-AnyLabeling (v2.4.2+):

```bash
cd ..
git clone https://github.com/CVHub520/X-AnyLabeling
cd X-AnyLabeling
```

Now, you can back to the installation guide ([简体中文](../../docs/zh_cn/get_started.md) | [English](../../docs/en/get_started.md)) to install the remaining dependencies.


## Getting Started

### Prerequisites

**Step 0:** Launch the app:

```bash
python3 anylabeling/app.py
```

**Step 1:** Load the SAM 2 Video model

![Load-Model](https://github.com/user-attachments/assets/8c3e0593-ccb5-45a8-bb61-73f4b9f5f82f)

<details>
<summary>Note: If the model fails to load due to network issues, please refer to the following settings.</summary>

First, you'll need to download a model checkpoint. For this tutorial, we'll use the [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) checkpoint as an example.

After downloading, place the checkpoint file in the corresponding model folder within your user directory (create the folder if it doesn't exist):

```bash
# Windows
C:\Users\${User}\xanylabeling_data\models\sam2_hiera_large_video-r20240901

# Linux or macOS
~/xanylabeling_data/models/sam2_hiera_large_video-r20240901
```

Additionally, if you want to use other sizes of SAM2 models or modify the model loading path, refer to this documentation for custom settings: [简体中文](../../docs/zh_cn/custom_model.md) | [English](../../docs/en/custom_model.md).

</details>

**Step 2:** Add a video file (Ctrl + O) or a folder of split video frames (Ctrl + U).

> [!NOTE]
> As of now, the supported file formats are limited to [*.jpg, *.jpeg, *.JPG, *.JPEG]. When loading video files, they will be automatically converted to jpg format by default.


### Usage

**Step 0:** Add Prompts

<video src="https://github.com/user-attachments/assets/25a05bf8-7393-4c52-baee-53a49d2859ae" width="100%" controls>
</video>

> [!TIP]
> - **Point (q):** Add a positive point.
> - **Point (e):** Add a negative point.
> - **+Rect:** Draw a rectangle around the object.
> - **Clear (b):** Erase all added marks.
> - **Finish (f):** Confirm the object.

For the initial frame, you can add prompts such as positive points, negative points, and rectangles (Marks) to guide the tracking of the desired object. Follow these steps:

1. If the segmentation result meets your expectations, click the `Finish (f)` button at the top of the screen or press the `f` key to confirm the object. If not, click the `Clear (b)` button or press the `b` key to quickly clear any invalid marks.
2. Then, you can sequentially assign custom labels and track IDs to each added target.

> [!WARNING]
> If you need to delete a confirmed object, follow these steps:</br>
> a. Open the edit mode (Ctrl + J) and remove all added objects from the current frame;</br>
> b. Click the `Reset Tracker` button at the top of the screen to reset the tracker;</br>
> c. Reapply the prompts (Marks) as described above.

![rectangle_tracklet](https://github.com/user-attachments/assets/1dbe1d41-1792-4c45-9ea0-51c26a08c6af)

Alternatively, if you only want to set up object detection tracking, you simply need to filter the output mode to Rectangle.


**Step 1:** Propagate the prompts to get the tracklet across the video

![run_video](https://github.com/user-attachments/assets/e4763f32-bfdb-4b0a-be23-4885e3cc9f96)

Once you've finished setting the prompts, you can start the video tracking by either clicking the video start button on the left-hand menu or using the shortcut `Ctrl+M` to get the tracklet throughout the entire video.

**Step 2:** Add New Prompts to Further Refine the tracklet

After tracking the entire video, if you notice any of the following issues in the middle frames:

- Target is lost
- Imperfections in boundary details
- New objects need to be tracked

You can treat the current frame as the starting frame and follow these steps:

a. Open the edit mode (`Ctrl + J`) and remove all added objects from the current frame.</br>
b. Click the `Reset Tracker` button at the top of the screen to reset the tracker.</br>
c. Reapply the prompts (Marks) as described earlier.

Then, repeat the steps in **Step 0** and **Step 1**.

![rename](https://github.com/user-attachments/assets/04707624-b13d-490f-a75d-7e35d5dee1c7)

After completing all tasks, you can:
- Use the `Tool` -> `Label Manager` option from the top menu to assign specific class names.
- Press `Alt+G` to open the GroupIDs manager and modify the track IDs if needed.

> [!NOTE]
> Just a reminder to click the `Reset Tracker` button at the top of the screen after uploading a new video file to reset the tracker.

---

Congratulations! 🎉 You’ve now mastered the basics of X-AnyLabeling. Feel free to experiment with it on your own videos and various use cases!
