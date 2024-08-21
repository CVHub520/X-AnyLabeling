# Multi-Object Tracking Example

## Introduction

Multi-Object Tracking (MOT) technology is used to simultaneously identify and track multiple targets within video sequences, involving the association of targets across different frames.

<img src=".data/tracking_any_task.gif" width="100%" />

## Usage

X-AnyLabeling is enhanced with a versatile tracking system capable of handling multiple tasks: - **Object Detection**
- **Oriented Bounding Boxes Object Detection**
- **Instance Segmentation**
- **Pose Estimation** 

It incorporates advanced tracking algorithms such as `ByteTrack` and `Bot-Sort` to ensure robust and accurate tracking across these diverse tasks.

Here's how to perform multi-object tracking using the tool:
1. Load the video file, e.g., [Bangkok.mp4](https://github.com/user-attachments/assets/b94db3da-c7fa-469c-a153-a7c98bb56e14). (Note: The file path must not contain Chinese characters!)
2. Load the tracking model, such as [yolov8n_obb_botsort](../../anylabeling/configs/auto_labeling/yolov8n_obb_botsort.yaml), [yolov8s_det_botsort](../../anylabeling/configs/auto_labeling/yolov8s_det_botsort.yaml), [yolov8m_seg_bytetrack](../../anylabeling/configs/auto_labeling/yolov8m_seg_bytetrack.yaml), [yolov8x_pose_p6_botsort](../../anylabeling/configs/auto_labeling/yolov8x_pose_p6_botsort.yaml) or a custom model.
3. Click to run, and after verifying that everything is correct, you can use the shortcut `Ctrl+M` to run all frames at once.

In this process, the `group_id` field represents the `track_id` of the current target box.

> [!WARNING]
> If object confidence score will be low, i.e lower than track_high_thresh, then there will be no tracks successfully returned and updated.

> [!TIP]
> You can open the Group ID Manager with Alt+G to modify the group_id. :)

## Export

For instructions on exporting MOT annotations, please consult the user guide available:
- [English version](../../docs/en/user_guide.md)
- [Chinese version](../../docs/zh_cn/user_guide.md)
