# Multi-Object Tracking Example

## Introduction

Multi-Object Tracking (MOT) technology is used to simultaneously identify and track multiple targets within video sequences, involving the association of targets across different frames.

<img src=".data/tracking_any_task.gif" width="100%" />

## Usage

X-AnyLabeling is enhanced with a versatile tracking system capable of handling multiple tasks:
- **Object Detection**
- **Oriented Bounding Boxes Object Detection**
- **Instance Segmentation**
- **Pose Estimation** 

It incorporates advanced tracking algorithms such as [TrackTrack](https://docs.ultralytics.com/modes/track#tracktrack), [ByteTrack](https://docs.ultralytics.com/modes/track#bytetrack) and [Bot-Sort](https://docs.ultralytics.com/modes/track#bot-sort) to ensure robust and accurate tracking across these diverse tasks.

Here's how to perform multi-object tracking using the tool:
1. Load the video file, e.g., [Bangkok.mp4](https://github.com/user-attachments/assets/b94db3da-c7fa-469c-a153-a7c98bb56e14). (Note: The file path must not contain Chinese characters!)
2. Load a built-in tracking model from the table below, or load a custom tracking model.
3. Click to run, and after verifying that everything is correct, you can use the shortcut `Ctrl+M` to run all frames at once.

Built-in tracking model configurations:

| Task | Tracker | Model config |
| :--- | :--- | :--- |
| Object Detection | BoT-SORT | [yolov5s_det_botsort](../../anylabeling/configs/auto_labeling/yolov5s_det_botsort.yaml), [yolov8s_det_botsort](../../anylabeling/configs/auto_labeling/yolov8s_det_botsort.yaml), [yolo11s_det_botsort](../../anylabeling/configs/auto_labeling/yolo11s_det_botsort.yaml) |
| Object Detection | TrackTrack | [yolo26s_det_tracktrack](../../anylabeling/configs/auto_labeling/yolo26s_det_tracktrack.yaml) |
| Oriented Bounding Box Detection | BoT-SORT | [yolov8n_obb_botsort](../../anylabeling/configs/auto_labeling/yolov8n_obb_botsort.yaml), [yolo11s_obb_botsort](../../anylabeling/configs/auto_labeling/yolo11s_obb_botsort.yaml) |
| Oriented Bounding Box Detection | TrackTrack | [yolo26s_obb_tracktrack](../../anylabeling/configs/auto_labeling/yolo26s_obb_tracktrack.yaml) |
| Instance Segmentation | ByteTrack | [yolov8m_seg_bytetrack](../../anylabeling/configs/auto_labeling/yolov8m_seg_bytetrack.yaml) |
| Instance Segmentation | BoT-SORT | [yolo11s_seg_botsort](../../anylabeling/configs/auto_labeling/yolo11s_seg_botsort.yaml) |
| Instance Segmentation | TrackTrack | [yolo26s_seg_tracktrack](../../anylabeling/configs/auto_labeling/yolo26s_seg_tracktrack.yaml) |
| Pose Estimation | BoT-SORT | [yolov8x_pose_p6_botsort](../../anylabeling/configs/auto_labeling/yolov8x_pose_p6_botsort.yaml), [yolo11s_pose_botsort](../../anylabeling/configs/auto_labeling/yolo11s_pose_botsort.yaml) |
| Pose Estimation | TrackTrack | [yolo26s_pose_tracktrack](../../anylabeling/configs/auto_labeling/yolo26s_pose_tracktrack.yaml) |

In this process, the `group_id` field represents the `track_id` of the current target box. During actual tracking, the first few frames may not have a `group_id` if the object confidence score is below the tracking threshold.

> [!NOTE]
> If you have not started a new task, run `Reset Tracker` before tracking a new video or image sequence.

> [!WARNING]
> If object confidence score will be low, i.e lower than track_high_thresh, then there will be no tracks successfully returned and updated.

> [!TIP]
> You can open the Group ID Manager with Alt+G to modify the group_id. :)

## Export

For instructions on exporting MOT annotations, please consult the user guide available:
- [English version](../../docs/en/user_guide.md)
- [Chinese version](../../docs/zh_cn/user_guide.md)
