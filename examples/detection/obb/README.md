# Oriented Bounding Boxes Object Detection Example

## Introduction

Oriented object detection surpasses standard object detection by adding angular precision to pinpoint objects in images.

<img src=".data/annotated_obb_task.png" width="100%" />

## Usage

The calibration process for the rotated object detection task is as follows:

- Import the image file.
- Click the `rotations` button on the left menu bar or press the shortcut key `O` to quickly draw a rotated box.
- Select the corresponding rotated box object and adjust the rotation angle using the following shortcut keys:

| Shortcut Key | Description                |
| ------------ | -------------------------- |
| z            | Rotate counterclockwise by a large angle |
| x            | Rotate counterclockwise by a small angle |
| c            | Rotate clockwise by a small angle |
| v            | Rotate clockwise by a large angle |

To display the rotation degrees, you can click 'View' in the top menu bar, and check 'Show Degrees' to display the estimated rotation angle in real-time.
