## User Manual

### Annotations Upload/Export

`X-AnyLabeling` currently supports the import and export of various popular annotation formats. Here is a brief guide on how to use these features.

#### Import/Export YOLO Labels

The latest version of X-AnyLabeling allows one-click upload/export of YOLOv5/v8 labels for object detection, instance segmentation, oriented bounding box, and keypoint detection tasks (`*.txt` files).</br>

Before importing/exporting YOLO label files, prepare a label configuration file:</br>
1. For object detection, instance segmentation, and oriented bounding box, refer to [classes.txt](../../assets/classes.txt), with each line representing a class, numbered sequentially from 0.
2. For keypoint detection, refer to [yolov8_pose.yaml](../../assets/yolov8_pose.yaml), following the `has_visible` parameter as per the [official definition](https://docs.ultralytics.com/datasets/pose/#ultralytics-yolo-format).</br>

> To export in YOLO-Pose format, you need to specify a group_id for each group (bounding box and its keypoints) during the annotation process so that X-Anylabeling can understand their relationships during export.

To import labels, click the `Upload` button on the menu, select the task, upload the configuration file, then choose the label file directory and confirm.</br>
To export labels, click the `Export` button, upload the configuration file, and confirm. Exported files are saved in the `labels` folder within the current image directory.</br>

> Example YOLO label file: [demo.txt](../../assets/labels/demo.txt).

#### Import/Export COCO Labels

X-AnyLabeling supports one-click import/export of COCO labels for object detection and instance segmentation tasks (`*.json` files).</br>

Prepare a label configuration file by referring to [classes.txt](../../assets/classes.txt), with each line representing a class, numbered sequentially from 0.</br>

To import labels, click the `Upload` button on the menu, select the task, upload the configuration file, then choose the label file directory and confirm.</br>
To export labels, click the `Export` button, upload the configuration file, and confirm. Exported files are saved in the `annotations` folder within the current image directory.

> Example COCO label file: [instances_default.json](../../assets/annotations/instances_default.json).

#### Import/Export VOC Labels

X-AnyLabeling supports one-click import/export of Pascal-VOC labels for detection and segmentation tasks (`*.xml` files).</br>

To import labels, click the `Import` button on the menu, select the task, choose the label file directory and confirm.</br>
To export labels, click the `Export` button and confirm. Exported files are saved in the `Annotations` folder within the current image directory.</br>

> Example VOC label file: [demo.xml](../../assets/Annotations/demo.xml).

#### Import/Export DOTA Labels

X-AnyLabeling supports one-click import/export of DOTA label files (`*.txt`). The format is:</br>

> x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty

To import labels, click the `Upload` button on the menu, select the task, choose the label file directory and confirm.</br>
To export labels, click the `Export` button and confirm. Exported files are saved in the `labelTxt` folder within the current image directory.</br>

> Example DOTA label file: [demo_obb.txt](../../assets/labelTxt/demo_obb.txt).

#### Import/Export MASK Labels

X-AnyLabeling supports one-click import/export of semantic segmentation mask label files (`*.png`).</br>

Before importing/exporting mask label files, prepare a configuration file:</br>
1. For color maps of colored images, refer to [mask_color_map.json](../../assets/mask_color_map.json).</br>
2. For grayscale maps, refer to [mask_grayscale_map.json](../../assets/mask_grayscale_map.json).</br>

To import labels, click the `Upload` button on the menu, select the task, upload the configuration file, then choose the label file directory and confirm.</br>
To export labels, click the `Export` button, upload the configuration file, and confirm. Exported files are saved in the `mask` folder within the current image directory.</br>

#### Import/Export MOT Labels

X-AnyLabeling supports one-click import/export of multi-object tracking label files (`*.csv`).</br>

Prepare a label configuration file by referring to [classes.txt](../../assets/classes.txt), with each line representing a class, numbered sequentially from 0.</br>

To import labels, click the `Import` button on the menu, select the task, upload the configuration file, then choose the label file directory and confirm.</br>
To export labels, click the `Export` button, upload the configuration file, and confirm. Exported files are saved in the `MOT` folder within the current image directory.</br>

> Example MOT label file: [demo_video.csv](../../assets/MOT/demo_video.csv).

### Configuration File Settings

The configuration file `.xanylabelingrc` in `X-AnyLabeling` is by default stored in the current user's directory. You can refer to:

```bash
# Linux
cd ~/.xanylabelingrc

# Windows
cd C:\\Users\\xxx\\.xanylabelingrc
```

Taking the example of modifying the color of custom labels, follow these steps:

1. Open your user directory's configuration file (.xanylabelingrc). You can use a text editor or command-line tools for editing.

2. Find the field `shape_color` in the configuration file and ensure its value is "manual," indicating that you will set the label colors manually.

3. Locate the `label_colors` field, which contains the various labels and their corresponding colors.

4. In `label_colors`, find the label whose color you want to modify, such as "person," "car," "bicycle," etc.

5. Use RGB values to represent the color, where [255, 0, 0] represents red, [0, 255, 0] represents green, and [0, 0, 255] represents blue.

6. Replace the color values of the label with the desired color values, save the file, and close the editor.

Example:
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

By following these steps, you have successfully customized the colors of your labels. The next time you use these labels in your annotations, they will appear in the colors you have set.

Additionally, you can load predefined labels by modifying the `labels` field. Note that label names starting with **numbers** should be enclosed in single quotes `''`:

```YAML
...
labels:
- car
- '1'
- apple
- _phone
```

You can also set different hotkeys for various functions. For more details, please refer to the configuration file.

### Quick Tag Modification Feature

This feature provides users with a convenient way to process annotation data and supports two core operations:

- **Delete Category:** By checking the checkbox in the `Delete` column of the corresponding rows, you can mark all objects of that category for deletion.
- **Replace Category:** Fill in the new category name in the `New Value` column to replace the labels of all objects under the current category with the new category.

You can follow these steps:

1. Click on the menu bar at the top, select `Tools`-> `Change Label` option.

2. In the popped-up `Label Change Manager` dialog, perform the desired operations on the respective categories.

3. After completing all modifications, click the `Confirm` button to confirm and submit the changes.

### Quick Screenshot and Save Feature

Implementation Guide:

1. Prepare a custom class file, you can refer to the example [classes.txt](../../assets/classes.txt);

2. Click on the `Tools` menu at the top and select the `Save Cropped Image` option. Choose the corresponding custom class file for upload. This action will generate a subimage folder `x-anylabeling-crops` in the current directory, where targets will be stored based on their respective class names.

### Quick Tag Correction Feature

This functionality is designed to swiftly address two common mislabeling scenarios during the calibration process:

- Incorrectly labeled background as foreground
- Errors in foreground labeling

Follow the steps below for implementation:

1. Prepare a custom class file, for a specific example, please refer to [classes.txt](../../assets/classes.txt).

2. Click on the menu bar at the top, select `Tools` -> `Save Expanded Sub-image`, and upload the corresponding custom class file. This action will generate a subgraph folder `x-anylabeling-crops` in the current directory, storing targets according to the respective class names. The directory structure is as follows:

```
|- root
  |- images
    |- xxx.jpg
    |- xxx.json
    |- yyy.jpg
    |- yyy.json
    |- ...
  |- x-anylabeling-crops
    |- src
      |- CLASS-A
        |- xxx.jpg
        |- xxx1.jpg
        |- xxx2.jpg
        |- ...
      |- CLASS-B
        |- yyy.jpg
        |- ...
      ...
    |- dst
      |- CLASS-A
      |- CLASS-B
      |- ...
    |- meta_data.json
```

Field explanations:

- src: Original cropped image files
- dst: Cropped image files after correction
- meta_data.json: Cropped information file

2. Open the src directory and perform the following actions for each subfolder:

- Remove all erroneously labeled background boxes.
- Move all foreground boxes with category errors to the corresponding folder in the dst directory.

3. Open the X-Anylabeling tool, select `Tool` -> `Update Label`, choose the `x-anylabeling-crops` folder, click upload, and reload the source images.

> Note: This feature is applicable only to `rectangle` objects.

### Multi-Label Classification Task Annotation

Follow these steps:

1. Prepare a custom attribute label file. An example can be found in [attributes.json](../../assets/attributes.json).

2. Run `X-AnyLabeling`, click `Upload` -> `Upload Attribute File` to import the prepared attribute label file.

3. Load an image, draw a rectangular box, and ensure that the label matches the custom class label.

4. Ensure that the current mode is in edit mode; otherwise, quickly switch by pressing the shortcut `Ctrl+J`.

5. Click on the selected object, and in the top-right corner, you can annotate the label attributes.

Note: If you plan to use an AI model for pre-labeling, you can first load the corresponding attribute classification model, choose the one-click run all image function, and then fine-tune as needed.

### Rotated Object Annotation

Follow these steps:

1. Upload the image file.

2. Click the `Rotated Box` button on the left menu or press the shortcut key `O` to quickly draw a rotated box.

3. Select the corresponding rotated box object, and adjust the rotation angle using the following shortcuts:

| Shortcut | Description |
| ---      | ---         |
| z        | Rotate counterclockwise at a large angle |
| x        | Rotate counterclockwise at a small angle |
| c        | Rotate clockwise at a small angle |
| v        | Rotate clockwise at a large angle |

### Keypoint Annotation for YOLOv8 Pose Estimation

1. **Import Image Files**;

2. Click the `Rectangle` button on the left menu or press the shortcut key `R` to quickly draw a bounding box. Set the `group_id` field to ensure each object in the image has a unique identifier;

3. Click the `Point` button on the left menu to draw keypoints, and set the `group_id` field to match the corresponding bounding box ID. Additionally:

- If a keypoint is not visible, it can be ignored;
- If a keypoint is occluded, check the `useDifficult` field;

4. After completing the annotations, export the corresponding label files as described in the `Import/Export YOLO Labels` section.

### SAM Series Models

Follow these steps:

1. Click the `Brain` icon button on the left side of the menu to activate the AI function options.

2. Choose the `Segment Anything Models` series models from the Model dropdown menu.

> Note: Model accuracy and speed vary. Among them,</br>
> - `Segment Anything Model (ViT-B)` is the fastest but less accurate;</br>
> - `Segment Anything Model (ViT-H)` is the slowest and most accurate;</br>
> - `Quant` indicates a quantized model;</br>

3. Use the auto-segmentation tool to mark objects:
- `Point (q)`: Add a point belonging to the object;
- `Point (e)`: Remove a point you want to exclude from the object;
- `+Rect`: Draw a rectangle containing the object. Segment Anything will automatically segment the object.

4. `Clear (b)`: Clear all automatic segmentation marks.

5. Complete the object (f): After completing the current mark, press the shortcut key `f`, enter the label name, and save the current object.

### Multi-Object Tracking

Multi-object tracking (MOT) technology is employed to simultaneously identify and track multiple targets within video sequences, involving the association of targets across different frames. The X-AnyLabeling tool currently integrates various detection and tracking algorithms, including `ByteTrack` and `OCSort`, and supports the import and export of MOT format labels.

#### Export Settings:

1. Load a video file; for example, refer to [demo_video.mp4](../../assets/demo_video.mp4) for a sample file.
2. Load a tracking model, such as [yolov5m_bytetrack](../../anylabeling/configs/auto_labeling/yolov5m_bytetrack.yaml) or [yolov8m_ocsort](../../anylabeling/configs/auto_labeling/yolov8m_ocsort.yaml).
3. Click "Run (i)" verify correctness, and press the shortcut `Ctrl+M` to execute tracking for all frames.
4. Prepare a custom class file; see [classes.txt](../../assets/classes.txt) for a specific example.
5. Click on the menu bar -> Export -> Choose the custom class file -> Confirm. A *.csv file will be generated in the same directory as the current video file.

#### Import Settings:

1. Load a video file.
2. Prepare a custom class file; refer to [classes.txt](../../assets/classes.txt) for an example.
3. Click on the menu bar -> Upload -> Choose the custom class file -> Select *.csv label file -> Confirm. MOT labels will be imported.

### Depth Estimation

Currently, the X-AnyLabeling tool incorporates the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything.git) model, allowing users to choose a model of the desired scale (Small/Base/Large) based on their specific needs.

The implementation steps are as follows:

1. Load the image/video file.
2. Load the Depth-Anything model or any other optional depth estimation model.
3. Click on the run button, verify for accuracy, and, if all is correct, press the shortcut key `Ctrl+M` to run the process on all images at once.

The final results of the run are saved by default in the `depth` folder located in the same directory as the current image.
