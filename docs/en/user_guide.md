## User Manual

### Annotations Upload/Export

Currently, `X-AnyLabeling` provides functionality for importing/exporting labels in six mainstream annotation formats. Below are the steps for each format.

- Upload YOLO Labels

Before importing YOLO annotation files, you need to prepare a custom class file. Once prepared, import the image file to be annotated, click the `Upload` button on the top menu, select `Upload YOLO Labels`, choose the prepared label file, and then select the directory containing the label files.

> Custom class file format can be referred to [classes.txt](../../assets/classes.txt).

- Upload VOC Labels

As Pascal-VOC XML label files come with category information, simply click the `Upload` button on the top menu, select `Upload VOC Labels`, and then choose the directory containing the label files.

> XML label file format can be referred to [demo.xml](../../assets/Annotations/demo.xml).

- Upload COCO Labels

Similar to VOC labels, importing COCO labels is straightforward. Click the `Upload` button on the top menu, select `Upload COCO Labels`, and choose the COCO JSON file.

> JSON label file format can be referred to [instances_default.json](../../assets/annotations/instances_default.json).

- Upload DOTA Labels

Click the `Upload` button on the top menu, select `Upload DOTA Labels`, and choose the directory containing the label files.

- Upload MASK Labels

Mask labels are mainly used for segmentation algorithms. Currently, `X-AnyLabeling` supports two forms: grayscale and color modes. Before importing mask label files, you need to prepare a corresponding color mapping table file (*.json). Then, following the previous steps, click the `Upload` button on the top menu, select `Upload MASK Labels`, choose the defined color mapping table file, and finally select the directory containing the mask label files.

> Color mapping table file for color images can be referred to [mask_color_map.json](../../assets/mask_color_map.json),</br>
> Color mapping table file for grayscale images can be referred to [mask_grayscale_map.json](../../assets/mask_grayscale_map.json).

- Upload MOT Labels

MOT labels are used for multi-object tracking algorithms, and the label file type is `*.csv`. Similar to importing YOLO label files, you need to prepare a custom class file. Then, click the `Upload` button, select `Upload MOT Labels`, choose the custom class file, and finally import the CSV format MOT label file.

---

Export settings are similar to import settings. After completing the annotation task and verifying its correctness, click the export button in the top menu, and choose the appropriate format for export. Note that for different export styles, you may need to prepare class files or color mapping table files in advance, as described in the import settings.

### Configuration File Settings

The configuration file `.anylabelingrc` in `X-AnyLabeling` is by default stored in the current user's directory. You can refer to:

```bash
# Linux
cd ~/.anylabelingrc

# Windows
cd C:\\Users\\xxx\\.anylabelingrc
```

Taking the example of modifying the color of custom labels, follow these steps:

1. Open your user directory's configuration file (.anylabelingrc). You can use a text editor or command-line tools for editing.

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

Now, you have successfully modified the color of the custom labels. The next time you use these labels during annotation, they will appear in the color you have set. Similarly, you can set some default configurations according to your needs, such as modifying the `labels` field for predefined labels or setting different shortcut keys for triggering settings based on your preferences.

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
