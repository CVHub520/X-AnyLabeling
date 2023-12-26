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

4. `Clear (c)`: Clear all automatic segmentation marks.

5. Complete the object (f): After completing the current mark, press the shortcut key `f`, enter the label name, and save the current object