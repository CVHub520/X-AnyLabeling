# Shape-level Classification Example

## Multi-task Classification

Multi-task classification assigns multiple attributes to each object, such as vehicle color and type.

<img src=".data/annotated_attributes.png" width="100%" />

Prepare an [attributes.json](./attributes.json) file whose keys match the object labels:

```json
{
  "vehicle": {
    "bodyColor": ["red", "white", "blue"],
    "vehicleType": ["SUV", "sedan", "bus", "truck"],
    "occluded_by": [],
    "note": ""
  },
  "__widget_types__": {
    "vehicle": {
      "bodyColor": "radiobutton",
      "vehicleType": "combobox",
      "occluded_by": "group_id",
      "note": "lineedit"
    }
  }
}
```

Supported widget types are `combobox` (default), `radiobutton`, `group_id`, and `lineedit`. See the [output example](./sources/multi-task/elon-musk-tesla.json).

Open an image folder with object annotations, launch the Image Classifier (`Ctrl+3` / `⌘+3`), and select **MultiTask**. Use the upload icon beside the object selector to load the attributes file, then select an object to edit its attributes. The configuration is also available in the main window.

Alternatively, configure a VLM in [Chatbot](../../../docs/en/chatbot.md) for AI-assisted attribute annotation. See the [Image Classifier guide](../../../docs/en/image_classifier.md#ai-assisted-labeling) for details.

To customize attribute label colors on the canvas, set `background_color`, `border_color`, and `text_color` under `canvas.attributes` in `.xanylabelingrc`, using RGBA values.

## Multiclass & Multilabel Classification

Shape-level annotations also support multiclass and multilabel flags. Prepare a [label_flags.yaml](./label_flags.yaml) file and load it through **Upload → Label Flags**, or pass it on the command line:

```bash
python anylabeling/app.py --labels labels.txt --labelflags label_flags.yaml --validatelabel exact
```

The `labelflags` keys support regular expressions. See the [output example](./sources/multi-label/worker.json) and the [image-level classification example](../image-level/README.md).
