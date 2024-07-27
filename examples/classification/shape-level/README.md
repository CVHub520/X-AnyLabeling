# Shape-level Classification Example

## Multi-task Classification

### Introduction

**Multi-task Classification** involves training a model to perform multiple classification tasks simultaneously. For example, a model could be trained to classify both the type of person and vehicle attributes in a single image.

<img src=".data/annotated_person_attributes.png" width="100%" />
<img src=".data/annotated_vehicle_attributes.png" width="100%" />

### Usage

**Step 0: Preparation**

Prepare a attributes file like [attributes.json](./attributes.json). An example is shown below:

```json
{
    "vehicle": {
        "bodyColor": [
            "red",
            "white",
            "blue"
        ],
        "vehicleType": [
            "SUV",
            "sedan",
            "bus",
            "truck"
        ]
    },
    "person": {
        "wearsGlasses": ["yes", "no"],
        "wearsMask": ["yes", "no"],
        "clothingColor": [
            "red",
            "green",
            "blue"
        ]
    }
}

```

**Step 1: Run the Application**

```bash
python anylabeling/app.py
```

**Step 2: Upload the Configuration File**

Click on `Upload -> Upload Attributes File` in the top menu bar and select the prepared configuration file to upload.

For detailed output examples, refer to [this file](./sources/multi-task/elon-musk-tesla.json).


## Multiclass & Multilabel Classification

Similar to [Image-Level Classification](../image-level/README.md), you can also conduct multiclass and multilabel classification for Shape-Level Annotation.

<img src=".data/annotated_person_flags.png" width="100%" />
<img src=".data/annotated_helmet_flags.png" width="100%" />

## Usage

### GUI Import (Recommended)

**Step 0: Preparation**

Prepare a flags file like [label_flags.yaml](./label_flags.yaml). An example is shown below:

```YAML
person:
  - male
  - female
helmet:
  - white
  - red
  - blue
  - yellow
  - green
```

**Step 1: Run the Application**

```bash
python anylabeling/app.py
```

**Step 2: Upload the Configuration File**

Click on `Upload -> Upload Label Flags File` in the top menu bar and select the prepared configuration file to upload.

### Command Line Loading

**Option 1: Quick Start**

```bash
python anylabeling/app.py --labels person,helmet --labelflags "{'person': ['male', 'female'], 'helmet': ['white', 'red', 'blue', 'yellow', 'green']}" --validatelabel exact
```

> [!TIP]
> The `labelflags` key field supports regular expressions. For instance, you can use patterns like `{person-\d+: [male, tall], "dog-\d+": [black, brown, white], .*: [occluded]}`.

**Option 2: Using a Configuration File**

```bash
python anylabeling/app.py --labels labels.txt --labelflags label_flags.yaml --validatelabel exact
```

For detailed output examples, refer to [this file](./sources/multi-label/worker.json).


