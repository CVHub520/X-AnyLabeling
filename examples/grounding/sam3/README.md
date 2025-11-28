# SAM 3: Segment Anything with Concepts

## Overview

[SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor SAM 2, SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase or exemplars.

<img src=".data/model_diagram.png" width="100%" />

## Installation

Please refer to [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server) for download, installation, and server setup instructions.

## Usage

Launch the X-AnyLabeling client, press `Ctrl+A` or click the `AI` button in the left menu bar to open the auto-labeling panel. In the model dropdown list, select `Remote-Server`, then choose `Segment Anything 3`.

### Text Prompting

<video src="https://github.com/user-attachments/assets/d442d08b-fff7-4673-9a61-1b4ea6862f7d" width="100%" controls>
</video>

1. Enter object names in the text field (e.g., `person`, `car`, `bicycle`)
2. Separate multiple classes with periods or commas: `person.car.bicycle` or `dog,cat,tree`
3. Click **Send** to initiate detection

### Visual Prompting

<video src="https://github.com/user-attachments/assets/5a953314-2611-42fc-81b7-90fbe3b018ee" width="100%" controls>
</video>

1. Click **+Rect** or **-Rect** to activate drawing mode
2. Draw bounding boxes around target objects or regions of interest (use **+Rect** for positive prompts, **-Rect** for negative prompts)
3. Add multiple prompts for different object instances
4. Click **Run Rect** to process visual cues
5. Click **Finish** (or press `f`) to complete the object, enter the label category and confirm, or use **Clear** to remove all visual prompts
