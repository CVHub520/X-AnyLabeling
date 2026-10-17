# GroundingPI

## Overview

[GroundingPI](https://github.com/groundingpi/GroundingPI) is a 4B-parameter grounding foundation model designed to provide spatial perception for physical intelligence. It combines a visual encoder with a language backbone, using staged training on visual primitives and an autoregressive spatial-token format to represent object locations. The model localizes targets from category names, natural-language descriptions, or reference boxes, producing bounding boxes or points for detection, referring-expression grounding, point localization, and visual prompting.

| Task | Input | Output |
| --- | --- | --- |
| Detection | Categories, for example `dog.cup.lamp` | Rectangle boxes with category labels |
| Grounding | A description, for example `the dog on the sofa` | Rectangle boxes |
| Pointing | A description, for example `the dog on the sofa` | Point annotations |
| Visual Prompting | One or more positive reference boxes | Rectangle boxes around similar objects |

## Installation

Follow the X-AnyLabeling-Server installation guide ([English](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/docs/source/en/get_started.md) / [简体中文](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/docs/source/zh_cn/get_started.md)), including the GroundingPI requirements listed there.

Enable `groundingpi` in `configs/models.yaml`; its model settings are in `configs/auto_labeling/groundingpi.yaml`.

## Usage

Open an image in X-AnyLabeling, press `Ctrl+A` to open the AI panel, select **Remote-Server**, enter the Server URL, and choose **GroundingPI**.

### Detection

Select **Detection**, enter categories such as `dog.cup.lamp`, and click **Send**. Commas, periods, newlines, and mixed separators are supported; repeated categories are removed while preserving their order.

![Detection with category prompts](.data/detection.jpg)

### Grounding

Select **Grounding**, enter `the dog on the sofa`, and click **Send**. The description is kept as one phrase, including its internal punctuation.

![Grounding with a description](.data/grounding.jpg)

### Pointing

Select **Pointing**, enter `the dog on the sofa`, and click **Send**. This task returns point annotations instead of boxes.

![Pointing with a description](.data/pointing.jpg)

### Visual Prompting

Select **Visual Prompting**, click **+Rect**, and draw a positive reference box around an example object. Click **Run Rect** to find similar objects, then click **Finish** and enter a label, such as `panda`, to name the results. **Clear** removes prompt marks and unfinished predictions while keeping finished annotations.

![Visual prompting with a positive reference box](.data/visual_prompting.jpg)

The screenshots use the Server's `assets/demo/images/demo.jpg` and `assets/demo/images/demo_pandas.jpg`.

For all tasks, **Replace (On)** replaces existing annotations. Click it to switch to **Preserve (On)** and keep existing annotations when applying new results.

Detection, Grounding, and Pointing support text-prompt batch processing: open an image folder, choose the task, click **Auto Run**, select the image range, and enter the prompt. Visual Prompting requires per-image reference boxes and disables batch processing.

## Notes

- GroundingPI generates coordinates on a 0–999 grid; the Server restores them to the original image size. It does not provide confidence scores or confidence/IoU controls.
- The model can return false positives for absent targets. In the demo image, an `elephant` query returned a box even though no elephant was present, in both upstream inference and the adapter. Review predictions before accepting annotations.
- Negative reference boxes, point prompts, segmentation, and video are not exposed by this adapter. Truncated or malformed model responses are rejected rather than applied as partial annotations.
