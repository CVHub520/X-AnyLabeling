# SAM3 Interactive Video Object Segmentation Example

## Introduction

SAM3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor SAM2, SAM3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase. SAM3 Video performs Promptable Concept Segmentation (PCS) on videos, taking text as prompts and automatically detecting and tracking all matching object instances across video frames.

<img src=".data/model_diagram.png" width="100%" />

In this tutorial, you'll learn how to leverage the video tracking feature of [SAM3](https://github.com/facebookresearch/sam3) on X-AnyLabeling to accomplish iVOS tasks. Let's get started!

## Installation

You'll need to get X-AnyLabeling-Server up and running first. Check out the [installation guide](https://github.com/CVHub520/X-AnyLabeling-Server) for the details. Make sure you're running at least v0.0.4 of the server and v3.3.4 of the X-AnyLabeling client, otherwise you might run into compatibility issues.

Once that's done, head over to `configs/models.yaml` and enable `segment_anything_3_video`. There's an [example config](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/configs/models.yaml) you can reference if you're not sure how to set it up.

Now for the model files. For users in China, the easiest way is to grab everything from [ModelScope](https://modelscope.cn/models/facebook/sam3/files) - they host both the main SAM3 model and the `bpe_simple_vocab_16e6.txt.gz` file (that's the text encoding vocabulary). If you're outside China or prefer GitHub, you can get the BPE file from [here](https://github.com/CVHub520/X-AnyLabeling/releases/download/v3.0.0/bpe_simple_vocab_16e6.txt.gz), though the download might be slower. Once you've got both files, update `bpe_path` and `model_path` in `configs/auto_labeling/segment_anything_3_video.yaml` to point to where you saved them.

You can tweak the settings in [segment_anything_3_video.yaml](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/configs/auto_labeling/segment_anything_3_video.yaml) to fit your needs. By default, detection boxes are shown, which works fine for most cases. If you want to see the actual masks instead, flip `show_masks` to `true`. Just keep in mind that rendering masks can slow things down a bit, so boxes are usually the safer bet.

## Usage 

Launch the X-AnyLabeling client, press `Ctrl+A` or click the `AI` button in the left menu bar to open the auto-labeling panel. In the model dropdown list, select `Remote-Server`, then choose `Segment Anything 3 Video`.

### Text Prompting

<video src="https://github.com/user-attachments/assets/4a676ebf-d2ae-4327-b078-8e63a5323793" width="100%" controls>
</video>

1. Enter object names in the text field (e.g., `person`, `car`, `bicycle`)
2. Click **Send** to initiate detection
3. After verification, click the `Auto Run` button in the left menu bar or press `Ctrl+M` to start forward propagation

> [!NOTE]
> Each session supports one category only. Propagation starts from the current frame and runs to the end of the video, with a brief warm-up period of approximately 15 frames before processing begins.

> [!WARNING]
> If you cancel the propagation task midway, all processed frame results will be lost. Please wait for the task to complete or ensure you have saved the results before canceling.

### Visual Prompting

<video src="https://github.com/user-attachments/assets/74ad72ea-0207-4ec5-95e7-9e6de8cb6aac" width="100%" controls>
</video>

With visual prompting, you can use points to mark what you want to track. 

1. Click on Point(q) to add positive sample points (things you want to include) or Point(e) for negative sample points (things you want to exclude). You can add multiple points to refine the selection. 
2. When you're happy with your points, hit Finish(f) to complete the drawing. This will pre-fill the target label name and track_id for you. If you mess up and want to start over, just press Clear(b) to remove everything.
3. Once you've confirmed the selection, you can start the forward propagation by clicking the `Auto Run` button in the left menu bar or pressing `Ctrl+M`. 

> [!TIP]
> The letters in parentheses are keyboard shortcuts, by the way - you can use `q` and `e` to quickly switch between positive and negative point modes, `f` to finish, and `b` to clear.

> [!NOTE]
> A few things to keep in mind: you can start tracking from any frame in the video, but each session only supports tracking a single target. Also, visual prompting uses a non-overwrite mode by default, so it won't replace existing annotations.

---

We've also included a couple of handy tools to make your life easier. The `Label Manager` and `Group ID Manager` let you quickly modify label names or track_ids either locally or globally. Check out the demo video below to see how they work.

<video src="https://github.com/user-attachments/assets/b85b661c-319c-4de3-96c0-833c5a53c01c" width="100%" controls>
</video>

See [User Guide](../../../docs/en/user_guide.md) for more details.