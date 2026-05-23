# Overview

The Video Classifier in X-AnyLabeling is a dedicated annotation window for action recognition and video clip classification datasets. It lets you load videos, define class labels, mark time segments on a timeline, assign each segment to a label, preview frames, and export class-organized video clips or raw frame sequences.

<video src="https://github.com/user-attachments/assets/33a57390-683d-4a24-b0cf-3668af5f7a13" width="100%" controls>
</video>

# Getting Started

Open the Video Classifier from the main window toolbar or use the configured shortcut:

- Windows/Linux: `Ctrl` + `5`
- macOS: `⌘` + `5`

You can load a video by dropping it into the dialog or by clicking the open-video button in the preview footer. Supported formats include common video files such as `mp4`, `avi`, `mov`, `mkv`, and `webm`.

The help icon in the preview header opens this tutorial on GitHub. The shortcuts icon beside it opens an in-app shortcut reference.

# Tutorial

The Video Classifier uses three main areas:

- The left preview panel shows the current video frame, playback controls, AI auto segmentation, frame saving, zoom controls, shortcut help, tutorial help, and dataset export access.
- The right panel manages labels, the segment list, and segment descriptions.
- The bottom timeline shows thumbnails, time ticks, the playhead, and editable labeled segments.

## Labels

Before you start labeling, set up the classes for the task in the Labels panel on the right. Each label gets its own color, and the same color is used later in the timeline and segment list, making it easier to scan long videos and understand how different classes are distributed.

Use the add button to create a new label. To rename labels, change colors, import a label list, or remove labels, open the label settings. The first ten labels show `(0)` through `(9)` after the name, so you can switch the active label with the matching number key while annotating.

New segments use the currently selected label. If you delete a label, any segments already assigned to that label are deleted as well, so only do this when those segments are no longer needed.

> [!TIP]
> For tasks with many classes, prepare a `classes.txt` file with one class name per line. Open the label settings from the Labels panel on the right, then click upload to import the full list at once.

## Creating Segments

Segments are created from the timeline ruler.

1. Select the target label in the Labels panel.
2. Move the mouse to the timeline ruler.
3. Hold the right mouse button, drag across the desired time range, and release.
4. The segment is added using the selected label.

If no label is selected, the dialog prompts you to select or create one first.

You can also press `I` and `O` to mark an in/out range, then press `Enter` to create a segment from those marks.

## AI Configuration

AI auto segmentation and segment description generation in the Video Classifier use the model configured in [Chatbot](../en/chatbot.md). Before using these features, open Chatbot first (Windows/Linux: `Ctrl` + `1`, macOS: `⌘` + `1`) and configure the API key, provider, endpoint, and current model in the right panel.

For example, when using Alibaba Cloud Model Studio with `qwen3.6-flash`, select the Qwen provider in Chatbot, enter your Model Studio API key, and set the current model to `qwen3.6-flash`. If you connect through the Custom provider with an OpenAI-compatible endpoint, make sure the endpoint matches the provider requirements. For Model Studio compatible mode, the endpoint is usually:

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

> [!NOTE]
> Make sure the selected model supports video understanding. Video file size, duration, frame rate, input format, and billing limits may vary by provider and model. For Alibaba Cloud Model Studio, see the [Image and Video Understanding](https://help.aliyun.com/zh/model-studio/vision?userCode=okjhlpr5#006b533c6cc0) documentation for video-related limits.

## AI Auto Segmentation

After configuring a video-capable model in Chatbot, click the wand button in the preview footer to let AI generate segments from the video content. The dialog shows the current model and provides an editable prompt. When the interface language is Chinese, the default prompt is Chinese; other languages use the English prompt by default.

AI auto segmentation supports two input scopes:

- When a segment is selected, only the selected segment is sent to the model by default, and the returned segments replace that selected segment.
- When `Use full video` is checked, the full video is sent to the model. If the current video already has segments, the dialog asks for confirmation before replacing them with AI-generated segments.

The model should return JSON, for example:

```json
{
  "events": [
    {
      "start_time": "00:00:00",
      "end_time": "00:00:05",
      "event": "The person walks toward the table holding a box and places the box on the table."
    }
  ]
}
```

`start_time` and `end_time` should use the `HH:mm:ss` format, and `event` is saved as the segment description. If a label is currently selected, generated segments use that label; otherwise, built-in label names such as `片段1` and `片段2` are assigned automatically.

## Editing Segments

Existing segments can be edited directly on the timeline or from the segment list.

| Action | Result |
|--------|--------|
| Single-click a segment | Select it and seek to its start |
| Double-click a segment | Edit its label and description |
| Hold left mouse button and drag a segment | Move the segment |
| Drag the left or right edge | Resize the segment |
| Delete/Backspace | Delete the selected segment |
| Scissors button | Split the selected segment at the playhead when possible |
| Undo/Redo | Revert or reapply segment edits |

The Segments list mirrors the timeline. Selecting a row jumps the playback frame to that segment's start time.

When the cursor is over a segment, the inline hint reminds you that double-click edits the label and description, and left-drag moves the segment.

After selecting a segment, you can edit its description in the Description panel on the right. The wand button in the Description header calls the AI model configured in Chatbot to generate a description for the current segment. Before generation starts, a prompt editor is shown so you can review the current model and adjust the prompt. You can cancel the wait while generation is running. This feature only updates the selected segment description; it does not change the segment time range or label.

## Playback And Navigation

Use the footer controls or keyboard shortcuts to review precise frames. The same shortcut table is available from the shortcuts icon in the preview header.

| Shortcut | Function |
|----------|----------|
| Space | Play or pause |
| Left/Right | Step one frame backward or forward |
| Shift + Left/Right | Jump one second backward or forward |
| `,`/`.` | Jump to the previous or next segment |
| I/O | Mark in/out |
| Enter | Create a segment from the in/out marks |
| Delete/Backspace | Delete the selected segment |
| Ctrl+S or ⌘+S | Save the sidecar annotation file |
| Ctrl+Z or ⌘+Z | Undo |
| Ctrl+Shift+Z or ⌘+Shift+Z | Redo |
| 0-9 | Select the matching label number |

The timeline can be zoomed with its zoom controls or with `Ctrl` + mouse wheel over the timeline. Use the horizontal scrollbar to inspect long videos.

Move the cursor to the timeline ruler to see the right-drag segment creation hint. The hint is only shown on timeline interaction areas, not over the thumbnail strip.

## Export

After labeling segments, click Export dataset in the preview footer to export the current video's annotations. The exported files are organized by class, making them easier to inspect, package, or connect to your own training pipeline.

The export dialog includes these options:

| Option | Description |
|--------|-------------|
| `Output` | Choose the export directory. By default, X-AnyLabeling creates a folder with the same basename as the current video next to the video. For example, `sample.mp4` exports to `sample/` by default. |
| `Video clips` | Export each segment as an `.mp4` file under its label folder. |
| `Raw frame sequences` | Export each segment as an `img_00001.jpg` image sequence. |
| `RawFrames FPS` | Set the frame sampling rate. Use `0` to keep the source video frame rate. |
| `Re-encode clips` | Re-encode segments. This is slower, but gives more accurate start and end cuts; when disabled, X-AnyLabeling uses faster stream copy whenever possible. |
| `Pack output as .zip` | Create a zip archive beside the output directory after export. |

The output directory looks like this:

```bash
sample/
├── videos/
│   ├── label_a/
│   │   ├── sample_s000001.mp4
│   │   └── sample_s000002.mp4
│   └── label_b/
│       └── sample_s000003.mp4
├── rawframes/
│   ├── label_a/
│   │   └── sample_s000001/
│   │       ├── img_00001.jpg
│   │       └── img_00002.jpg
│   └── label_b/
│       └── sample_s000003/
│           ├── img_00001.jpg
│           └── img_00002.jpg
├── label_map.txt
└── metadata.json
```

Video clipping and frame extraction are handled by `ffmpeg`. X-AnyLabeling first looks for a system `ffmpeg`; if it is not available, it tries the executable provided by `imageio-ffmpeg`. If neither is available, export cannot continue.

# Data Format

Video annotations are stored beside each video as a sidecar JSON file with the same stem as the video. For example, `sample.mp4` is paired with `sample.json`.

```json
{
  "version": "1.0.0",
  "type": "video_classification",
  "video": "sample.mp4",
  "fps": 30.0,
  "duration_ms": 120000,
  "width": 1920,
  "height": 1080,
  "labels": ["run", "walk"],
  "label_colors": {
    "run": "#ff7f0e",
    "walk": "#1f77b4"
  },
  "segments": [
    {
      "id": "s1234567890",
      "label": "run",
      "start_ms": 1000,
      "end_ms": 3500,
      "start_frame": 30,
      "end_frame": 105,
      "description": "The person starts running after the whistle."
    }
  ]
}
```
