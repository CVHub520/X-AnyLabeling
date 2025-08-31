# X-AnyLabeling Changelog

## `v3.2.2` (Aug 31, 2025)

### 🚀 New Features

- Support batch editing for multiple shapes (#1084)
- Introduce AI Assistant for VQA
- Add prompt template management to VQA
- Enhance VQA dialog with a new UI including sidebar toggles, streamlined navigation controls, improved page navigation, loading indicators, and updated button styles

### 🐛 Bug Fixes

- Fix issue with dragging and moving the image (#1088)

### 🛠️ Improvements

- Optimize SAM inference memory management (#1086)

### 🌟 Contributors

A total of 4 developers contributed to this release.

Thank @jsolobang, @zhaoruibing, @zhixuwei, @CVHub520

## `v3.2.1` (Aug 23, 2025)

### 🚀 New Features

- Add support for showing/hiding shape attributes on the canvas (#1076)
- Add functionality to save training logs with timestamp upon dialog closure (#1077)

### 🐛 Bug Fixes

- Skip validation for auto-labeling special constants
- Prevent closing UltralyticsDialog during active training session (#1077)
- Improve WSL2 detection for image file handling in UltralyticsDialog (#1077)
- Add UTF-8 encoding to file opening in validate_data_file function (#1077)
- Resolve Windows multiprocessing and matplotlib segfault issues

### 🌟 Contributors

A total of 2 developer contributed to this release.

Thank @FreemanTang, @CVHub520

## `v3.2.0` (Aug 19, 2025)

### 🚀 New Features

- Introduce Auto Training Platform for Ultralytics tasks (Detect/Segment/OBB/Pose)

### 🌟 Contributors

A total of 1 developers contributed to this release.

Thank @CVHub520

## `v3.1.2` (Aug 16, 2025)

### 🚀 New Features

- Introduce Auto Mask Decode (AMD) mode for continuous point tracking (#1060)
- Add new RF-DETR models (medium, small, nano) and fix input width typo in configuration files (#1069)
- Enhance range selection for group ID modification with new input fields and validation (#1035)
- Add support for MM-Grounding-DINO annotations upload

### 🐛 Bug Fixes

- Update error message for label validation to specify 'exact' in config file (#1064)
- Fix issues when drawing rectangular boxes (#1063 by zhixuwei)
- Add try-except for mask_to_polygons for supervision version compatibility (#1055 by adarshs)
- Improve segmentation handling by filtering invalid entries and avoiding duplicate points in polygon mode (#1032)
- Resolve KeyError when importing files via drag and drop (#1030)
- Enhance image saving logic to handle non-ASCII paths and improve multiprocessing handling in frozen environments (#1021)
- Improve crop region validation and handle empty cropped images with warnings (#1021)
- Emit model_loaded signal even if loading custom model configuration fails
- Update frame ID extraction logic to handle underscores and non-digit cases (#1020)

### 🛠️ Improvements

- Correct variable name from 'has_vasiable' to 'has_visible' for accurate keypoint processing
- Simplify toggle button text for clarity in label and shape information display
- When looping through shapes, display their fill colors (#1025 by zhixuwei)

### 🌟 Contributors

A total of 3 developers contributed to this release.

Thank @zhixuwei, @adarshs, @CVHub520

## `v3.1.1` (Jul 05, 2025)

### 🚀 New Features

- Add customizable field export options for VQA dialog
- Add ability to adjust the visible area of the image by dragging the mouse (#1019)

### 🐛 Bug Fixes

- Fix VQA keyboard shortcut (Ctrl+Q → Ctrl+1)

### 🌟 Contributors

A total of 2 developer contributed to this release.

Thank @zhixuwei, @CVHub520

## `v3.1.0` (Jul 02, 2025)

### 🚀 New Features

- Support `RMBG v2.0` model for image matting
- Add output_path parameter to COCO label converter methods for custom output paths
- Add real-time result preview for matting and depth estimation tasks
- Add GUI support for uploading custom label classes (#988)
- Add rectangle scaling and edge adjustment with mouse wheel support (#989)
- Add automatic update check on startup
- Add Visual Question Answering tool

### 🐛 Bug Fixes

- Improve error handling and logging for annotation export and upload processes (#974)
- Fix annotation_id increment in COCO data processing (#976)
- Fix failure to click again after custom model loading
- Fix scrollbar slider display issue
- Fix issue where copied shapes fail to be saved
- Fix auto-save bug after undo operations when switching images (#1013)

### 🛠️ Improvements

- Add solution to CUDA dependency error: `Could not locate cublasLt64_12.dll. Please make sure it is in your library path!` (#1014)
- Add solution to efficiency improvement plan for multi-object keypoint annotation and grouping (#982)
- Add CLA, contributing templates, and README contributor section
- Improve QWebEngineView import error handling in chatbot
- Improve thumbnail rendering by mapping file extensions to model types in auto-labeling service
- Improve shape adjustment convenience
- Improve click-to-move editing with state cleanup and cursor feedback

### 🌟 Contributors

A total of 7 developer contributed to this release.

Thank @1955946542, @donkinone, @ljh725, @pipihuang2, @sunmooncode, @zhixuwei, @CVHub520

## `v3.0.3` (May 28, 2025)

### 🚀 New Features

- Added Digit Shortcut Manager for quick shape creation using numeric keys (#945)
- Added support for preserving existing annotations in SegmentAnything2Video model

### 🛠️ Improvements

- Updated FAQ entries for ONNX model IR version issues with resolution steps
- Added FAQ entry on accessing external models like 'Google Gemini' through proxy settings
- Converted BrightnessContrastDialog to singleton to reduce instantiation overhead (#954)

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520

## `v3.0.2` (May 24, 2025)

### 🚀 New Features

- Support `YOLOE` model for text/visual prompting and prompt-free grounding

### 🐛 Bug Fixes

- Fixed crash issue during keypoint annotation (#952)

### 🌟 Contributors

A total of 1 developers contributed to this release.

Thank @CVHub520

## `v3.0.1` (May 20, 2025)

### 🚀 New Features

- Add support for Ultralytics RT-DETR object detection models (#944)
- Enhance label history management by adding removal and addition methods for label history

### 🐛 Bug Fixes

- Add 'pillow' to requirements for macOS, Windows, and Linux (#942)
- Initialize image_data before base64 encoding in save_auto_labeling_result (#946)

### 🛠️ Improvements

- Optimize image folder import performance by disabling automatic EXIF processing (#945)
- Update requirements-macos.txt (#939)
- Add FAQ entry for OpenSSL Uplink error during startup (#941)

### 🌟 Contributors

A total of 3 developers contributed to this release.

Thank @DenDen047, @4399123, @CVHub520


## `v3.0.0` (May 15, 2025)

### 🚀 New Features

- Add ffmpeg acceleration and non-ASCII path support (#891)
- Allow downloading models from [ModelScope](https://www.modelscope.cn/collections/X-AnyLabeling-7b0e1798bcda43) in addition to existing sources
- Enable one-click import and export of labels for [VLM-R1-OVD](https://github.com/om-ai-lab/VLM-R1)
- Enable the [Chatbot](./docs/en/chatbot.md) to annotate multimodal datasets for Vision-Language Models (VLMs)
- Enable automatic saving of group IDs when grouping or ungrouping shapes using shortcut keys G (group) and U (ungroup). (#855)
- Introduce GroupID filter and improve label filtering functionality (#686)
- Support [GeCo](./examples/counting/geco/README.md) zero-shot counting model (#863)
- Support [Grounding-DINO-1.6-API](https://algos.deepdataspace.com/en#/model/grounding_dino) open-set object detection model
- Support [YOLO12](https://arxiv.org/abs/2502.12524) object detection model
- Support [D-FINE](./tools/onnx_exporter/export_dfine_onnx.py) object detection model
- Support [RF-DETR](./tools/onnx_exporter/export_rfdetr_onnx.py) object detection model

### 🐛 Bug Fixes

- Fix bug in `predict_shapes` for `Florence-2` model (#913)
- Replace `os_sorted` with `natsorted` to avoid potential segfault (#906)
- Merge multi-part segmentations into single instance on export (#910)
- Handle exceptions in model loading by initializing local_model_data to an empty dictionary for improved stability (#901)
- Prevent UI disappearance when ESC key is pressed during AI annotation (#423)
- Fixed the bug about exporting the empty labels (#881)

### 🛠️ Improvements

- Move image conversion to avoid redundant processing when using cached embeddings (#915)
- Add new FAQs addressing common runtime errors and file loading issues, including solutions and references to related GitHub issues (#869, #906, #907)
- Enhance image processing logic to support dynamic batch handling based on model type, improving efficiency in auto-labeling operations
- Introduce `iou_threshold` and `conf_threshold` parameters across various model configurations for enhanced detection accuracy
- Remove imgviz dependency from requirements and update colormap implementation in labeling utilities for improved modularity
- Optimize batch processing with UI/backend separation (#757)
- Enhance shape visibility handling in labeling interface (#669)
- Updated the merge_shapes method to handle both rectangle and polygon shapes, allowing for more versatile shape unions. (#561)

### 🌟 Contributors

A total of 8 developers contributed to this release.

Thank @Pecako2001, @liutao, @shyhyawJou, @talebolano, @urbaneman, @wangxiang0722, @Little-King2022, @CVHub520


## `v2.5.4` (Feb 18, 2025)

### 🚀 New Features

- Support `YOLOv8-SAM2.1` instance segmentation model

### 🐛 Bug Fixes

- Improve canvas painting state management during image processing
- Reset `pose_data` for each new image in pose mode (#791)

### 🛠️ Improvements

- Add guide for selecting all annotation shapes (#759)
- Remove `YOLOv8-EfficientViT-SAM` model support

### 🌟 Contributors

A total of 2 developers contributed to this release.

Thank @aiyou9, @CVHub520


## `v2.5.3` (Jan 12, 2025)

### 🐛 Bug Fixes

- Fix loading mistake in batch processing (#777)
- Reset `pose_data` for each new image in pose mode (#791)

### 🛠️ Improvements

- Optimize performance for large files (#743)

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.5.2` (Jan 02, 2025)

### 🚀 New Features

- Enhance export functionality with `classes.txt` and zip output (#775)

### 🐛 Bug Fixes

- Fix image dimension validation errors (#762)

### 🛠️ Improvements

- Update downloads badges in `README`

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.5.1` (Jan 01, 2024)

### 🚀 New Features

- Optimize inference efficiency during batch task execution
- Support `Hyper-YOLO`
- Add CPU support and optimize model loading for `Florence-2`

### 🐛 Bug Fixes

- Fix GBK codec decoding error in `RAM` model loading process
- Fix dtype casting in `RAM` model preprocessing
- Fix `text_encoder_type` path in `open_vision.yaml`

### 🛠️ Improvements

- Improve visibility in dark mode for MacOS
- Update `OpenVision` `README` with new installation instructions

### 🌟 Contributors

A total of 2 developers contributed to this release.

Thank @chevydream, @CVHub520


## `v2.5.0` (Oct 15, 2024)

### 🚀 New Features

- Support interactive visual-text prompting for generic vision tasks
- Optimize rectangle mode in auto-labeling to use minimum bounding box
- Support `SAM2.1` model
- Support `Florence-2` model (#679)
- Support `UPN` model for proposal box generation
- Support `YOLOv5-SAHI` model
- Add range selection for label batch modification (#708)
- Add options dialog with additional export path selection (#702)
- Support importing/exporting COCO keypoint annotations (#190)
- Support `DocLayout-YOLO` model
- Add option to color bounding boxes by category or instance
- Add action to loop through each label

### 🐛 Bug Fixes

- Handle invalid file paths in natural sort during import (#734)
- Improve mask overlapping handling in `custom_to_mask` method during export
- Fix path parsing error of `save_crop` function
- Disable delete action when no shapes present
- Fix image normalization in `Recognize-Anything-Model` preprocessing (#657)

### 🛠️ Improvements

- Add ONNX Runtime compatibility information to installation docs
- Modernize `GroupIDModifyDialog` with improved styling

### 🌟 Contributors

A total of 3 developers contributed to this release.

Thank @julianstirling, @CVHub520, @wpNZC


## `v2.4.4` (Sep 30, 2024)

### 🚀 New Features

- Support `YOLOv11` Det/OBB/Pose/Seg/Track models (integrating Ultralytics v8.3.0)

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.4.3` (Sep 08, 2024)

### 🚀 New Features

- Support `RMBG v1.4` model for image matting

### 🐛 Bug Fixes

- Ensure integer values for shape dimensions in `show_shape` signal
- Fix model loading error for `YOLOv6lite` face models (#638)

### 🛠️ Improvements

- Enhance logging with bold and colored headers
- Modify indexing operations to improve file navigation efficiency
- Improve EXIF orientation handling with backup and logging
- Implement natural sorting for `QListWidget` labels (#627)
- Support user-defined labels and track IDs (#629)

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.4.2` (Sep 06, 2024)

### 🚀 New Features

- Support interactive video object tracking by `SAM2` (#602)
- Implement functionality to visualize drawing results

### 🐛 Bug Fixes

- Fix typo in `upload_coco_annotation` function

### 🛠️ Improvements

- Add vscode configuration files for module debugging and profiling

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.4.1` (Aug 29, 2024)

### 🚀 New Features

- Add dialog for modifying `group_id`
- Support exporting MOTS annotations

### 🐛 Bug Fixes

- Fix patch memory leak in image caching during image transitions
- Retain labels during switch model instances

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.4.0` (Jul 14, 2024)

### 🚀 New Features

- Implement masked image saving functionality
- Support tracking by HBB/OBB/SEG/POSE task
- Support `GroundingSAM2` model
- Support lightweight model for Japanese recognition
- Support `Segment-Anything-2` model
- Enable `move_mode` parameter in `label_dialog`
- Add `use_system_clipboard` action
- Enable import/export of ODVG annotations (`Grounding DINO` dataset)
- Support `RT-DETRv2` model
- Support `RAM++` and `YOLOW-RAM++` models
- Implement feature to draw KIE linking lines with arrowheads
- Support displaying and exporting shape-level information
- Enable `Depth-Anything` model prediction (color/grayscale)
- Add import/export functionality for PPOCR-KIE annotations
- Support annotating KIE linking field
- Support annotating out-of-pixmap rotation shapes
- Support `depth-anything-v2` model
- Add import/export functionality for PPOCR label
- Add toggle for continuous drawing mode
- Support union of multiple selected rectangle shapes
- Add system clipboard copy mode
- Add ability to delete label items based on checkbox selection
- Enable opening previous/next labeled image
- Implement crosshair and marking box style customization
- Add widget for converting polygon to HBB
- Add `YOLO-Pose` import/export functionality
- Enable exporting VOC-format annotations for polygon shape
- Add preserve existing annotations checkbox and real-time confidence adjustment
- Add visibility feature for keypoint detection task
- Support `YOLOv8-World` and `YOLOv8-OIV7` models
- Add feature to display confidence score

### 🐛 Bug Fixes

- Fix image distortion issue during brightness/contrast adjustment
- Fix `TypeError` in `fillRect` for higher Python versions
- Fix `too many values to unpack` error during YOLO class post-process
- Fix `invalid literal for int()` with base issue
- Avoid `directory not empty` error when loading model
- Prevent crash when switching from image directory to imported image
- Fix BMP image loading issue (missing `_getexif` attribute)

### 🛠️ Improvements

- Refresh X-Anything app icon
- Implement structured `ISSUE_TEMPLATE`
- Add `SECURITY.md` file
- Optimize code related to "actions"
- Add extensive documentation examples for various tasks (OCR, MOT, Pose, Segmentation, Detection, Depth, Description, Classification)
- Add `faq.md` file

### 🌟 Contributors

A total of 3 developers contributed to this release.

Thank @UnlimitedWand, @PairZhu, @CVHub520


## `v2.3.7` (May 29, 2024) - *Pre-release*

### 🚀 New Features

- Support `YOLOv8-World` and `YOLOv8-OIV7` models

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.3.6` (May 25, 2024)

### 🐛 Bug Fixes

- Fix `TypeError` in YOLO model (ensure `setText` accepts only strings)
- Fix `ValueError` by adding a check for selected shape existence (#388)
- Fix `list index out of range` when exporting DOTA annotations

### 🛠️ Improvements

- Optimize brightness and contrast adjustment performance

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.3.5` (Apr 01, 2024)

### 🚀 New Features

- Enhance image cropping: polygon and rotation `shape_type` support (#331)
- Add widget for converting OBB to HBB

### 🐛 Bug Fixes

- Fix `TypeError` in YOLO model (ensure `setText` accepts only strings)
- Fix `IndexError` in Canvas widget's shape module (#332)
- Fix invalid path issue when loading JSON file in Windows environment

### 🛠️ Improvements

- Refine canvas reset behavior to prevent unintended blank canvas
- Improve performance for loading large image files

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.3.4` (Mar 16, 2024)

### 🚀 New Features

- Support `YOLO-World` model
- Enable label display feature

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.3.3` (Feb 27, 2024)

### 🚀 New Features

- Add expanded sub-image save feature
- Support converting YOLO-HBB/OBB/SEG labels to custom format

### 🛠️ Improvements

- Add real-time progress bar for mask annotation uploads

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.3.2` (Feb 24, 2024)

### 🚀 New Features

- Support `YOLOv9` model
- Support converting horizontal bounding box (HBB) to rotated bounding box (OBB)
- Support label deletion and renaming
- Support quick tag correction

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.3.1` (Jan 31, 2024)

### 🚀 New Features

- Support saving cropped rectangle shapes
- Combine `CLIP` and `SAM` models
- Support `Depth Anything` model for depth estimation

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.3.0` (Jan 13, 2024)

### 🚀 New Features

- Support `YOLOv8-OBB` model
- Support `RTMDet` and `RTMO` models
- Release Chinese license plate detection and recognition model based on `YOLOv5`

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.2.0` (Dec 26, 2023)

### 🚀 New Features

- Add label background color rendering
- Add one-click clear point prompts in SAM annotation mode
- Extend rectangle box editing to four-point editing mode
- Support automatically switching to editing mode on hover
- Implement one-click export for segmented mask images
- Add one-click import/export for YOLO/VOC/COCO/DOTA/MASK/MOT labels
- Introduce data statistics for the current task
- Add functionality to hide/show selected objects
- Add real-time preview of filename and annotation progress
- Support "Difficult" label
- Add bottom status bar displaying mouse coordinates and selected object dimensions
- Enable direct modification of object text descriptions in the label editing box

### 🛠️ Improvements

- Remove confirmation dialog for object deletion

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.1.0` (Nov 24, 2023)

- Support `InternImage` classification model

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v2.0.0` (Nov 13, 2023)

### 🚀 New Features

- Support `Grounding-SAM` (`GroundingDINO` + `HQ-SAM`)
- Enhance support for `HQ-SAM` model
- Support `PersonAttribute` and `VehicleAttribute` models for multi-label classification
- Introduce multi-label attribute annotation functionality

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v1.1.0` (Nov 06, 2023)

### 🚀 New Features

- Support pose estimation: `YOLOv8-Pose` (#103)
- Support object-level tagging with `yolov5_ram`
- Add capability to adjust `keep_prev_brightness` and `keep_prev_contrast` (#104)
- Add feature for batch labeling arbitrary unknown categories based on `Grounding-DINO`

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v1.0.0` (Oct 25, 2023)

### 🚀 New Features

- Release X-AnyLabeling `v1.0.0`
- Add rotation box annotation feature
- Support `YOLOv5-OBB` with `DroneVehicle` and `DOTA` models
- Add `GroundingDINO` model for zero-shot object detection
- Add `Recognize Anything` model for image tagging

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.3.0` (Oct 10, 2023)

### 🚀 New Features

- Release `Gold-YOLO` and `DAMO-YOLO` models
- Release MOT algorithm: `OC_Sort` (CVPR'23)
- Add feature for small object detection using `SAHI`

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.2.4` (Sep 20, 2023)

### 🚀 New Features

- Support `EfficientViT-SAM` model

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.2.3` (Sep 18, 2023)

### 🚀 New Features

- Support `YOLOv5-SAM` model

### 🐛 Bug Fixes

- Fix issues #51, #60, #62

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.2.2` (Sep 14, 2023)

### 🚀 New Features

- Support `PP-OCRv4` model
- Add Model Lists

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.2.1` (Sep 06, 2023)

*No specific changes listed for this release tag.*

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.2.0` (Aug 09, 2023)

### 🚀 New Features

- Add `MobileSAM`, `MedSAM`, `SAM-Med2D` models
- Add `LVM-Med` model (Kvasir, ISIC, BUID segmentation)
- Add `CLRNet` model for lane detection
- Add `DWPose` model for whole-body pose estimation

### 🐛 Bug Fixes

- Fix `YOLOv8` issue
- Fix GPU inference result exception

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.1.2` (Jun 20, 2023)

### 🚀 New Features

- Add `YOLO-NAS` model (v3.1.1)
- Add `YOLOv8-Seg` model

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.1.1` (May 25, 2023)

*Update executable files.*

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520


## `v0.1.0` (May 23, 2023)

### 🚀 New Features

- Release initial public version of X-AnyLabeling
- Support `YOLOv5` (v7.0)
- Support `YOLOv6` (v0.4.0)
- Support `YOLOv6Face` (v0.4.0)
- Support `YOLOv7` (main)
- Support `YOLOv8` (main)
- Support `YOLOX` (main)
- Support `YOLO-NAS` (v3.1.1)
- Support `YOLOv8-Seg` (main)
- Support `Mobile-SAM`

### 🌟 Contributors

A total of 1 developer contributed to this release.

Thank @CVHub520
