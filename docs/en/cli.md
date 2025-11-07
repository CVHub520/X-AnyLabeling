# Command Line Interface

X-AnyLabeling provides a powerful command-line interface for label format conversion and system management.

> [!NOTE]
> Before using these commands, ensure all dependencies are installed. See the [Quick Start Guide](./get_started.md) for details.

## 0. Table of Contents

- [1. GUI Launch Options](#1-gui-launch-options)
- [2. System Commands](#2-system-commands)
    - [2.1 Display Help Information](#21-display-help-information)
    - [2.2 Display System and Package Information](#22-display-system-and-package-information)
    - [2.3 Display Application Version](#23-display-application-version)
    - [2.4 Display Configuration File Path](#24-display-configuration-file-path)
- [3. Label Format Conversion](#3-label-format-conversion)
    - [3.1 List All Available Conversion Tasks](#31-list-all-available-conversion-tasks)
    - [3.2 Show Detailed Help for a Specific Task](#32-show-detailed-help-for-a-specific-task)
    - [3.3 Run a Conversion Task](#33-run-a-conversion-task)
    - [3.4 Conversion Parameters](#34-conversion-parameters)
- [4. FAQ](#4-faq)

## 1. GUI Launch Options

```bash
# Standard launch
xanylabeling

# Open a specific image file
xanylabeling --filename /path/to/image.jpg

# Open an image directory
xanylabeling --filename /path/to/folder

# Set output directory
xanylabeling --output /path/to/output

# Use a custom configuration file
xanylabeling --config /path/to/config.yaml

# Set logging level
xanylabeling --logger-level debug

# Disable automatic update check
xanylabeling --no-auto-update-check
```

## 2. System Commands

### 2.1 Display Help Information

- Input

```bash
xanylabeling --help
```

- Output

```bash
usage: xanylabeling [-h] [--reset-config] [--logger-level {debug,info,warning,fatal,error}] [--no-auto-update-check] [--qt-platform QT_PLATFORM] [--filename [FILENAME]] [--output OUTPUT]
                    [--config CONFIG] [--nodata] [--autosave] [--nosortlabels] [--flags FLAGS] [--labelflags LABEL_FLAGS] [--labels LABELS] [--validatelabel {exact}] [--keep-prev]
                    [--epsilon EPSILON]
                    {help,checks,version,config,convert} ...

positional arguments:
  {help,checks,version,config,convert}
                        available commands
    help                show help message
    checks              display system and package information
    version             show version information
    config              show config file path
    convert             run conversion tasks

options:
  -h, --help            show this help message and exit
  --reset-config        reset qt config
  --logger-level {debug,info,warning,fatal,error}
                        logger level
  --no-auto-update-check
                        disable automatic update check on startup
  --qt-platform QT_PLATFORM
                        Force Qt platform plugin (e.g., 'xcb', 'wayland'). If not specified, Qt will auto-detect the platform.
  --filename [FILENAME]
                        image or label filename; If a directory path is passed in, the folder will be loaded automatically
  --output OUTPUT, -O OUTPUT, -o OUTPUT
                        output file or directory (if it ends with .json it is recognized as file, else as directory)
  --config CONFIG       config file or yaml-format string (default: /home/cvhub/.xanylabelingrc)
  --nodata              stop storing image data to JSON file
  --autosave            auto save
  --nosortlabels        stop sorting labels
  --flags FLAGS         comma separated list of flags OR file containing flags
  --labelflags LABEL_FLAGS
                        yaml string of label specific flags OR file containing json string of label specific flags (ex. {person-\d+: [male, tall], dog-\d+: [black, brown, white], .*:
                        [occluded]})
  --labels LABELS       comma separated list of labels OR file containing labels
  --validatelabel {exact}
                        label validation types
  --keep-prev           keep annotation of previous frame
  --epsilon EPSILON     epsilon to find nearest vertex on canvas
```

### 2.2 Display System and Package Information

- Input

```bash
xanylabeling checks
```

- Output

```bash
Application
────────────────────────────────────────────────────────────
  App Name:          X-AnyLabeling
  App Version:       3.3.0
  Preferred Device:  CPU
────────────────────────────────────────────────────────────
System
────────────────────────────────────────────────────────────
  Operating System:  Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31
  CPU:               x86_64
  CPU Count:         20
  RAM:               31.2 GB
  Disk:              841.6/1006.9 GB
  GPU:               CUDA:0 (NVIDIA GeForce RTX 3060, 12288MiB)
  CUDA:              V11.6.124
  Python Version:    3.10.10
────────────────────────────────────────────────────────────
Packages
────────────────────────────────────────────────────────────
  PyQt5 Version:                           5.15.11
  ONNX Version:                            1.19.1
  ONNX Runtime Version:                    1.23.2
  ONNX Runtime GPU Version:                None
  OpenCV Contrib Python Headless Version:  4.11.0.86
────────────────────────────────────────────────────────────
```

### 2.3 Display Application Version

- Input

```bash
xanylabeling version
```

- Output

```bash
3.3.0
```

### 2.4 Display Configuration File Path

- Input

```bash
xanylabeling config
```

- Output

```bash
~/.xanylabelingrc
```

## 3. Label Format Conversion

### 3.1 List All Available Conversion Tasks

- Input

```bash
xanylabeling convert
```

- Output

```bash
================================================================================
SUPPORTED CONVERSION TASKS
================================================================================

📥 IMPORT TO XLABEL
--------------------------------------------------------------------------------
  • yolo2xlabel [detect, segment, obb, pose]
  • voc2xlabel [detect, segment]
  • coco2xlabel [detect, segment, pose]
  • dota2xlabel
  • mot2xlabel
  • ppocr2xlabel [rec, kie]
  • mask2xlabel
  • vlmr12xlabel
  • odvg2xlabel

📤 EXPORT FROM XLABEL
--------------------------------------------------------------------------------
  • xlabel2yolo [detect, segment, obb, pose]
  • xlabel2voc [detect, segment]
  • xlabel2coco [detect, segment, pose]
  • xlabel2dota
  • xlabel2mask
  • xlabel2mot
  • xlabel2mots
  • xlabel2odvg
  • xlabel2vlmr1
  • xlabel2ppocr [rec, kie]

================================================================================
Total: 19 conversion tasks
================================================================================

Usage:
  xanylabeling convert                          # Show all tasks
  xanylabeling convert --task <task>            # Show detailed help for a task
  xanylabeling convert --task <task> [options]  # Run conversion
```

### 3.2 Show Detailed Help for a Specific Task

- Input (using `yolo2xlabel` as an example)

```bash
xanylabeling convert --task yolo2xlabel
```

- Output

```bash
================================================================================
TASK: yolo2xlabel
================================================================================

Description:
  Convert YOLO format to XLABEL

Modes:
  detect, segment, obb, pose

Required Arguments:
  --images
  --labels
  --output

Mode-Specific Arguments:
  detect: --classes
  segment: --classes
  obb: --classes
  pose: --pose_cfg

Examples:
  # Detection
  xanylabeling convert --task yolo2xlabel --mode detect --images ./images --labels ./labels \
    --output ./output --classes classes.txt

  # Segmentation
  xanylabeling convert --task yolo2xlabel --mode segment --images ./images --labels ./labels \
    --output ./output --classes classes.txt

  # OBB (Oriented Bounding Box)
  xanylabeling convert --task yolo2xlabel --mode obb --images ./images --labels ./labels \
    --output ./output --classes classes.txt

  # Pose
  xanylabeling convert --task yolo2xlabel --mode pose --images ./images --labels ./labels \
    --output ./output --pose-cfg pose_config.yaml

================================================================================
```

### 3.3 Run a Conversion Task

- Input (using `yolo2xlabel:detect` as an example)

```bash
xanylabeling convert --task yolo2xlabel --mode detect --images ./images --labels ./labels \
    --output ./output --classes classes.txt
```

- Output

```bash
Converting YOLO detect to XLABEL: 100%|█████████████████████████████████████| 128/128 [00:00<00:00, 2456.47it/s]
✓ Converted 128 files to XLABEL format: ./output
```

### 3.4 Conversion Parameters

| Parameter             | Description                                                          | Required |
|-----------------------|----------------------------------------------------------------------|----------|
| `--task`              | Conversion task name                                                 | Yes      |
| `--images`            | Path to the image directory                                          | No       |
| `--labels`            | Path to the label directory                                          | No       |
| `--output`            | Path to the output directory                                         | Yes      |
| `--classes`           | Path to the classes file                                             | No       |
| `--pose-cfg`          | Path to the pose configuration file                                  | No       |
| `--mode`              | Conversion mode (e.g., detect, segment, obb, pose)                   | No       |
| `--mapping`           | Path to the mapping table file                                       | No       |
| `--skip-empty-files`  | Skip creating empty output files (supported by xlabel2yolo and xlabel2voc only) | No |

> For more details, refer to the [User Guide - Label Import/Export](./user_guide.md#4-label-importexport) section.

## 4. FAQ

### Q1: What is the XLABEL format?

XLABEL is X-AnyLabeling's native JSON format. It stores all annotation information in a human-readable format, including shape data, labels, attributes, and metadata.

### Q2: Do I need to provide class names for all conversions?

Class names are required for:
- YOLO conversions (detect, segment, obb modes)
- COCO conversions (detect, segment modes)
- MOT conversions

VOC format embeds class names directly in the XML files, so a separate classes file is not required.

### Q3: How do I create a classes.txt file?

Simply create a text file with one class name per line:

```
person
car
bicycle
dog
cat
```

The line number (0-indexed) corresponds to the class ID.

### Q4: What's the difference between MOT and MOTS?

- **MOT**: Multiple Object Tracking with bounding boxes
- **MOTS**: Multiple Object Tracking with segmentation masks

### Q5: Can I convert directly between formats without using XLABEL?

Not currently. All conversions use XLABEL as an intermediate format:
1. Convert to XLABEL first
2. Then convert from XLABEL to the target format

### Q6: What if my images are in subdirectories?

The converter currently processes images in the specified directory only. For nested directory structures, you'll need to either run the conversion multiple times or flatten your directory structure.

### Q7: How do I handle non-ASCII paths on Windows?

The converter has built-in support for non-ASCII paths. Ensure your terminal encoding is set to UTF-8:

```bash
chcp 65001  # For Windows CMD
```

### Q8: What does the `--skip-empty-files` option do?

This option (supported by `xlabel2yolo` and `xlabel2voc`) prevents the creation of empty label files for images without annotations. This is useful when you need to distinguish between "unlabeled" and "labeled but empty" images.

### Q9: Can I use relative paths?

Yes, both relative and absolute paths are supported. Relative paths are resolved from your current working directory.

### Q10: How do I convert a single file?

Place your file in a directory and run the conversion on that directory. The converter processes all matching files in the specified directory.
