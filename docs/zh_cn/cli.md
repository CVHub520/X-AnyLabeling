# 命令行界面 (CLI)

X-AnyLabeling 提供了强大的命令行界面，用于标签格式转换和系统管理任务。

> [!NOTE]
> 在使用本命令之前，你需要确保已安装对应的依赖，详情可参考[快速入门指南](./get_started.md)。

## 0. 目录

- [1. 图形界面启动选项](#1-图形界面启动选项)
- [2. 系统命令](#2-系统命令)
    - [2.1 显示帮助信息](#21-显示帮助信息)
    - [2.2 显示系统和软件包信息](#22-显示系统和软件包信息)
    - [2.3 显示应用版本信息](#23-显示应用版本信息)
    - [2.4 显示配置文件路径](#24-显示配置文件路径)
- [3. 标签格式转换命令](#3-标签格式转换命令)
    - [3.1 列出所有可用的转换任务](#31-列出所有可用的转换任务)
    - [3.2 显示特定任务的详细帮助](#32-显示特定任务的详细帮助)
    - [3.3 运行特定的转换任务](#33-运行特定的转换任务)
    - [3.4 转换参数说明](#34-转换参数说明)
- [4. 常见问题](#4-常见问题)

## 1. 图形界面启动选项

```bash
# 常规启动
xanylabeling

# 打开指定的图像文件
xanylabeling --filename /path/to/image.jpg

# 打开指定的图像文件夹
xanylabeling --filename /path/to/folder

# 设置输出目录
xanylabeling --output /path/to/output

# 使用自定义配置文件
xanylabeling --config /path/to/config.yaml

# 设置日志级别
xanylabeling --logger-level debug

# 禁用自动更新检查
xanylabeling --no-auto-update-check
```

## 2. 系统命令

### 2.1 显示帮助信息

- 输入

```bash
xanylabeling --help
```

- 输出

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

### 2.2 显示系统和软件包信息

- 输入

```bash
xanylabeling checks
```

- 输出

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
  PyQt6 Version:                           6.9.1
  ONNX Version:                            1.19.1
  ONNX Runtime Version:                    1.23.2
  ONNX Runtime GPU Version:                None
  OpenCV Contrib Python Headless Version:  4.11.0.86
────────────────────────────────────────────────────────────
```

### 2.3 显示应用版本信息

- 输入

```bash
xanylabeling version
```

- 输出

```bash
3.3.0
```

### 2.4 显示配置文件路径

- 输入

```bash
xanylabeling config
```

- 输出

```bash
~/.xanylabelingrc
```

## 3. 标签格式转换命令

### 3.1 列出所有可用的转换任务

- 输入

```bash
xanylabeling convert
```

- 输出

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

### 3.2 显示特定任务的详细帮助

- 输入（以 `yolo2xlabel` 为例）

```bash
xanylabeling convert --task yolo2xlabel
```

- 输出

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

### 3.3 运行特定的转换任务

- 输入（以 `yolo2xlabel:detect` 为例）

```bash
xanylabeling convert --task yolo2xlabel --mode detect --images ./images --labels ./labels \
    --output ./output --classes classes.txt
```

- 输出

```bash
Converting YOLO detect to XLABEL: 100%|█████████████████████████████████████| 128/128 [00:00<00:00, 2456.47it/s]
✓ Converted 128 files to XLABEL format: ./output
```

### 3.4 转换参数说明

| 参数                  | 说明                                                      | 是否必需 |
|-----------------------|-----------------------------------------------------------|----------|
| `--task`              | 转换任务名称                                              | 是       |
| `--images`            | 图像目录路径                                              | 否       |
| `--labels`            | 标签目录路径                                              | 否       |
| `--output`            | 输出目录路径                                              | 是       |
| `--classes`           | 类别文件路径                                              | 否       |
| `--pose-cfg`          | 姿态配置文件路径                                          | 否       |
| `--mode`              | 转换模式（如：detect、segment、obb、pose）                | 否       |
| `--mapping`           | 映射表文件路径                                            | 否       |
| `--skip-empty-files`  | 跳过创建空输出文件（仅支持 xlabel2yolo 和 xlabel2voc）    | 否       |

> 更多详情，请参考[用户手册-标签导入导出](./user_guide.md#4-标签导入导出)章节。

## 4. 常见问题

### Q1：什么是 XLABEL 格式？

XLABEL 是 X-AnyLabeling 的原生 JSON 格式。它以人类可读的格式存储所有标注信息，包括对象信息、标签信息、属性信息及元数据等。

### Q2：是否需要为所有转换提供类别名称？

类别名称对以下任务是必需的：
- YOLO 转换（detect、segment、obb 模式）
- COCO 转换（detect、segment 模式）
- MOT 转换

VOC 格式在 XML 文件中嵌入了类别名称，因此不需要单独的类别文件。

### Q3：如何创建 classes.txt 文件？

只需创建一个文本文件，每行一个类别名称：

```
person
car
bicycle
dog
cat
```

行号（从 0 开始索引）对应类别 ID。

### Q4：MOT 和 MOTS 有什么区别？

- **MOT**：使用边界框的多目标跟踪
- **MOTS**：使用分割掩码的多目标跟踪

### Q5：可以直接在格式之间转换而不经过 XLABEL 吗？

暂不支持。目前，所有转换都通过 XLABEL 作为中间格式：
1. 首先转换为 XLABEL
2. 然后从 XLABEL 转换为目标格式

### Q6：如果图像在子目录中怎么办？

转换器当前仅处理指定目录中的图像。对于嵌套目录，您可能需要多次运行转换或展平目录结构。

### Q7：如何在 Windows 上处理非 ASCII 路径？

转换器内置了对非 ASCII 路径的支持。确保您的终端编码设置为 UTF-8：

```bash
chcp 65001  # 在 Windows CMD 中
```

### Q8：`--skip-empty-files` 选项是什么？

此选项（由 `xlabel2yolo` 和 `xlabel2voc` 支持）可防止为没有标注的图像创建空标签文件。当您想区分"未标注"和"已标注但为空"的图像时，这很有用。

### Q9：可以使用相对路径吗？

是的，支持相对路径和绝对路径。相对路径从当前工作目录解析。

### Q10：如何转换单个文件？

将单个文件放在目录中，然后对该目录运行转换。转换器会处理指定目录中的所有匹配文件。
