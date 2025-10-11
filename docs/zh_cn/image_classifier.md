# 概述

X-AnyLabeling 的图像分类器是一个专门用于图像分类标注的功能模块，它提供了一个独立的对话框界面，让用户可以方便地对图像数据集进行分类标注。该模块支持多类分类（单标签）和多标签分类（多标签）两种模式，提供添加、删除、编辑标签等完整的标签管理功能，集成了AI智能分类功能支持单张和批量自动分类，提供标签使用频次等数据集统计信息，支持键盘快捷键快速切换图片和标注，并可将已分类的图片按类别导出到对应文件夹。

<video src="https://github.com/user-attachments/assets/0652adfb-48a4-4219-9b18-16ff5ce31be0" width="100%" controls>
</video>

# 启动

要打图像分类器窗口，请先确保主窗口图像目录已加载，随后点击主窗口左侧工具栏中的图像分类器图标（布偶猫头像）或使用以下快捷键启动：

- Windows/Linux: `Ctrl` + `3`
- macOS: `⌘` + `3`

> [!NOTE]
> 打开图像分类器窗口时主窗口会自动隐藏。如果最小化本窗口后任务栏图标消失，可使用 `Alt` + `Tab`（Windows）或 `⌘` + `Tab`（macOS）切换回窗口。关闭分类器窗口后主窗口会重新显示。

# 教程

视觉问答工具采用双面板设计，左侧为图像预览区域，右侧为标注控制区域。

<img src="../../assets/resources/image_classifier/entire_panel.png" width="100%" />

## 左侧面板 - 图像预览区

- **文件名与进度指示**：显示当前图像文件名及其在数据集中的位置（如：husky.png (1/3) ✅ 或 ragdoll.png (3/3) ❌）。
- **图像预览区域**：居中显示图像，支持自适应缩放。当选中标签时，标签文字会实时显示在预览区域右上角，便于快速查看和确认。

> [!TIP]
> ✅ 代表当前图像已标注
> ❌ 代表当前图像未标注

## 右侧面板 - 标注控制区

- **功能组件区**：

| 功能按钮 | 说明 |
|----------|------|
| Export | 将已分类的图像按类别导出到文件夹中（仅支持 MultiClass 模式） |
| MultiClass | 多类分类模式，每张图片只能选择一个标签 |
| MultiLabel | 多标签分类模式，每张图片可以选择多个标签（v3.2.7+） |
| AutoRun | 使用AI模型批量自动分类所有图像 |

> [!NOTE]
> - MultiClass 模式：适用于互斥分类任务，如动物种类识别（一张图片只能是一种动物）
> - MultiLabel 模式：适用于多属性标注任务，如图片标签标注（一张图片可以同时具有多个属性）
> - 需要注意的是，当从 MultiLabel 切换到 MultiClass 模式时，系统只会保留每张图片的第一个勾选标签


```bash
classified/
├── husky/
│   ├── husky_playing.jpg
│   └── husky_sleeping.jpg
├── psyduck/
│   ├── psyduck_swimming.png
│   └── psyduck_standing.png
└── ragdoll/
    ├── ragdoll_grooming.jpg
    └── ragdoll_sitting.jpg
```

- **标注组件区**：

右侧中间面板包含标题名称及以下5个标注组件。

### AI助手

若需使用AI智能辅助功能，请先参照 [Chatbot](../zh_cn/chatbot.md) 章节完成相关配置。

<img src="../../assets/resources/vqa/chatbot.png" width="100%" />

完成配置后，你可以通过点击标注组件区右侧的魔法棒（🪄）图标来打开AI智能对话框。

<img src="../../assets/resources/image_classifier/assistance.png" width="100%" />

软件内置了一套标准的提示词模板，会根据当前选择的模式（MultiClass 或 MultiLabel）自动调整提示内容。你可以直接使用或根据实际需求自定义。

**MultiClass 模式示例：**

```prompt
@image
You are an expert image classifier. Your task is to perform multi-class classification.

Task Definition: Analyze the given image and classify it based on the provided categories.

Available Categories: ["husky", "psyduck", "ragdoll"]

Instructions:
1. Carefully examine the image and identify the main subject and their activity
2. Be precise - only select categories that clearly match what you observe

Return your result in strict JSON format:
{"husky": false, "psyduck": false, "ragdoll": false}

Set exactly ONE category to 'true' that best matches the image, keep all others as 'false'.
```

**MultiLabel 模式示例：**

```prompt
@image
You are an expert image classifier. Your task is to perform multi-label classification.

Task Definition: Analyze the given image and classify it based on the provided categories.

Available Categories: ["outdoor", "sunny", "people", "building"]

Instructions:
1. Carefully examine the image and identify the main subject and their activity
2. Be precise - only select categories that clearly match what you observe

Return your result in strict JSON format:
{"outdoor": false, "sunny": false, "people": false, "building": false}

Set ALL applicable categories to 'true', keep non-applicable ones as 'false'.
```

> 关于智能对话框的高阶使用方法，如特殊引用及模板库设置等，请参考[VQA标注文档](./vqa.md)。

### 添加标签

<img src="../../assets/resources/image_classifier/add_label.png" width="100%" />

软件提供了两种添加标签的方式：

1. **手动输入**：适用于少量标签的情况，直接在编辑框中输入标签名称，每行一个，且标签名之间不允许重复。
2. **文件导入**：适用于大量标签的情况，可以提前准备一个 [classes.txt](../../assets/labels.txt) 文件（每行一个标签），通过左侧的上传按钮导入。文件上传后会自动解析到编辑框中。

确认标签列表无误后，点击添加按钮完成标签添加。

> ![NOTE]
> 此处仅支持添加新标签，不支持修改已有标签，如需编辑或删除现有标签，请参考后续小节。

### 删除标签

<img src="../../assets/resources/image_classifier/delete_label.png" width="100%" />

删除标签组件用于移除已有的标签。你可以通过点击选择一个或多个标签（被选中的标签会以高亮显示），然后点击删除按钮。在确认对话框中再次确认后，系统将自动遍历当前任务中的所有标签文件，并删除所选标签相关的所有内容。

### 编辑标签

<img src="../../assets/resources/image_classifier/edit_label.png" width="100%" />

标签编辑组件以表格形式展示，左侧列显示当前标签（只读），右侧列用于输入新的标签名称。要修改标签，只需双击右侧单元格进入编辑模式，输入新的标签名称即可。完成所有修改后，点击保存按钮使更改生效。

### 数据统计

<img src="../../assets/resources/image_classifier/dataset_statistics.png" width="100%" />

数据统计组件提供了当前任务的数据集统计信息，包括：

1. **总体统计**：显示数据集中的总样本数、已标注数量和未标注数量。
2. **标签分布**：以水平条形图的形式展示每个标签的使用频次及占比。图中使用不同颜色区分不同标签，每个标签后显示具体的数量和百分比。

- **导航组件区**

支持用户手动点击左/右箭头来手动切换上/下一张图片，或通过中间页码输入框输入指定数字跳转到指定图片。此外，提供以下快捷键方便快速切换图片：

| 快捷键 | 功能 |
|--------|------|
| A | 上一张图片 |
| D | 下一张图片 |
| Ctrl+A 或 ⌘+A | 上一张未标注的图片 |
| Ctrl+D 或 ⌘+D | 下一张未标注的图片 |

# 数据

图像分类器的标注信息共享 X-AnyLabeling 标签文件的 `flags` 字段，详情可参考[用户手册-1.4 保存标签数据](./user_guide.md#14-保存标签数据)章节。

```json
{
  "version": "3.2.3",
  "flags": {
    "husky": true,
    "psyduck": false,
    "ragdoll": false
  },
  "shapes": [],
  "imagePath": "husky.png",
  "imageData": null,
  "imageHeight": 200,
  "imageWidth": 200,
  "description": ""
}
```
