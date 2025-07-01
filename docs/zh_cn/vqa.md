# 概述

`X-AnyLabeling` 视觉问答工具是一款专为图像问答数据集标注设计的专业系统。该工具支持创建图像对应的问题-答案对，提供多种输入组件类型，具备高度可配置性，能够灵活适应不同标注任务的需求。视觉问答工具特别适用于构建高质量的视觉问答训练数据，可用于监督式微调、视觉链式思维（CoT）等前沿任务的模型训练。

<video src="https://github.com/user-attachments/assets/92807672-f101-475b-b790-9caa1c31da05" width="100%" controls>
</video>

# 入门

## 访问视觉问答工具

要打开视觉问答工具，请点击 X-AnyLabeling 中左侧工具栏的视觉问答图标或使用如下快捷键快速启动聊天机器人对话界面。

- Windows/Linux: `Ctrl` + `1`
- macOS: `⌘` + `1`

## 初始设置

每次启动时，系统会自动从以下路径加载默认配置。您可以根据标注需求添加自定义组件来构建个性化的标注界面。

```bash
~/xanylabeling_data/vqa/components.json
```

# 用户界面

视觉问答工具采用双面板界面设计，左侧为图像展示与导航区域，右侧为标注控制区域，用于配置、编辑及导出问题-答案对。

<img src="../../assets/resources/vqa_panel.png" width="100%" />

## 左侧面板 - 图像显示区

* **文件名与进度指示**：显示当前图像文件名及其在数据集中的位置（如：`000000000154.jpg (33/128)`）。
* **图像预览区域**：居中显示图像，支持自适应缩放。
* **图像导航功能**：

  * "上一张 / 下一张"按钮用于顺序浏览；
  * 页码输入框支持快速跳转。

## 右侧面板 - 标注控制区

* **操作功能区**：

  * **Load Images**：加载图像目录；
  * **Export Labels**：导出标注数据为 JSONL 格式；
  * **Clear All**：清除当前图像的所有标注项。

* **标注组件区**：根据用户配置动态加载标注组件，支持实时编辑。组件类型包括：

  文本输入框（`QLineEdit`）

  * 用于开放式问答，如图像描述、详细回答等。

  单选按钮组（`QRadioButton`）

  * 适用于单选题，如任务类型选择、数据集划分等；
  * 默认选中第一个选项；
  * 支持自定义选项内容。

  复选框组（`QCheckBox`）

  * 适用于多选题，如图像标签、属性标记等；
  * 支持多选，默认无选中项。

  下拉菜单（`QComboBox`）

  * 适用于选项较多的单选场景；
  * 默认显示"-- Select --"提示；
  * 支持自定义选项列表。

* **组件管理区**：

  * **Add Component**：新增标注组件；
  * **Del Component**：删除已配置组件。

<div style="display: flex; justify-content: space-between;">
  <img src="../../assets/resources/vqa_add_componet.png" width="56%" />
  <img src="../../assets/resources/vqa_del_componet.png" width="43%" />
</div>

# 数据存储

X-AnyLabeling 默认启用 VQA 标注的自动保存功能。所有标注数据都会自动保存到与图像文件同目录下的 JSON 文件中。

## VQA数据格式

VQA 标注数据存储在标签文件的 `vqaData` 字段中。数据结构包含通过配置组件捕获的所有信息：

```json
{
  "version": "3.0.0",
  "flags": {},
  "shapes": [],
  ...
  "vqaData": {
    "question": "这里有多少只斑马？",
    "answer": "3只",
    "split": "train",
    "task": "计数",
    "tags": [
      "动物"
    ]
  },
  "imagePath": "0000000000154.jpg",
  "imageHeight": 640,
  "imageWidth": 480
}
```

## 数据导出

使用 **Export Labels** 按钮可将所有 VQA 标注导出为 JSONL 格式，其中每行包含一张图像的标注数据，示例如下：

```jsonl
{"image": "0000000000154.jpg", "question": "图像中有几只斑马？", "answer": 3, "split": "train"}
{"image": "0000000000155.jpg", "question": "猫在做什么？", "answer": "睡觉", "split": "val"}
...
```
