# PaddleOCR 文档解析与智能文字识别

## 简介

[PaddleOCR](https://aistudio.baidu.com/paddleocr) 是百度飞桨生态中的 OCR 与文档智能工具，覆盖通用文字识别、文档版面分析、表格解析、公式识别等能力，适用于扫描件、拍照文档、多页 PDF、技术文档等常见资料处理场景。

现在，我们将这一能力集成到 `X-AnyLabeling` 中，提供了面向文档理解与智能文字识别工作流的 **PaddleOCR** 面板，支持对图片和 PDF 文件进行版面解析、文字识别、公式识别和表格识别，并在识别完成后对结果进行校对、编辑、复制和导出。

同时，提供了两种服务接入方式：既可以直接调用 PaddleOCR 官方 API，也可以接入兼容 X-AnyLabeling 远程推理服务的 PaddleOCR 模型。解析结果会保存为本地 JSON 文件，并在界面中同步展示源文件区域框、结构化内容和可编辑的结果块。

<video src="https://github.com/user-attachments/assets/0c018b6e-f8e9-4045-bc22-0d388ab4853d" width="100%" controls>
</video>

## 模型配置

### 官方 API（推荐）

X-AnyLabeling 客户端默认支持 PaddleOCR 官方 API 调用形式，无需额外部署推理服务。首次加载 PaddleOCR 面板且尚未配置 API 信息时，界面会自动弹出 `PPOCR API Settings` 配置窗口；用户只需填写相应的 `API_URL` 和 `API_KEY`，即可通过 `PPOCR (API)` 模型发起文档解析请求。后续如需修改配置，也可以点击右侧结果面板顶部的齿轮按钮手动打开该窗口。

<video src="https://github.com/user-attachments/assets/59be57c3-b95e-4f4b-9c02-8bb52496a419" width="100%" controls>
</video>

获取方式：

1. 访问 [PaddleOCR 官网](https://aistudio.baidu.com/paddleocr/task)。
2. 进入 API 调用示例，复制 `API_URL` 和 `API_KEY`。
3. 回到 X-AnyLabeling 的 `PPOCR API Settings`，粘贴并确认。

配置会保存在本地：

```text
${workspace}/xanylabeling_data/paddleocr/api_settings.json
```

默认情况下，`${workspace}` 为用户主目录 `~`；如果启动 X-AnyLabeling 时传入了 `--work-dir`，则以该目录为准。

### 本地部署（可选）

如果您希望在本地或私有环境中运行 PaddleOCR，也可以通过 [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server) 自行部署推理服务。具体流程可参考此[示例](../../examples/optical_character_recognition/multi_task/README.md)，安装相关依赖并启动服务。

请确保 X-AnyLabeling-Server 已更新至最新版本，并检查 `ppocr_layoutstructv3_vl_1_5` 模型配置是否存在。如果您希望自行实现并集成 PaddleOCR 推理管道，请在模型配置中声明以下能力标识，客户端会据此判断该模型是否可用于 PaddleOCR 面板：

```yaml
...
capabilities:
  ppocr_pipeline: true
...
```

服务启动后，重新打开 PaddleOCR 标注面板，右侧顶部的 `解析模型` 下拉框会显示当前可用模型。选择非 `PPOCR (API)` 的模型时，解析任务会自动发送至已部署的推理服务。


## 使用手册

您可以通过以下任一方式打开 PaddleOCR：点击左侧工具栏中的 `PaddleOCR` 图标或直接使用快捷键 `Ctrl+4`。

打开后，点击左侧面板顶部的 `+ New Parsing` 导入文件。文件导入后会同步复制到本地 PaddleOCR 工作目录，并自动加入解析队列。

当前，X-AnyLabeling PaddleOCR 面板支持导入以下文件：

| 类型 | 后缀 |
| :--- | :--- |
| PDF 文档 | `.pdf` |
| 图片 | `.bmp`, `.cif`, `.gif`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, `.webp` |

这里，PDF 文件会先在本地渲染为逐页 PNG 预览图，再按页进行解析。因此多页 PDF 的页数、预览图和识别结果都会在本地工作目录中保留。

> [!TIP]
> - 在源文件预览区按住 `Ctrl` 并滚动鼠标滚轮，可以快速缩放预览页面。
> - 单击左侧预览区或右侧结果区中的任意块，可以在两侧快速匹配并高亮对应内容。
> - 双击右侧识别结果区某个块，或点击该块的`纠正`按钮，可以进入编辑状态。
> - 鼠标悬停在源文件预览区的块上时，可以直接点击浮出的`复制`按钮方便复制该块内容。
> - 多页 PDF 可以通过底部页码控件跳转页面，也可以在右侧结果区滚动查看按页分隔的解析结果。
> - 对识别结果做人工修正后，JSON 中会记录已编辑块；如需重新获取模型结果，可使用右侧重解析按钮。

> [!NOTE]
> - PaddleOCR 官方 API 需要可用的 `API_URL` 和 `API_KEY`；如果接口返回 401，请检查密钥是否有效。
> - 远程服务只有在 `/v1/models` 返回具备 `ppocr_pipeline` 能力的模型时，才会出现在模型下拉框中。
> - 导入文件会复制到 PaddleOCR 工作目录中；删除原始外部文件不会影响已经导入的副本。


## 界面布局

### 整体布局

PaddleOCR 面板由三部分组成：

| 区域 | 说明 |
| :--- | :--- |
| 左侧文件导航栏 | 导入文件、查看最近文件、收藏文件、搜索、筛选、删除文件 |
| 中间源文件预览区 | 显示图片或 PDF 页面，并叠加 PaddleOCR 返回的版面块、多边形框和类别颜色 |
| 右侧解析结果区 | 切换 `Document parsing` 和 `JSON` 视图，复制、下载、重解析、编辑识别块 |

> [!NOTE]
> 左侧文件导航栏每个文件左下角的彩色点表示解析状态：
> - 蓝色表示待解析或正在解析
> - 绿色表示解析完成
> - 红色表示解析失败

### 相关组件说明

| 位置 | 按钮/组件 | 功能 |
| :--- | :--- | :--- |
| 左侧顶部 | `+ 新建解析` | 导入图片或 PDF，并自动启动解析 |
| 左侧导航 | `最近使用` | 显示最近导入和解析的文件 |
| 左侧导航 | `收藏夹` | 只显示已收藏文件 |
| 左侧导航 | 搜索按钮 | 展开文件名搜索框 |
| 左侧导航 | 筛选按钮 | 按排序、文件类型、解析状态筛选 |
| 左侧文件项 | 星标按钮 | 收藏或取消收藏当前文件 |
| 左侧文件项 | 删除按钮 | 删除源文件、JSON、PDF 预览页和块截图等关联数据 |
| 中间页码栏 | 左/右箭头 | 切换 PDF 上一页或下一页 |
| 中间页码栏 | 页码输入框 | 跳转到指定 PDF 页 |
| 中间页码栏 | 缩小/放大按钮 | 缩放源文件预览区 |
| 中间页码栏 | 重置缩放按钮 | 恢复为适合宽度的预览比例 |
| 源文件预览区 | 悬浮 `复制` | 复制当前悬浮块的内容 |
| 右侧顶部 | `解析模型` | 选择 `PPOCR (API)` 或远程 PaddleOCR 模型 |
| 右侧视图 | `文档解析` | 以卡片形式查看版面块、文本、公式、表格、图片 |
| 右侧视图 | `JSON` | 查看当前文件完整 JSON 结果 |
| 右侧工具 | 齿轮按钮 | 配置 PaddleOCR 官方 `API_URL` 和 `API_KEY` |
| 右侧工具 | 重解析按钮 | 重新解析当前文件 |
| 右侧工具 | 复制按钮 | 在文档视图复制 Markdown 内容，在 JSON 视图复制 JSON |
| 右侧工具 | 下载按钮 | 在文档视图下载 ZIP，在 JSON 视图下载 JSON |
| 结果块卡片 | `复制` | 复制单个块内容 |
| 结果块卡片 | `纠正` | 进入当前块编辑状态 |
| 解析中横幅 | `取消解析` | 取消当前批量解析任务 |
| 解析失败横幅 | `复制日志` | 复制错误日志 |
| 解析失败横幅 | `重新解析` | 重新解析失败文件 |

### 解析块与编辑器

解析完成后，右侧 `Document parsing` 会按版面顺序展示各个块。不同块类型会使用不同颜色：

| 类型 | 示例标签 | 颜色含义 |
| :--- | :--- | :--- |
| 文本类 | `text`, `doc_title`, `paragraph_title`, `footer`, `seal` 等 | 蓝色 |
| 表格类 | `table` | 绿色 |
| 图片类 | `image`, `chart`, `header_image`, `footer_image` | 紫色 |
| 页眉类 | `header` | 淡紫色 |
| 公式类 | `display_formula`, `formula`, `formula_number`, `algorithm` | 黄色 |
| 已编辑块 | 任意 block | 橙色边框或编辑状态标记 |

当前支持以下编辑器：

| 编辑器 | 触发场景 | 说明 |
| :--- | :--- | :--- |
| 富文本编辑器 | 普通文本、标题、页脚、印章等非表格/公式内容 | 支持基础富文本编辑，并保存为 Markdown/文本内容 |
| LaTeX 公式编辑器 | `display_formula`, `formula`, `formula_number`, `algorithm` | 支持编辑 LaTeX 源码，并在下方实时渲染预览 |
| 表格编辑器 | `table` 或被识别为表格结构的内容 | 支持单元格编辑、选择复制、增删行列、基础文本样式 |

> [!WARNING]
> 如果某个 item 中包含较多公式，首次打开或首次滚动到相关结果块时可能需要等待一小段时间。这主要来自公式预览渲染；渲染结果会被缓存，后续再次加载同一内容不会重复等待。

### 数据保存与目录结构

PaddleOCR 面板会把导入文件和解析结果保存在本地工作目录中：

```text
${workspace}/xanylabeling_data/paddleocr/
├── api_settings.json
├── ui_state.json
├── files/
│   ├── example.pdf
│   ├── image.png
│   ├── __PDF_example/
│   │   ├── page_001.png
│   │   └── page_002.png
│   └── __BLOCK_IMAGES_image.png/
│       └── page_001_block_0001.png
└── jsons/
    ├── example.pdf.json
    └── image.png.json
```

| 路径 | 说明 |
| :--- | :--- |
| `api_settings.json` | 缓存 PaddleOCR 官方 API 的 `API_URL` 和 `API_KEY` |
| `ui_state.json` | UI 状态，例如收藏文件列表 |
| `files/` | 导入文件的本地副本 |
| `files/__PDF_<文件名>/` | PDF 渲染后的逐页 PNG 预览图 |
| `files/__BLOCK_IMAGES_<文件名>/` | 图片类 block 的本地裁剪图 |
| `jsons/<文件名>.json` | 当前文件的 PaddleOCR 解析结果和编辑结果 |

> [!NOTE]
> 删除左侧文件项时，会同时删除源文件、本地 JSON、PDF 预览页和 block 裁剪图。

### JSON 数据结构

每个导入文件对应一个 JSON 文件，核心结构如下：

```json
{
  "layoutParsingResults": [
    {
      "prunedResult": {
        "page_count": 1,
        "width": 1240,
        "height": 1754,
        "model_settings": {
          "pipeline_model": "__ppocr_api__"
        },
        "parsing_res_list": [
          {
            "block_label": "text",
            "block_content": "识别出的文本内容",
            "block_bbox": [100, 120, 500, 180],
            "block_id": 1,
            "block_order": 1,
            "group_id": 1,
            "global_block_id": 1,
            "global_group_id": 1,
            "block_polygon_points": [
              [100, 120],
              [500, 120],
              [500, 180],
              [100, 180]
            ]
          }
        ]
      },
      "markdown": {
        "text": "整页 Markdown 内容",
        "images": {
          "page_1:block_1": "files/__BLOCK_IMAGES_image.png/page_001_block_0001.png"
        }
      },
      "outputImages": {},
      "inputImage": "files/image.png"
    }
  ],
  "preprocessedImages": [],
  "dataInfo": {
    "type": "image",
    "numPages": 1,
    "pages": [
      {
        "width": 1240,
        "height": 1754
      }
    ]
  },
  "_ppocr_meta": {
    "status": "parsed",
    "source_path": "files/image.png",
    "updated_at": "2026-04-18 12:00:00",
    "error_message": "",
    "edited_blocks": [],
    "block_image_paths": {},
    "pipeline_model": "__ppocr_api__"
  }
}
```

关键字段说明：

| 字段 | 说明 |
| :--- | :--- |
| `layoutParsingResults` | 按页保存的解析结果；图片通常为 1 页，PDF 为多页 |
| `prunedResult.parsing_res_list` | 当前页的 block 列表 |
| `block_label` | block 类型，例如 `text`、`table`、`display_formula`、`image` |
| `block_content` | 可查看、复制和编辑的识别内容 |
| `block_bbox` | block 的矩形外接框，格式为 `[x1, y1, x2, y2]` |
| `block_polygon_points` | block 的多边形点位，用于预览区高亮 |
| `markdown.text` | PaddleOCR 返回或由 block 拼接生成的 Markdown 文本 |
| `markdown.images` | Markdown 中引用的图片资源映射 |
| `dataInfo.type` | 文件类型，取值为 `image` 或 `pdf` |
| `dataInfo.numPages` | 页数 |
| `_ppocr_meta.status` | 解析状态：`pending`、`parsed`、`error` |
| `_ppocr_meta.edited_blocks` | 已被人工编辑过的 block key 列表 |
| `_ppocr_meta.block_image_paths` | 图片类 block 的本地资源路径 |
| `_ppocr_meta.pipeline_model` | 生成该结果的解析模型 |

### 下载结果

右侧下载按钮会根据当前视图导出不同内容：

| 当前视图 | 下载内容 |
| :--- | :--- |
| `Document parsing` | ZIP 包，包含 `doc_0.md`、`imgs/` 图片资源和 `layout_det_res_*.jpg` 版面检测可视化图 |
| `JSON` | 当前文件的完整 JSON，默认文件名为 `<原文件名>_by_<模型名>.json` |
