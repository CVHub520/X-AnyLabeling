# PaddleOCR Document Parsing and Intelligent Text Recognition

## Overview

[PaddleOCR](https://aistudio.baidu.com/paddleocr) is an OCR and document intelligence toolkit in the Baidu PaddlePaddle ecosystem. It covers general text recognition, document layout analysis, table parsing, formula recognition, and other capabilities for common document-processing scenarios such as scanned documents, photographed documents, multi-page PDFs, and technical documents.

X-AnyLabeling integrates these capabilities through a dedicated **PaddleOCR** panel for document understanding and intelligent text recognition workflows. The panel supports layout parsing, text recognition, formula recognition, and table recognition for images and PDF files. After parsing is complete, you can review, edit, copy, and export the recognized results.

Two service modes are supported: you can call the official PaddleOCR API directly, or connect to a PaddleOCR model served through an X-AnyLabeling-compatible remote inference service. Parsed results are saved as local JSON files and displayed in the interface together with source-file regions, structured content, and editable result blocks.

<video src="https://github.com/user-attachments/assets/0c018b6e-f8e9-4045-bc22-0d388ab4853d" width="100%" controls>
</video>

## Model Configuration

### Official API (Recommended)

The X-AnyLabeling client supports the official PaddleOCR API by default, so you do not need to deploy an additional inference service. When the PaddleOCR panel is opened for the first time and no API information has been configured, the `PPOCR_API_Settings` dialog appears automatically. Enter the corresponding `API_URL` and `API_KEY`, and then use the `PPOCR (API)` model to submit document parsing requests. To update the configuration later, click the gear button at the top of the right-side result panel.

<video src="https://github.com/user-attachments/assets/59be57c3-b95e-4f4b-9c02-8bb52496a419" width="100%" controls>
</video>

To obtain the required API information:

1. Visit the [PaddleOCR website](https://aistudio.baidu.com/paddleocr/task).
2. Open the API call example and copy the `API_URL` and `API_KEY`.
3. Return to `PPOCR_API_Settings` in X-AnyLabeling, paste the values, and confirm.

The configuration is saved locally:

```text
${workspace}/xanylabeling_data/paddleocr/api_settings.json
```

By default, `${workspace}` is the user's home directory, `~`. If X-AnyLabeling is started with `--work-dir`, that directory is used instead.

### Local Deployment (Optional)

If you want to run PaddleOCR locally or in a private environment, you can deploy an inference service with [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server). For the setup process, see this [example](../../examples/optical_character_recognition/multi_task/README.md), install the required dependencies, and start the service.

Make sure X-AnyLabeling-Server has been updated to the latest version, and check that the `ppocr_layoutstructv3_vl_1_5` model configuration is available. If you want to implement and integrate your own PaddleOCR inference pipeline, declare the following capability flag in the model configuration. The client uses this flag to determine whether the model can be used in the PaddleOCR panel:

```yaml
...
capabilities:
  ppocr_pipeline: true
...
```

After the service starts, reopen the PaddleOCR annotation panel. The `Parsing Model` drop-down list at the top right displays the currently available models. When you select a model other than `PPOCR (API)`, parsing requests are sent to the deployed inference service automatically.

## User Guide

You can open PaddleOCR in either of the following ways: click the `PaddleOCR` icon in the left toolbar, or use the `Ctrl+4` shortcut.

After the panel opens, click `+ New Parsing` at the top of the left panel to import a file. The imported file is copied to the local PaddleOCR workspace and added to the parsing queue automatically.

The X-AnyLabeling PaddleOCR panel currently supports the following file types:

| Type | Extensions |
| :--- | :--- |
| PDF document | `.pdf` |
| Image | `.bmp`, `.cif`, `.gif`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, `.webp` |

PDF files are first rendered locally into per-page PNG previews, and then parsed page by page. For multi-page PDFs, the page count, preview images, and recognition results are all retained in the local workspace.

> [!TIP]
> - Hold `Ctrl` and scroll the mouse wheel in the source-file preview area to zoom the preview page quickly.
> - Click any block in the left preview area or the right result area to match and highlight the corresponding content on both sides.
> - Double-click a block in the right-side recognition result area, or click the block's `Correct` button, to enter edit mode.
> - Hover over a block in the source-file preview area and click the floating `Copy` button to copy that block's content.
> - For multi-page PDFs, use the page controls at the bottom to jump between pages, or scroll through the page-separated parsing results in the right result area.
> - After you manually correct recognition results, the edited blocks are recorded in the JSON file. To fetch model results again, use the reparse button on the right.

> [!NOTE]
> - The official PaddleOCR API requires a valid `API_URL` and `API_KEY`. If the API returns `401`, check whether the key is valid.
> - Remote-service models appear in the model drop-down list only when `/v1/models` returns models with the `ppocr_pipeline` capability.
> - Imported files are copied to the PaddleOCR workspace. Deleting the original external file does not affect the imported copy.

## Interface Layout

### Overall Layout

The PaddleOCR panel consists of three main areas:

| Area | Description |
| :--- | :--- |
| Left file navigation panel | Import files, view recent files, view favorites, search, filter, and delete files |
| Middle source-file preview area | Display images or PDF pages with PaddleOCR layout blocks, polygon boxes, and category colors overlaid |
| Right parsing result area | Switch between `Document parsing` and `JSON` views, and copy, download, reparse, or edit recognized blocks |

> [!NOTE]
> The colored dot in the lower-left corner of each file item in the left navigation panel indicates the parsing status:
> - Blue means pending or parsing.
> - Green means parsing completed.
> - Red means parsing failed.

### Components

| Location | Button/Component | Function |
| :--- | :--- | :--- |
| Top left | `+ New Parsing` | Import an image or PDF and start parsing automatically |
| Left navigation | `Recent` | Show recently imported and parsed files |
| Left navigation | `Favorites` | Show favorited files only |
| Left navigation | Search button | Expand the filename search box |
| Left navigation | Filter button | Filter by sorting rule, file type, and parsing status |
| Left file item | Star button | Add or remove the current file from favorites |
| Left file item | Delete button | Delete the source file, JSON file, PDF preview pages, block screenshots, and other related data |
| Middle page bar | Left/right arrows | Switch to the previous or next PDF page |
| Middle page bar | Page number input | Jump to a specific PDF page |
| Middle page bar | Zoom out / zoom in buttons | Zoom the source-file preview area |
| Middle page bar | Reset zoom button | Restore the preview scale to fit width |
| Source-file preview area | Floating `Copy` | Copy the content of the currently hovered block |
| Top right | `Parsing Model` | Select `PPOCR (API)` or a remote PaddleOCR model |
| Right view | `Document parsing` | View layout blocks, text, formulas, tables, and images as cards |
| Right view | `JSON` | View the complete JSON result for the current file |
| Right tools | Gear button | Configure the official PaddleOCR `API_URL` and `API_KEY` |
| Right tools | Reparse button | Reparse the current file |
| Right tools | Copy button | Copy Markdown content in the document view, or copy JSON in the JSON view |
| Right tools | Download button | Download a ZIP file in the document view, or download JSON in the JSON view |
| Result block card | `Copy` | Copy the content of a single block |
| Result block card | `Correct` | Enter edit mode for the current block |
| Parsing banner | `Cancel Parsing` | Cancel the current batch parsing task |
| Parsing-failed banner | `Copy Log` | Copy the error log |
| Parsing-failed banner | `Reparse` | Reparse the failed file |

### Parsed Blocks and Editors

After parsing is complete, the right-side `Document parsing` view displays blocks in layout order. Different block types use different colors:

| Type | Example Labels | Color Meaning |
| :--- | :--- | :--- |
| Text | `text`, `doc_title`, `paragraph_title`, `footer`, `seal`, etc. | Blue |
| Table | `table` | Green |
| Image | `image`, `chart`, `header_image`, `footer_image` | Purple |
| Header | `header` | Light purple |
| Formula | `display_formula`, `formula`, `formula_number`, `algorithm` | Yellow |
| Edited block | Any block | Orange border or edit-state marker |

The following editors are currently supported:

| Editor | Trigger Scenario | Description |
| :--- | :--- | :--- |
| Rich text editor | Plain text, titles, footers, seals, and other non-table or non-formula content | Supports basic rich text editing and saves the result as Markdown/text content |
| LaTeX formula editor | `display_formula`, `formula`, `formula_number`, `algorithm` | Supports editing LaTeX source and renders a live preview below |
| Table editor | `table` or content recognized as a table structure | Supports cell editing, selection copying, adding and deleting rows or columns, and basic text styling |

> [!WARNING]
> If an item contains many formulas, the first time you open it or scroll to the corresponding result block may take a short while. This is mainly caused by formula preview rendering. Rendered results are cached, so the same content does not need to be rendered again on subsequent loads.

### Data Storage and Directory Structure

The PaddleOCR panel saves imported files and parsed results in the local workspace:

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

| Path | Description |
| :--- | :--- |
| `api_settings.json` | Cached `API_URL` and `API_KEY` for the official PaddleOCR API |
| `ui_state.json` | UI state, such as the list of favorited files |
| `files/` | Local copies of imported files |
| `files/__PDF_<filename>/` | Per-page PNG previews rendered from a PDF |
| `files/__BLOCK_IMAGES_<filename>/` | Local crops for image-type blocks |
| `jsons/<filename>.json` | PaddleOCR parsing results and edited results for the current file |

> [!NOTE]
> Deleting a file item in the left panel also deletes the source file, local JSON file, PDF preview pages, and block crop images.

### JSON Data Structure

Each imported file has a corresponding JSON file. The core structure is as follows:

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
            "block_content": "Recognized text content",
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
        "text": "Full-page Markdown content",
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

Key fields:

| Field | Description |
| :--- | :--- |
| `layoutParsingResults` | Page-level parsing results. Images usually have one page, while PDFs can have multiple pages |
| `prunedResult.parsing_res_list` | Block list for the current page |
| `block_label` | Block type, such as `text`, `table`, `display_formula`, or `image` |
| `block_content` | Recognized content that can be viewed, copied, and edited |
| `block_bbox` | Rectangular bounding box of the block, in `[x1, y1, x2, y2]` format |
| `block_polygon_points` | Polygon points of the block, used for highlighting in the preview area |
| `markdown.text` | Markdown text returned by PaddleOCR or generated by combining blocks |
| `markdown.images` | Mapping of image resources referenced in the Markdown |
| `dataInfo.type` | File type. Valid values are `image` and `pdf` |
| `dataInfo.numPages` | Number of pages |
| `_ppocr_meta.status` | Parsing status: `pending`, `parsed`, or `error` |
| `_ppocr_meta.edited_blocks` | List of block keys that have been manually edited |
| `_ppocr_meta.block_image_paths` | Local resource paths for image-type blocks |
| `_ppocr_meta.pipeline_model` | Parsing model that generated the result |

### Downloading Results

The download button on the right exports different content depending on the current view:

| Current View | Downloaded Content |
| :--- | :--- |
| `Document parsing` | A ZIP file containing `doc_0.md`, image resources under `imgs/`, and layout detection visualizations named `layout_det_res_*.jpg` |
| `JSON` | The complete JSON for the current file. The default filename is `<original_filename>_by_<model_name>.json` |
