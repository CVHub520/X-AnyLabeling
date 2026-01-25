# AGENT

## 项目概览
X-AnyLabeling 是基于 PyQt5 的桌面标注工具，覆盖图像与视频标注场景，集成 ONNX 推理的自动标注与多任务能力（检测、分割、跟踪、OCR、VQA、grounding 等）。入口为 `xanylabeling` CLI 与 GUI。

## 关键目录
- `anylabeling/`：核心 Python 包
- `anylabeling/views/`：GUI 视图与功能模块
- `anylabeling/services/auto_labeling/`：自动标注服务与模型适配
- `anylabeling/configs/`：默认配置与模型 YAML
- `anylabeling/resources/`：图标、翻译、qrc 资源
- `docs/`：用户与开发文档
- `tests/`：测试用例
- `scripts/`、`tools/`：构建、打包与工具脚本

## 安装与运行
### 依赖与可选组
- 元数据与依赖见 `pyproject.toml`
- Python 版本：>= 3.10
- 可选依赖组：`cpu` / `gpu` / `gpu-cu11` / `dev`

### 安装（源码开发）
```bash
pip install -e .[cpu]
```

### 启动 GUI
```bash
xanylabeling
```

### 直接运行入口
```bash
python anylabeling/app.py
```

### CLI
```bash
xanylabeling checks
xanylabeling version
xanylabeling config
```

## 测试与格式化
### 测试
```bash
pytest
```

### 格式化
```bash
bash scripts/format_code.sh
```

### 预提交
```bash
pre-commit run -a
```

## 打包与发布
- PyInstaller 打包：`scripts/build_executable.sh`
- 平台 spec：`x-anylabeling-*.spec`
- PyPI 构建脚本：`scripts/build_and_publish_pypi.sh`

## 配置约定
- 用户配置默认写入：`~/.xanylabelingrc`
- 内置默认配置：`anylabeling/configs/xanylabeling_config.yaml`
- 自动标注模型配置：`anylabeling/configs/auto_labeling/*.yaml`

## 贡献规范
- 贡献指南：`CONTRIBUTING.md`
- CLA：`CLA.md`
- PR 模板：`.github/PULL_REQUEST_TEMPLATE.md`
- Issue 模板：`.github/ISSUE_TEMPLATE/`
