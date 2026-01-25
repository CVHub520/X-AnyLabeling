# PyQt6 适配差异梳理与修复清单（基于官方 API 文档）

本文用于把项目中 **PyQt5 风格写法** 迁移到 **PyQt6/Qt6**。内容以 Riverbank（PyQt6 官方文档）为准，按“差异点 → 常见报错 → Qt6 正确写法 → 本仓库命中位置”组织，便于逐项收敛。

## 文档依据（Context7）

- QAction：<https://www.riverbankcomputing.com/static/Docs/PyQt6/api/qtgui/qaction>
- Qt / ScrollBarPolicy / WindowType / PyQt5 differences：<https://www.riverbankcomputing.com/static/Docs/PyQt6/api/qtcore/qt>、<https://www.riverbankcomputing.com/static/Docs/PyQt6/pyqt5_differences>
- QCompleter：<https://www.riverbankcomputing.com/static/Docs/PyQt6/api/qtwidgets/qcompleter>
- QFileDialog：<https://www.riverbankcomputing.com/static/Docs/PyQt6/api/qtwidgets/qfiledialog>
- QDockWidget：<https://www.riverbankcomputing.com/static/Docs/PyQt6/api/qtwidgets/qdockwidget>

## 关键差异点与修复方式

### 1) 枚举/Flags “扁平常量”变为“嵌套枚举”

**现象/报错**
- `AttributeError: type object 'Qt' has no attribute 'ScrollBarAlwaysOff'`
- `AttributeError: type object 'Qt' has no attribute 'FramelessWindowHint'`
- `AttributeError: type object 'Qt' has no attribute 'MoveAction'`

**原因（Qt6/PyQt6 设计）**
- PyQt6 将命名枚举实现为 Python `Enum`，并将 QFlags 合并成 `Flag` 子类；同时大量常量变为 **作用域枚举**（scoped enums），导致 `Qt.Xxx` 不再直接暴露旧式常量名。参考：PyQt6 `pyqt5_differences`。

**修复方式（推荐：改成 Qt6 原生写法）**
- ScrollBarPolicy：
  - 旧：`Qt.ScrollBarAlwaysOff`
  - 新：`Qt.ScrollBarPolicy.ScrollBarAlwaysOff`
  - 适用 API：`QAbstractScrollArea.setHorizontalScrollBarPolicy(...)`
- Window flags：
  - 旧：`Qt.FramelessWindowHint`
  - 新：`Qt.WindowType.FramelessWindowHint`
  - 适用 API：`QWidget.setWindowFlags(...)` / `setWindowFlag(...)`
- DropAction：
  - 旧：`Qt.MoveAction`
  - 新：`Qt.DropAction.MoveAction`
  - 适用 API：`QAbstractItemView.setDefaultDropAction(...)`
- MatchFlag（常用于 QCompleter / 过滤）：`Qt.MatchFlag.MatchContains` 等

**本仓库命中位置（优先修复）**
- `anylabeling/views/labeling/widgets/canvas.py`（FocusPolicy、鼠标/按键相关）
- `anylabeling/views/labeling/widgets/toolbar.py`（WindowType）
- `anylabeling/views/labeling/label_widget.py`（ScrollBarPolicy、ContextMenuPolicy、DockWidgetFeature 等大量枚举）
- `anylabeling/views/common/toaster.py`、`anylabeling/views/labeling/shape.py` 等

### 2) QAction 归属模块变化：QtWidgets → QtGui

**现象/报错**
- `AttributeError: module 'PyQt6.QtWidgets' has no attribute 'QAction'`

**正确写法（文档：QAction 属于 PyQt6.QtGui）**
- `from PyQt6.QtGui import QAction`
- 或使用命名空间：`QtGui.QAction(...)`

**本仓库命中位置**
- `anylabeling/views/labeling/utils/qt.py`（创建 action 的工厂函数）

### 3) QCompleter CompletionMode / filterMode 变为作用域枚举

**现象/报错**
- `AttributeError: type object 'QCompleter' has no attribute 'InlineCompletion'`
- 过滤相关 API 参数类型不匹配

**正确写法（文档：QCompleter）**
- CompletionMode：
  - 旧：`QCompleter.InlineCompletion`
  - 新：`QCompleter.CompletionMode.InlineCompletion`
- filterMode：
  - 新：`Qt.MatchFlag.MatchContains`（或其它 MatchFlag）

**本仓库命中位置**
- `anylabeling/views/labeling/widgets/label_dialog.py`（completer 初始化与过滤）

### 4) QFileDialog Option 变为作用域枚举

**现象/报错**
- `AttributeError: type object 'QFileDialog' has no attribute 'ShowDirsOnly'`
- `AttributeError: type object 'QFileDialog' has no attribute 'DontResolveSymlinks'`

**正确写法（文档：QFileDialog）**
- `QFileDialog.getExistingDirectory(..., options=QFileDialog.Option.ShowDirsOnly)`（默认参数也是 `ShowDirsOnly`）\n+- 组合 options：`QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks`

**本仓库命中位置（高频、建议统一处理）**
- `anylabeling/views/labeling/label_widget.py`
- `anylabeling/views/labeling/utils/upload.py`
- `anylabeling/views/labeling/utils/video.py`
- `anylabeling/views/labeling/widgets/chatbot_dialog.py`
- `anylabeling/views/training/ultralytics_dialog.py`

### 5) QDockWidget 的 features 类型变化（DockWidgetFeature）

**现象/报错**
- `AttributeError: type object 'QDockWidget' has no attribute 'DockWidgetFeatures'`
- 或 features 初始化/组合时类型不匹配

**正确写法（文档：QDockWidget）**
- 特性枚举：`QDockWidget.DockWidgetFeature`，成员包括 `DockWidgetClosable`、`DockWidgetMovable`、`DockWidgetFloatable`、`NoDockWidgetFeatures`。\n+- 初始值建议使用：\n+  - `features = QDockWidget.DockWidgetFeature.NoDockWidgetFeatures`\n+  - 或 `features = QDockWidget.DockWidgetFeature(0)`\n+- 组合：`features = features | QDockWidget.DockWidgetFeature.DockWidgetClosable`

**本仓库命中位置**
- `anylabeling/views/labeling/label_widget.py`（多处 setFeatures/位运算）

### 6) bool vs CheckState：setChecked 只能传 bool

**现象/报错**
- `TypeError: setChecked(self, a0: bool): argument 1 has unexpected type 'CheckState'`

**正确写法**
- `setChecked(True/False)` 只接受 bool\n+- 若要设置三态/检查状态，用 `setCheckState(Qt.CheckState.Checked)`（对应 API 为 setCheckState）

**本仓库命中位置**
- `anylabeling/views/labeling/label_widget.py`、以及包含 `Qt.Checked/Unchecked` 的对话框/控件文件

## 实施建议（落地顺序）

1. 先把启动链路（`app.py → MainWindow → LabelingWidget`）阻塞错误清零。\n+2. 再按 GUI 核心路径逐项点开修：打开目录/文件（QFileDialog）、工具栏与菜单（QAction）、DockWidget（features/flags）、搜索/补全（QCompleter）。\n+3. 最后覆盖训练、VQA、Chatbot、Auto-labeling 相关窗口，保证它们能被打开并基础交互不崩溃。\n+
