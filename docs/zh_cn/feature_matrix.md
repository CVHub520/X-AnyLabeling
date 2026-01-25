# 功能回归覆盖矩阵

本文件用于跟踪“用户可见功能 → 自动化用例”的覆盖情况，避免只追求代码覆盖率而遗漏关键业务回归。

## 运行方式

```bash
python -m pytest -q
```

## 覆盖矩阵

| 功能 | 覆盖点 | 用例位置 |
|---|---|---|
| 应用启动 | Qt 兼容层与核心部件构造 | tests/test_qt_compat.py, tests/test_pyqt6_smoke.py |
| 图像导入 | 导入目录、文件去重、已标注标记 | tests/test_labeling_widget_qt.py |
| 画布绘制 | 矩形绘制、shape 进入列表、信号触发 | tests/test_canvas_qt.py |
| 画布编辑 | 移动后 store、撤销 restore | tests/test_canvas_qt.py |
| 标签对话框 | OK/Cancel 返回值、flags/群组/描述 | tests/test_label_dialog_qt.py |
| WebEngine 可选 | WebEngine 不可用时 import 路径不崩 | tests/test_optional_webengine.py |
| 日志与诊断 | 日志落盘、diagnostics.zip 脱敏 | tests/test_diagnostics.py |
| ONNX 模型检查 | 子进程安全检查与超时兜底 | tests/test_models/test_model_check.py |
| CLI 子命令 | version/config/diagnostics/convert | tests/test_cli.py |
| 导出与转换 | 多格式转换与 round-trip | tests/test_converter_roundtrip.py |
| 自动标注 | Fake 模型契约、可选真实冒烟 | tests/test_autolabeling_contract.py |
| Chatbot/VQA | 离线 mock、状态机与导入导出 | tests/test_chatbot_offline.py, tests/test_vqa_offline.py |
| 训练向导 | dry-run、启动/停止/错误状态 | tests/test_training_dryrun.py |

