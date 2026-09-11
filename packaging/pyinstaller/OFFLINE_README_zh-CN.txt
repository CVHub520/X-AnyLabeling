X-AnyLabeling v4.0.1 Windows 离线便携版

使用方法：
1. 必须先完整解压 ZIP，不能直接在压缩包内运行。
2. EXE 与 _internal 文件夹必须始终放在同一目录，不要只复制 EXE。
3. 双击目录中的 X-AnyLabeling EXE 启动。

运行环境：
- Windows 10/11 64 位。
- 无需安装 Python、PyQt6、OpenCV 或 Microsoft VC++ Redistributable。
- CUDA12 版已携带 CUDA 12、cuDNN 9 和 ONNX Runtime CUDA 运行库，
  但仍需要兼容的 NVIDIA 显卡及 NVIDIA 显卡驱动。

离线自动标注：
- 标注、像素边缘贴合和格式导出可直接离线使用。
- 自动标注模型权重不在程序包内。需要把已经下载的模型文件或模型缓存
  一起复制到离线电脑，再从本地加载模型。

故障排查：
- 如果提示 QtCore、Qt6Core.dll 或 qwindows.dll 缺失，先确认 _internal
  文件夹没有被删除、隔离或漏拷贝，并检查杀毒软件的隔离记录。
- 不支持 Windows 7/8；Qt 6 需要受支持的 Windows 10/11 系统。
