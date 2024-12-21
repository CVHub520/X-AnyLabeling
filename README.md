## 问题记录

1) 加快推理速度(方案: 并行计算, 不显图片)
2) 检验时找出重叠框(原标签中的重叠框, 检测出来的重叠框)
3) 标签不同的框(检测出来的 与 原始标签的 中框重叠, 但签不同的)


## 修改说明

  记录我对这个开源库做的一些修改。

## 一.与导出有关的修改

1) 导出yolo标签时, 对齐到小数点后6位
2) 涉及到的文件: label_converter.py

## 二.与检验有关的修改

1) 增加检测结果与原有标签的iou计算, 并找出: 0)标准, 1)多检, 2)漏检, 3)偏移
2) 转存多检, 漏检, 偏移三种标签和图片
3) 转存时, 将 多检 和 偏移 的框也画在原图片上, 方便人工矫正
4) 将 保留现有标签 改成了 提取疑错标签, 还不能用, 可能需要做本地化处理  
5) 涉及到的文件: label_widget.py, zh_CN.ts

## 三.与GPU有关的修改

1) 查看CUDA和Python版本  
    NVIDIA-SMI 560.94  
	Driver Version: 560.94  
	CUDA Version: 12.6  
	Python 3.12.8  

2) 安装requirements  
   pip install -r requirements-gpu-dev.txt

3) 安装ONNXRuntime-GPU  
   pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

4) 修改配置文件  
   E:\DLCode\X-AnyLabeling\x-anylabeling-win-gpu.spec
   E:\DLCode\X-AnyLabeling\anylabeling\app_info.py

5) 安装pytorch  
   用迅雷从官网下载torch-2.5.1+cu124-cp312-cp312-win_amd64.whl  
   https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl#sha256=3c3f705fb125edbd77f9579fa11a138c56af8968a10fc95834cdd9fdf4f1f1a6

6) 增加版本打印  
   在 anylabeling/app.py 中增加了版本打印

7) 涉及到的文件: app.py, app_info.py, spec文件