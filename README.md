# 问题记录

1. 将各个步骤的耗时, 追加到状态栏的文件名后面
2. 让推理时也可以切子图, 方便标签的验证和矫正
3. 导出路径, 改成当前目录的同级目录(train和val)
4. 自动标定时, 框还在画, 能否不在画框啊
5. 

# 修改说明

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

## 三.与保存子图的修改

1. 保存子图时, 文件名中应该包含原图像的文件名, 方便后续的子图和原图的对应
2. 保存子图时, 不支持中文路径的问题
3. 涉及到的文件: image_dialog.py

## 四.推理加速的修改

1. 增加了一些耗时打印, 通过这些打印发现: 耗时主要在load_file()和to_rgb_cv_img()两个函数中  
2. 通过测试发现: 推理时, 发现根本不需要调用load_file函数, 只需执行其中的五句就行了
3. 通过测试发现: 推理时, 发现根本不需要调用progress_dialog.setValue()函数
4. 保存label时, 若没有图片数据, 则基于图片路径读取宽和高; 否则, 使用图片数据的宽高
5. 修改LabelFile类, 让其支持只加载标签数据, 不加载图片数据
6. 修改save_labels函数, 若没有图片数据, 则基于图片路径读取宽和高; 否则, 使用图片数据的宽高
7. 自动标注时, 不绘制目标框 和 侧边栏中的复选框, 以加快速度
8. 涉及到的文件: label_widget.py, yolo.py, canvas.py

run_all_images									
	show_progress_dialog_and_process
		process_next_image(递归)											138ms
			load_file											88ms	
			model_manager.predict_shapes()						45ms
					predict_shapes        
						to_rgb_cv_img	 				 30ms
						preprocess						   5ms
						inference						   9ms
						postprocess						   0ms
					self.new_auto_labeling_result.emit
					== AutoLabelingWidget的new_shapes_from_auto_labeling方法, 计算IOU并做判定
						new_shapes_from_auto_labeling() 
							set_dirty()	按要求保存label
								save_labels(label_file)
									label_file.save()
			progress_dialog.setValue()							3.7ms

## 五.与GPU有关的修改

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