# 常见问题解答 (FAQ)


### 安装与运行相关问题

<details>
<summary>Q: 启动时报错：`qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.` </summary>

可参考[#541](https://github.com/CVHub520/X-AnyLabeling/issues/541)、[#496](https://github.com/CVHub520/X-AnyLabeling/issues/496)。
</details>

<details>
<summary>Q: GPU版本运行时闪退？</summary>

可参考[#500](https://github.com/CVHub520/X-AnyLabeling/issues/500)。
</details>


### 界面交互相关问题

<details>
<summary>Q: 应用启动时，首次点击无效？</summary>

此问题暂时无解。 
</details>


### 模型相关问题

<details>
<summary>Q: 下载完的模型每次重新启动应用时都被自动删除重新下载</summary>

注意模型路径不得有中文字符，否则会有异常。（[#600](https://github.com/CVHub520/X-AnyLabeling/issues/600)）
</details>

<details>
<summary>Q: AI模型推理时如何识别特定的类别？</summary>

当前仅支持部分模型设置此选项。具体地，以 yolo 系列模型为例，用户可通过在配置文件中添加 `filter_classes`，具体可参考此[文档](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/zh_cn/custom_model.md#%E5%8A%A0%E8%BD%BD%E5%B7%B2%E9%80%82%E9%85%8D%E7%9A%84%E7%94%A8%E6%88%B7%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B)。
</details>

<details>
<summary>Q: in paint assert len(self.points) in [1, 2, 4] AssertionError.</summary>

可参考[#491](https://github.com/CVHub520/X-AnyLabeling/issues/491)。
</details>

<details>
<summary>Q: 加载自定义模型报错？</summary>

可参考以下步骤解决：

1. 检查配置文件中的定义的类别与模型支持的类别列表是否一致。
2. 使用 [netron](https://netron.app/) 工具比较自定义模型与官方对应的内置模型输入输出节点维度是否一致。
</details>

<details>
<summary>Q: 运行完没有输出结果？</summary>

可参考[#536](https://github.com/CVHub520/X-AnyLabeling/issues/536)。
</details>

<details>
<summary>Q: 使用 Grounding DINO 模型进行 GPU 推理时报错：

```shell
Error in predict_shapes: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Expand node. Name:'Expand_33526' Status Message: Expand_33526: left operand cannot broadcast on dim 2 LeftShape: {1,900,4}, RightShape: {1,900,256}
```
</summary>

可参考[#389](https://github.com/CVHub520/X-AnyLabeling/issues/389)。
</details>

<details>
<summary>Q: Missing config : num_masks</summary>

可参考[#515](https://github.com/CVHub520/X-AnyLabeling/issues/515)。
</details>

<details>
<summary>Q: AttributeError: 'int' object has no attribute 'replace'</summary>
查看配置文件是否有定义纯数字标签。请注意在定义以**纯数字**命名的标签名称时，请务必将其加上单引号 `''`
</details>

<details>
<summary>Q: Unsupported model IR version: 10, max supported IR version: 9</summary>

ONNX Runtime 版本过低，请更新：

```shell
# 安装 CPU 版本
pip install --upgrade onnxruntime

# 安装 GPU 版本
pip install --upgrade onnxruntime-gpu
```
</details>

<details>
<summary>Q: Your model ir_version is higher than the checker's.</summary>
onnx 版本过低，请更新：

```shell
pip install --upgrade onnx
```
</details>


### 标签导入导出相关问题

<details>
<summary>Q: Export mask error: "imageWidth"</summary>

可参考[#477](https://github.com/CVHub520/X-AnyLabeling/issues/477)。
</details>

<details>
<summary>Q: operands could not be broadcast together with shapes (0,) (1,2)</summary>

可参考[#492](https://github.com/CVHub520/X-AnyLabeling/issues/492)。
</details>

<details>
<summary>Q: 导出关键点标签时报错：int0 argument must be a string, a byteslike object or a number, not 'NoneType'</summary>

`group_id`字段缺失，请确保每个矩形框和关键点都有对应的群组编号。
</details>
