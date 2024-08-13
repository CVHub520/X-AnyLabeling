# 常见问题解答 (FAQ)

这里是关于项目的常见问题的列表及其答案。(持续更新中)


## 安装与运行相关问题

> 启动时报错：`qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.`

**答：** 可参考#541、#496。

> GPU版本运行时闪退？

**答：** 可参考#500。

## GUI交互相关问题

> 应用启动时，首次点击无效？

**答：** 此问题暂时无解，首次点击可视为正式触发交互生效。 

> 问：

## 模型相关问题

> yolo 系列模型如何仅识别特定的类别？

**答：** 可通过在配置文件中添加 `filter_classes`，具体可参考此[文档](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/zh_cn/custom_model.md#%E5%8A%A0%E8%BD%BD%E5%B7%B2%E9%80%82%E9%85%8D%E7%9A%84%E7%94%A8%E6%88%B7%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B)。

> in paint assert len(self.points) in [1, 2, 4] AssertionError.

**答：** 可参考#491。

> 加载自定义模型报错？

**答：** 可参考以下步骤解决：

1. 检查配置文件中的定义的类别与模型支持的类别列表是否一致。
2. 使用 [netron](https://netron.app/) 工具比较自定义模型与官方对应的内置模型输入输出节点维度是否一致。

> 运行完没有输出结果？

**答：** 可参考#536。

> 使用 Grounding DINO 模型进行 GPU 推理时报错：Error in predict_shapes: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Expand node. Name:'Expand_33526' Status Message: Expand_33526: left operand cannot broadcast on dim 2 LeftShape: {1,900,4}, RightShape: {1,900,256}

**答：** 可参考#389。

> Missing config : num_masks

**答：** 可参考#515。

> AttributeError: 'int' object has no attribute 'replace'

**答：** 查看配置文件是否有定义纯数字标签。请注意在定义以**纯数字**命名的标签名称时，请务必将其加上单引号 `''`

## 标签导入导出相关问题

> Export mask error: "imageWidth"

**答：** 可参考#477。

> operands could not be broadcast together with shapes (0,) (1,2)

**答：** 可参考#492。

> 导出关键点标签时报错：int0 argument must be a string, a byteslike object or a number, not 'NoneType'

**答：** `group_id`字段缺失，请确保每个矩形框和关键点都有对应的群组编号。

## 其它

略。
