
# 使用 XPU 进行预测

API定义如下：

```python
# 启用 XPU 进行预测
# 参数：l3_workspace_size - l3 cache 分配的显存大小，最大为16M
# 参数：locked - 分配的L3 cache是否可以锁定。如果为false，表示不锁定L3 cache，则分配的L3 cache可以多个模型共享，多个共享L3 cache的模型在卡上将顺序执行
# 参数：autotune - 是否对模型中的conv算子进行autotune。如果为true，则在第一次执行到某个维度的conv算子时，将自动搜索更优的算法，用以提升后续相同维度的conv算子的性能
# 参数：autotune_file - 指定autotune文件路径。如果指定autotune_file，则使用文件中指定的算法，不再重新进行autotune
# 参数：precision - multi_encoder的计算精度
# 参数：adaptive_seqlen - multi_encoder的输入是否可变长
# 返回：None
paddle.inference.Config.enable_xpu(l3_workspace_size: int = 0xfffc00,
                                   locked: bool = False,
                                   autotune: bool = True,
                                   autotune_file: string = "",
                                   precision: string = "int16",
                                   adaptive_seqlen: bool = False)
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 启用 XPU，并设置 l3 cache 大小为 10M
config.enable_xpu(10 * 1024 * 1024)
```
