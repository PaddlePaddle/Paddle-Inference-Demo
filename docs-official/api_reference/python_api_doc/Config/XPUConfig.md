
# 使用 XPU 进行预测

API定义如下：

```python
# 启用 XPU 进行预测
# 参数：l3_workspace_size - L3 cache 分配的显存大小，最大为 16 MB
# 参数：locked            - 分配的 L3 cache 是否可以锁定。如果为 False，表示不锁定 L3 cache，则分配的 L3 cache 可以多个模型共享，多个共享 L3 cache 的模型在卡上将顺序执行
# 参数：autotune          - 是否对模型中的 conv 算子进行 autotune。如果为 True，则在第一次执行到某个维度的conv 算子时，将自动搜索更优的算法，用以提升后续相同维度的 conv 算子的性能
# 参数：autotune_file     - 指定 autotune 文件路径。如果指定 autotune_file，则使用文件中指定的算法，不再重新进行 autotune
# 参数：precision         - multi_encoder 的计算精度
# 参数：adaptive_seqlen   - multi_encoder 的输入是否可变长
# 返回：None
paddle.inference.Config.enable_xpu(l3_workspace_size: int = 0xfffc00,
                                   locked: bool = False,
                                   autotune: bool = True,
                                   autotune_file: string = "",
                                   precision: string = "int16",
                                   adaptive_seqlen: bool = False)

# 设置 XPU 配置
# 参数：quant_post_dynamic_weight_bits - 动态量化时，量化权重的位数。可选值为：-1，8，16。默认值为 -1，表示不进行配置，使用推荐的精度。
# 参数：quant_post_dynamic_op_types - 动态量化时，需要进行量化操作的算子类型。
paddle.inference.Config.set_xpu_config(quant_post_dynamic_weight_bits: int = -1,
                                       quant_post_dynamic_op_types: List[str] = [])
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 启用 XPU，并设置 L3 cache 大小为 10 MB
config.enable_xpu(10 * 1024 * 1024)

# 设置默认 XPU 推理配置：使用推荐的量化配置
config.set_xpu_config()
```
