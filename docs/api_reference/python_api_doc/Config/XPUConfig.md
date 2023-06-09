
# 使用 XPU 进行预测

API定义如下：

```python
# 启用 XPU 进行预测
# 参数：l3_size - l3 cache 分配的显存大小。注：昆仑1上最大为 16773120 Byte，昆仑2上最大为 67104768 Byte
# 参数：l3_locked - 分配的L3 cache是否可以锁定。如果为false，表示不锁定L3 cache，则分配的L3 cache可以多个模型共享，多个共享L3 cache的模型在卡上将顺序执行
# 参数：conv_autotune - 是否对模型中的conv算子进行autotune。如果为true，则在第一次执行到某个维度的conv算子时，将自动搜索更优的算法，用以提升后续相同维度的conv算子的性能
# 参数：conv_autotune_file - 指定autotune文件路径。如果指定autotune_file，则使用文件中指定的算法，不再重新进行autotune
# 参数：transformer_encoder_precision - multi_encoder的计算精度
# 参数：transformer_encoder_adaptive_seqlen - multi_encoder的输入是否可变长
# 参数：enable_multi_stream - 是否启用多流推理，如果启动，将自动创建新的流用于推理
# 返回：None
# 备注：此接口仅用于启动 xpu 推理。详细的 xpu 配置参数请使用 SetXpuConfig 接口进行设置
paddle.inference.Config.enable_xpu(l3_size: int = 0xfffc00,
                                   l3_locked: bool = False,
                                   conv_autotune: bool = True,
                                   conv_autotune_file: string = "",
                                   transformer_encoder_precision: string = "int16",
                                   transformer_encoder_adaptive_seqlen: bool = False,
                                   enable_multi_stream: bool = False)

# XpuConfig 类定义，输入为 None
'''
可配置的成员变量及默认值如下：
# 选择几号卡进行推理
device_id: int = 0

# 可用的 L3 大小
# 昆仑1设备上最大的 L3 大小为 16773120 Byte
# 昆仑2设备上最大的 L3 大小为 67104768 Byte
l3_size: int = 0
# 用于进行 L3 aututune 的大小
# 如果 l3_autotune_size 为 0，则不开启 l3 autotune 功能
# 备注: 剩余的 L3 大小 (l3_size - l3_autotune_size) 将被 kernel 使用（paddle/xdnn kernel 共享剩余的 l3）
l3_autotune_size: int = 0

# conv autotune 等级
# 如果 conv_autotune_level 为 0，则不开启 conv aututune 功能
# 备注：目前仅使用 Paddle-Lite 推理时生效
conv_autotune_level: int = 0
# 从 conv_autotune_file 读取初始的 conv aututune 信息
# 备注：目前仅使用 Paddle-Lite 推理时生效
conv_autotune_file: string = ""
# 是否将新的 conv aututune 信息写会到 conv_autotune_file
# 备注：目前仅使用 Paddle-Lite 推理时生效
conv_autotune_file_writeback: bool = False

# fc autotune 等级
# 如果 fc_autotune_level 为 0，则不开启 fc aututune 功能
# 备注：目前仅使用 Paddle-Lite 推理时生效
fc_autotune_level: int = 0
# 从 fc_autotune_file 读取初始的 fc aututune 信息
# 备注：目前仅使用 Paddle-Lite 推理时生效
fc_autotune_file: string = ""
# 是否将新的 fc aututune 信息写会到 fc_autotune_file
# 备注：目前仅使用 Paddle-Lite 推理时生效
fc_autotune_file_writeback: bool = False

# gemm 计算精度。可选值为：0（int8）、1（int16）、2（int31）
# 备注：gemm_compute_precision 对量化模型中的量化算子不生效
# 备注：目前仅使用 Paddle-Lite 推理时生效
gemm_compute_precision: int = 1
# 对 transformer 结构中的 softmax 使用什么样的优化策略。可选择值为：0，1，2
# 备注：目前仅使用 Paddle-Lite 推理时生效
transformer_softmax_optimize_level: int = 0
# 是否在 transformer encoder 结构中使用可变长序列优化
# 备注：目前仅使用 Paddle-Lite 推理时生效
transformer_encoder_adaptive_seqlen: bool = True

# 在静态离线量化推理中，将 gelu 输出的最大阈值限制为 quant_post_static_gelu_out_threshold
# 备注：目前仅使用 Paddle-Lite 推理时生效
quant_post_static_gelu_out_threshold: float = 0.f
# 在动态在线量化推理中，处理激活值的方式
# 如果使用昆仑1推理，可选值为：0(per_tensor)，1(per_batch)，2(per_head)
# 如果使用昆仑2推理，可选值为：0(per_tensor)，1(every_16)
# 备注：目前仅使用 Paddle-Lite 推理时生效
quant_post_dynamic_activation_method: int = 0
# 在动态离线量化中，将权重数据预处理为哪种精度。可选值为：0(int8)，1(int16)，2(float)
# 备注：目前仅使用 Paddle-Inference 推理时生效
quant_post_dynamic_weight_precision: int = 1
quant_post_dynamic_op_types: List[string] = []
'''
class paddle.inference.XpuConfig()

# 设置 XPU 推理配置参数
# 参数：config - xpu 可用的配置参数，详见 XpuConfig 定义
# 返回：None
paddle.inference.Config.set_xpu_config(xpu_config)
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 启用 XPU
config.enable_xpu()

# 设置 xpu l3 size 为昆仑2上可以使用的最大值
xpu_config = paddle_infer.XpuConfig()
xpu_config.l3_size = 67104768
config.set_xpu_config(xpu_config)
```
