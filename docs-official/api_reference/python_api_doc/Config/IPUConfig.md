
# 使用 IPU 进行预测

API定义如下：

```python
# 启用 IPU 进行预测
# 参数：ipu_device_num - 所需要的 IPU 个数
# 参数：ipu_micro_batch_size - 计算图输入的 batch size，用于根据输入 batch size 进行全图 Tensor shape 推导，仅在动态输入 batch size 的情况生效
# 参数：ipu_enable_pipelining - 使能 IPU 间数据流水
# 参数：ipu_batches_per_step - 在使能数据流水的条件下，指定每次跑多少 batch 的数据，如果关闭数据流水，该值应设置为 1
# 返回：None
paddle.inference.Config.enable_ipu(ipu_device_num = 1,
                                   ipu_micro_batch_size = 1,
                                   ipu_enable_pipelining = False,
                                   ipu_batches_per_step = 1)



# 配置 IPU 构图参数
# 参数：ipu_enable_fp16 - 使能 float16 模式，将 float32 计算图转换为 float16 计算图
# 参数：ipu_replica_num - 设置实例个数，举例 ipu_device_num = 2，表示单个实例需要 2 个 IPU 运行，设置ipu_replica_num = 8，表示总共有 8 个相同实例，所以总共需要 16 个 IPU
# 参数：ipu_available_memory_proportion - 设置 matmul / conv OP 可使用的内存比例，取值 (0.0, 1.0]，比例越高，计算性能越好
# 参数：ipu_enable_half_partial - matmul OP 中间结果以 float16 存储于片上
# 返回：None
paddle.inference.Config.set_ipu_config(ipu_enable_fp16 = False,
                                       ipu_replica_num = 1,
                                       ipu_available_memory_proportion = 1.0,
                                       ipu_enable_half_partial = False)



# 配置 IPU Custom Ops 和 Patterns
# 参数：ipu_custom_ops_info - 设置 Paddle Op 和 IPU Custom Op 信息，需要给定 Paddle Op name，IPU Custom Op name，Op Domain 和 Op Version。例如：[["custom_relu", "Relu", "custom.ops", "1"]]
# 参数：ipu_custom_patterns - 开启或关闭特定 IPU pattern，需要给定 Pattern name 和 Pattern 状态。例如：{"AccumulatePriorityPattern", false}
# 返回：None
paddle.inference.Config.set_ipu_custom_info(ipu_custom_ops_info = None,
                                            ipu_custom_patterns = None)



# 从文件载入 IPU 配置信息
# 参数：config_path - 指定文件路径
# 返回：None
paddle.inference.Config.load_ipu_config(config_path)
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 启用 IPU，并设置单个实例所需要的 IPU 个数为 1
config.enable_ipu(1)

# 使能 float16 模式
config.set_ipu_config(True)
```

```text
# IPU 配置文件示例如下：
ipu_device_num,1
ipu_micro_batch_size,1
ipu_enable_fp16,false
ipu_custom_ops_info,[[custom_relu, Relu, custom.ops, 1]]
```
