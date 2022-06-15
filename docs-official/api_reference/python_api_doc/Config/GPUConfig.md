# 使用 GPU 进行预测

**注意：**
1. Config 默认使用 CPU 进行预测，需要通过 `EnableUseGpu` 来启用 GPU 预测
2. 可以尝试启用 CUDNN 和 TensorRT 进行 GPU 预测加速

## GPU 设置

API定义如下：

```python
# 启用 GPU 进行预测
# 参数：memory_pool_init_size_mb - 初始化分配的gpu显存，以MB为单位
#      device_id - 设备id
# 返回：None
paddle.inference.Config.enable_use_gpu(memory_pool_init_size_mb: int, device_id: int)

# 禁用 GPU 进行预测
# 参数：None
# 返回：None
paddle.inference.Config.disable_gpu()

# 判断是否启用 GPU 
# 参数：None
# 返回：bool - 是否启用 GPU 
paddle.inference.Config.use_gpu()

# 获取 GPU 的device id
# 参数：None
# 返回：int -  GPU 的device id
paddle.inference.Config.gpu_device_id()

# 获取 GPU 的初始显存大小
# 参数：None
# 返回：int -  GPU 的初始的显存大小
paddle.inference.Config.memory_pool_init_size_mb()

# 初始化显存占总显存的百分比
# 参数：None
# 返回：float - 初始的显存占总显存的百分比
paddle.inference.Config.fraction_of_gpu_memory_for_pool()

# 启用 GPU FP16 计算精度进行预测
# 参数：op_list - 保持 FP32 计算精度算子名单
# 返回：None
paddle.inference.Config.exp_enable_use_gpu_fp16(op_list: Set[str])
```

GPU设置代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet.pdmodel", "./mobilenet.pdiparams")

# 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
config.enable_use_gpu(100, 0)
# 通过 API 获取 GPU 信息
print("Use GPU is: {}".format(config.use_gpu())) # True
print("Init mem size is: {}".format(config.memory_pool_init_size_mb())) # 100
print("Init mem frac is: {}".format(config.fraction_of_gpu_memory_for_pool())) # 0.003
print("GPU device id is: {}".format(config.gpu_device_id())) # 0

# 禁用 GPU 进行预测
config.disable_gpu()
# 通过 API 获取 GPU 信息
print("Use GPU is: {}".format(config.use_gpu())) # False

# 启用 GPU FP16 计算精度进行预测
config.enable_use_gpu(100, 0);
config.exp_enable_use_gpu_fp16();
```

## TensorRT 设置

**注意：** 
1. 启用 TensorRT 的前提为已经启用 GPU，否则启用 TensorRT 无法生效
2. 对存在 LoD 信息的模型，如 BERT、ERNIE 等 NLP 模型，必须使用动态 Shape
3. 启用 TensorRT OSS 可以支持更多 plugin，详细参考 [TensorRT OSS](https://news.developer.nvidia.com/nvidia-open-sources-parsers-and-plugins-in-tensorrt/)

更多 TensorRT 详细信息，请参考 [使用Paddle-TensorRT库预测](../../../optimize/paddle_trt)。

API定义如下：

```python
# 启用 TensorRT 进行预测加速
# 参数：workspace_size     - 指定 TensorRT 在网络编译阶段进行kernel选择时使用的工作空间大小，不影响运
#                           行时显存占用。该值设置过小可能会导致选不到最佳kernel，设置过大时会增加初始
#                           化阶段的显存使用，请根据实际情况调整，建议值256MB
#      max_batch_size     - 设置最大的 batch 大小，运行时 batch 大小不得超过此限定值
#      min_subgraph_size  - Paddle 内 TensorRT 是以子图的形式运行，为了避免性能损失，当 TensorRT 
#                           子图内部节点个数大于 min_subgraph_size 的时候，才会使用 TensorRT 运行
#      precision          - 指定使用 TensorRT 的精度，支持 FP32(kFloat32)，FP16(kHalf)，Int8(kInt8)
#      use_static         - 若指定为 true，在初次运行程序退出Predictor析构的时候会将 TensorRT 的优
#                           化信息进行序列化到磁盘上。下次运行时直接加载优化的序列化信息而不需要重新生
#                           成，以加速启动时间（需要在同样的硬件和相同 TensorRT 版本的情况下）
#      use_calib_mode     - 若要运行 TensorRT INT8 离线量化校准，需要将此选项设置为 True
# 返回：None
paddle.inference.Config.enable_tensorrt_engine(workspace_size: int = 1 << 20,
                                               max_batch_size: int,
                                               min_subgraph_size: int,
                                               precision_mode: PrecisionType,
                                               use_static: bool,
                                               use_calib_mode: bool)

# 判断是否启用 TensorRT 
# 参数：None
# 返回：bool - 是否启用 TensorRT
paddle.inference.Config.tensorrt_engine_enabled()

# 设置 TensorRT 的动态 Shape
# 参数：min_input_shape          - TensorRT 子图支持动态 shape 的最小 shape，推理时输入 shape 的任何
#                                 维度均不能小于该项配置
#      max_input_shape          - TensorRT 子图支持动态 shape 的最大 shape，推理是输入 shape 的任何
#                                 维度均不能大于该项配置
#      optim_input_shape        - TensorRT 子图支持动态 shape 的最优 shape，TensorRT 在初始化选
#                                 kernel 阶段以此项配置的 shape 下的性能表现作为选择依据
#      disable_trt_plugin_fp16  - 设置 TensorRT 的 plugin 不在 fp16 精度下运行
# 返回：None
paddle.inference.Config.set_trt_dynamic_shape_info(min_input_shape: Dict[str, List[int]]={}, 
                                                   max_input_shape: Dict[str, List[int]]={}, 
                                                   optim_input_shape: Dict[str, List[int]]={}, 
                                                   disable_trt_plugin_fp16: bool=False)

#
# TensorRT 动态 shape 的自动推导
# 参数： shape_range_info_path  - 统计生成的 shape 信息存储文件路径
#       allow_build_at_runtime - 是否开启运行时重建 TensorRT 引擎功能，当设置为 true 时，输入 shape 
#                                超过 tune 范围时会触发 TensorRT 重建。当设置为 false 时，输入 shape
#                                超过 tune 范围时会引起推理出错
# 返回：None
paddle.inference.Config.enable_tuned_tensorrt_dynamic_shape(
                                     shape_range_info_path: str,
                                     allow_build_at_runtime: bool=True)

# 启用 TensorRT OSS 进行 ERNIE / BERT 预测加速（原理介绍 https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/ernie-varlen ）
# 参数：None
# 返回：None
paddle.inference.Config.enable_tensorrt_oss()

# 判断是否启用 TensorRT OSS
# 参数：None
# 返回：bool - 是否启用 TensorRT OSS
paddle.inference.Config.tensorrt_oss_enabled()

# 启用 TensorRT DLA 进行预测加速
# 参数：dla_core - DLA 设备的 id，可选 0，1，...，DLA 设备总数 - 1
# 返回：None
paddle.inference.Config.enable_tensorrt_dla(dla_core: int = 0)

# 判断是否已经开启 TensorRT DLA 加速
# 参数：None
# 返回：bool - 是否已开启 TensorRT DLA 加速
paddle.inference.Config.tensorrt_dla_enabled()
```

代码示例 (1)：使用 TensorRT FP32 / FP16 / INT8 进行预测

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet.pdmodel", "./mobilenet.pdiparams")

# 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
config.enable_use_gpu(100, 0)

# 启用 TensorRT 进行预测加速 - FP32
config.enable_tensorrt_engine(workspace_size = 1 << 28, 
                              max_batch_size = 1, 
                              min_subgraph_size = 3, 
                              precision_mode=paddle_infer.PrecisionType.Float32, 
                              use_static = False, use_calib_mode = False)
# 通过 API 获取 TensorRT 启用结果 - true
print("Enable TensorRT is: {}".format(config.tensorrt_engine_enabled()))


# 启用 TensorRT 进行预测加速 - FP16
config.enable_tensorrt_engine(workspace_size = 1 << 28, 
                              max_batch_size = 1, 
                              min_subgraph_size = 3, 
                              precision_mode=paddle_infer.PrecisionType.Half, 
                              use_static = False, use_calib_mode = False)
# 通过 API 获取 TensorRT 启用结果 - true
print("Enable TensorRT is: {}".format(config.tensorrt_engine_enabled()))

# 启用 TensorRT 进行预测加速 - Int8
config.enable_tensorrt_engine(workspace_size = 1 << 28, 
                              max_batch_size = 1, 
                              min_subgraph_size = 3, 
                              precision_mode=paddle_infer.PrecisionType.Int8, 
                              use_static = False, use_calib_mode = False)
# 通过 API 获取 TensorRT 启用结果 - true
print("Enable TensorRT is: {}".format(config.tensorrt_engine_enabled()))
```

代码示例 (2)：使用 TensorRT 动态 Shape 进行预测

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet.pdmodel", "./mobilenet.pdiparams")

# 启用 GPU 进行预测 - 初始化 GPU 显存 100 M, Deivce_ID 为 0
config.enable_use_gpu(100, 0)

# 启用 TensorRT 进行预测加速 - Int8
config.enable_tensorrt_engine(workspace_size = 1 << 29, 
                              max_batch_size = 1, 
                              min_subgraph_size = 1, 
                              precision_mode=paddle_infer.PrecisionType.Int8, 
                              use_static = False, use_calib_mode = True)

# 设置 TensorRT 的动态 Shape
config.set_trt_dynamic_shape_info(min_input_shape={"image": [1, 1, 3, 3]},
                                  max_input_shape={"image": [1, 1, 10, 10]},
                                  optim_input_shape={"image": [1, 1, 3, 3]})
```

代码示例 (3)：使用 TensorRT OSS 进行预测

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./ernie.pdmodel", "./ernie.pdiparams")

# 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
config.enable_use_gpu(100, 0)

# 启用 TensorRT 进行预测加速
config.enable_tensorrt_engine()

# 启用 TensorRT OSS 进行预测加速
config.enable_tensorrt_oss()

# 通过 API 获取 TensorRT OSS 启用结果 - true
print("Enable TensorRT OSS is: {}".format(config.tensorrt_oss_enabled()))
```
