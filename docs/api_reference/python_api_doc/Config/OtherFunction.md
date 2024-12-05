# 启用内存优化

API定义如下：

```python
# 开启内存/显存复用，具体降低内存效果取决于模型结构。
# 参数：None
# 返回：None
paddle.inference.Config.enable_memory_optim()
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 开启 CPU 显存优化
config.enable_memory_optim()

# 启用 GPU 进行预测
config.enable_use_gpu(100, 0)
# 开启 GPU 显存优化
config.enable_memory_optim()
```

# 设置缓存路径

**注意：** 如果当前使用的为 TensorRT INT8 且设置从内存中加载模型，则必须通过 `SetOptimCacheDir` 来设置缓存路径。

API定义如下：

```python
# 设置缓存路径
# 参数：opt_cache_dir - 缓存路径
# 返回：None
paddle.inference.Config.set_optim_cache_dir(opt_cache_dir: str)
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 设置缓存路径
config.set_optim_cache_dir("./OptimCacheDir")
```

# Profile 设置

API定义如下：

```python
# 打开 Profile，运行结束后会打印所有 OP 的耗时占比。
# 参数：None
# 返回：None
paddle.inference.Config.enable_profile()
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 打开 Profile
config.enable_profile()
```

执行预测之后输出的 Profile 的结果如下：

```bash
------------------------->     Profiling Report     <-------------------------

Place: CPU
Time unit: ms
Sorted by total time in descending order in the same thread

-------------------------     Overhead Summary      -------------------------

Total time: 1085.33
  Computation time       Total: 1066.24     Ratio: 98.2411%
  Framework overhead     Total: 19.0902     Ratio: 1.75893%

-------------------------     GpuMemCpy Summary     -------------------------

GpuMemcpy                Calls: 0           Total: 0           Ratio: 0%

-------------------------       Event Summary       -------------------------

Event                            Calls       Total       Min.        Max.        Ave.        Ratio.
thread0::conv2d                  210         319.734     0.815591    6.51648     1.52254     0.294595
thread0::load                    137         284.596     0.114216    258.715     2.07735     0.26222
thread0::depthwise_conv2d        195         266.241     0.955945    2.47858     1.36534     0.245308
thread0::elementwise_add         210         122.969     0.133106    2.15806     0.585568    0.113301
thread0::relu                    405         56.1807     0.021081    0.585079    0.138718    0.0517635
thread0::batch_norm              195         25.8073     0.044304    0.33896     0.132345    0.0237783
thread0::fc                      15          7.13856     0.451674    0.714895    0.475904    0.0065773
thread0::pool2d                  15          1.48296     0.09054     0.145702    0.0988637   0.00136636
thread0::softmax                 15          0.941837    0.032175    0.460156    0.0627891   0.000867786
thread0::scale                   15          0.240771    0.013394    0.030727    0.0160514   0.000221841
```

# Log 设置

API定义如下：

```python
# 去除 Paddle Inference 运行中的 LOG
# 参数：None
# 返回：None
paddle.inference.Config.disable_glog_info()

# 判断是否禁用 LOG
# 参数：None
# 返回：bool - 是否禁用 LOG
paddle.inference.Config.glog_info_disabled()
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 去除 Paddle Inference 运行中的 LOG
config.disable_glog_info()

# 判断是否禁用 LOG - true
print("GLOG INFO is: {}".format(config.glog_info_disabled()))
```

# 查看config配置

API定义如下：

```python
# 返回config的配置信息
# 参数：None
# 返回：string - config配置信息
paddle.inference.Config.summary()
```

调用summary()的输出如下所示：
```
+-------------------------------+----------------------------------+
| Option                        | Value                            |
+-------------------------------+----------------------------------+
| model_dir                     | ./inference_pass/TRTFlattenTest/ |
+-------------------------------+----------------------------------+
| cpu_math_thread               | 1                                |
| enable_mkdlnn                 | false                            |
| mkldnn_cache_capacity         | 10                               |
| use_openvino                  | false                            |
| openvino_inference_precision  | fp32                            |
+-------------------------------+----------------------------------+
| use_gpu                       | true                             |
| gpu_device_id                 | 0                                |
| memory_pool_init_size         | 100MB                            |
| thread_local_stream           | false                            |
| use_tensorrt                  | true                             |
| tensorrt_precision_mode       | fp32                             |
| tensorrt_workspace_size       | 1073741824                       |
| tensorrt_max_batch_size       | 32                               |
| tensorrt_min_subgraph_size    | 0                                |
| tensorrt_use_static_engine    | false                            |
| tensorrt_use_calib_mode       | false                            |
| tensorrt_enable_dynamic_shape | false                            |
| tensorrt_use_oss              | true                             |
| tensorrt_use_dla              | false                            |
+-------------------------------+----------------------------------+
| use_xpu                       | false                            |
+-------------------------------+----------------------------------+
| ir_optim                      | true                             |
| ir_debug                      | false                            |
| memory_optim                  | false                            |
| enable_profile                | false                            |
| enable_log                    | true                             |
+-------------------------------+----------------------------------+
```
