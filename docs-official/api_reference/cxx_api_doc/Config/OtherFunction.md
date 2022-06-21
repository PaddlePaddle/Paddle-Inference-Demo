# 启用内存优化

API定义如下：

```c++
// 开启内存/显存复用，具体降低内存效果取决于模型结构
// 参数：None
// 返回：None
void EnableMemoryOptim();

// 判断是否开启内存/显存复用
// 参数：None
// 返回：bool - 是否开启内/显存复用
bool enable_memory_optim() const;
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config("./mobilenet.pdmodel", "./mobilenet.pdiparams");

// 开启 CPU 显存优化
config.EnableMemoryOptim();
// 通过 API 获取 CPU 是否已经开启显存优化 - true
std::cout << "CPU Mem Optim is: " << config.enable_memory_optim() << std::endl;

// 启用 GPU 进行预测
config.EnableUseGpu(100, 0);
// 开启 GPU 显存优化
config.EnableMemoryOptim();
// 通过 API 获取 GPU 是否已经开启显存优化 - true
std::cout << "GPU Mem Optim is: " << config.enable_memory_optim() << std::endl;
```

# 设置缓存路径

**注意：** 
如果当前使用的为 TensorRT INT8 且设置从内存中加载模型，则必须通过 `SetOptimCacheDir` 来设置缓存路径。


API定义如下：

```c++
// 设置缓存路径
// 参数：opt_cache_dir - 缓存路径
// 返回：None
void SetOptimCacheDir(const std::string& opt_cache_dir);
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config();

// 设置缓存路径
config.SetOptimCacheDir("./model/OptimCacheDir");
```

# FC Padding

在使用MKL时，启动此配置项可能会对模型推理性能有提升（[参考PR描述](https://github.com/PaddlePaddle/Paddle/pull/20972)）。

API定义如下：

```c++
// 禁用 FC Padding
// 参数：None
// 返回：None
void DisableFCPadding();

// 判断是否启用 FC Padding
// 参数：None
// 返回：bool - 是否启用 FC Padding
bool use_fc_padding() const;
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config("./mobilenet.pdmodel", "./mobilenet.iparams");

// 禁用 FC Padding
config.DisableFCPadding();

// 通过 API 获取是否禁用 FC Padding - false
std::cout << "Disable FC Padding is: " << config.use_fc_padding() << std::endl;
```

# Profile 设置

API定义如下：

```c++
// 打开 Profile，运行结束后会打印所有 OP 的耗时占比。
// 参数：None
// 返回：None
void EnableProfile();

// 判断是否开启 Profile
// 参数：None
// 返回：bool - 是否开启 Profile
bool profile_enabled() const;
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config("./mobilenet.pdmodel", "./mobilenet.iparams");

// 打开 Profile
config.EnableProfile();

// 判断是否开启 Profile - true
std::cout << "Profile is: " << config.profile_enabled() << std::endl;
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

```c++
// 去除 Paddle Inference 运行中的 LOG
// 参数：None
// 返回：None
void DisableGlogInfo();

// 判断是否禁用 LOG
// 参数：None
// 返回：bool - 是否禁用 LOG
bool glog_info_disabled() const;
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config("./mobilenet.pdmodel", "./mobilenet.iparams");

// 去除 Paddle Inference 运行中的 LOG
config.DisableGlogInfo();

// 判断是否禁用 LOG - true
std::cout << "GLOG INFO is: " << config.glog_info_disabled() << std::endl;
```

# 查看config配置

API定义如下：

```c++
// 返回 config 的配置信息
// 参数：None
// 返回：string - config 配置信息
std::string Summary();
```

调用 Summary() 的输出如下所示：
```
+-------------------------------+----------------------------------+
| Option                        | Value                            |
+-------------------------------+----------------------------------+
| model_dir                     | ./inference_pass/TRTFlattenTest/ |
+-------------------------------+----------------------------------+
| cpu_math_thread               | 1                                |
| enable_mkdlnn                 | false                            |
| mkldnn_cache_capacity         | 10                               |
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
