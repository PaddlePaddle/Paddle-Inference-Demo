# 启用内存优化

API定义如下：

```c
// 开启内存/显存复用，具体降低内存效果取决于模型结构
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigEnableMemoryOptim(PD_Config* pd_config);

// 判断是否开启内存/显存复用
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否开启内/显存复用
PD_Bool PD_ConfigMemoryOptimEnabled(PD_Config* pd_config);
```
代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 开启 CPU 内存优化
PD_ConfigEnableMemoryOptim(config);

// 通过 API 获取 CPU 是否已经开启内存优化 - True
printf("CPU Mem Optim is: %s\n", PD_ConfigMemoryOptimEnabled(config) ? "True" : "False");

// 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
PD_ConfigEnableUseGpu(config, 100, 0);

// 开启 GPU 显存优化
PD_ConfigEnableMemoryOptim(config);

// 通过 API 获取 GPU 是否已经开启显存优化 - True
printf("GPU Mem Optim is: %s\n", PD_ConfigMemoryOptimEnabled(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

# 设置缓存路径

**注意：** 如果当前使用的为 TensorRT INT8 且设置从内存中加载模型，则必须通过 `PD_ConfigSetOptimCacheDir` 来设置缓存路径。


API定义如下：

```c
// 设置缓存路径
// 参数：pd_config     - Config 对象指针
//      opt_cache_dir - 缓存路径
// 返回：None
void PD_ConfigSetOptimCacheDir(PD_Config* pd_config, const char* opt_cache_dir);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径，这里为 Combined 模型
const char* model_path  = "./model/inference.pdmodel";  
const char* params_path = "./model/inference.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);

// 设置缓存路径
PD_ConfigSetOptimCacheDir(config,"./model/OptimCacheDir");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

# FC Padding

API定义如下：

```c++
// 禁用 FC Padding
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigDisableFCPadding(PD_Config* pd_config);

// 判断是否启用 FC Padding
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 FC Padding
PD_Bool PD_ConfigUseFcPadding(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 禁用 FC Padding
PD_ConfigDisableFCPadding(config);

// 通过 API 获取是否启用 FC Padding - False
printf("FC Padding is: %s\n", PD_ConfigUseFcPadding(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

# Profile 设置

API定义如下：

```c++
// 打开 Profile，运行结束后会打印所有 OP 的耗时占比。
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigEnableProfile(PD_Config* pd_config);

// 判断是否开启 Profile
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否开启 Profile
PD_Bool PD_ConfigProfileEnabled(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 打开 Profile
PD_ConfigEnableProfile(config);

// 通过 API 获取是否启用Profile - True
printf("Profile is: %s\n", PD_ConfigProfileEnabled(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
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
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigDisableGlogInfo(PD_Config* pd_config);

// 判断是否禁用 LOG
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否禁用 LOG
PD_Bool PD_ConfigGlogInfoDisabled(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 去除 Paddle Inference 运行中的 LOG
PD_ConfigDisableGlogInfo(config);

// 通过 API 获取是否启用LOG - False
printf("GLOG INFO is: %s\n", PD_ConfigGlogInfoDisabled(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

# 查看config配置

API定义如下：

```c++
// 返回config的配置信息
// 参数：pd_config - Config 对象指针
// 返回：const char* - config配置信息，注意用户需释放该指针。
const char* PD_ConfigSummary(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

PD_Cstr* summary = PD_ConfigSummary(config);

printf("summary is %s\n", summary->data);

// 销毁 summary 对象
PD_CstrDestroy(summary);
// 销毁 Config 对象
PD_ConfigDestroy(config)
```
