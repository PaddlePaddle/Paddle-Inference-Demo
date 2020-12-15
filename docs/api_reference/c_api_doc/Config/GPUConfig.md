# 使用 GPU 进行预测

**注意：**
1. AnalysisConfig 默认使用 CPU 进行预测，需要通过 `EnableUseGpu` 来启用 GPU 预测
2. 可以尝试启用 CUDNN 和 TensorRT 进行 GPU 预测加速

## GPU 设置

API定义如下：

```c
// 启用 GPU 进行预测
// 参数：config - AnalysisConfig 对象指针
//      memory_pool_init_size_mb - 初始化分配的gpu显存，以MB为单位
//      device_id - 设备id
// 返回：None
void PD_EnableUseGpu(PD_AnalysisConfig* config, int memory_pool_init_size_mb, int device_id);

// 禁用 GPU 进行预测
// 参数：config - AnalysisConfig 对象指针
// 返回：None
void PD_DisableGpu(PD_AnalysisConfig* config);

// 判断是否启用 GPU 
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 是否启用 GPU 
bool PD_UseGpu(const PD_AnalysisConfig* config);

// 获取 GPU 的device id
// 参数：config - AnalysisConfig 对象指针
// 返回：int -  GPU 的device id
int PD_GpuDeviceId(const PD_AnalysisConfig* config);

// 获取 GPU 的初始显存大小
// 参数：config - AnalysisConfig 对象指针
// 返回：int -  GPU 的初始的显存大小
int PD_MemoryPoolInitSizeMb(const PD_AnalysisConfig* config);

// 初始化显存占总显存的百分比
// 参数：config - AnalysisConfig 对象指针
// 返回：float - 初始的显存占总显存的百分比
float PD_FractionOfGpuMemoryForPool(const PD_AnalysisConfig* config);
```

GPU设置代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
PD_EnableUseGpu(config, 100, 0);

// 通过 API 获取 GPU 信息
printf("Use GPU is: %s\n", PD_UseGpu(config) ? "True" : "False"); // True
printf("GPU deivce id is: %d\n", PD_GpuDeviceId(config));
printf("GPU memory size is: %d\n", PD_MemoryPoolInitSizeMb(config));
printf("GPU memory frac is: %f\n", PD_FractionOfGpuMemoryForPool(config));

// 禁用 GPU 进行预测
PD_DisableGpu(config);

// 通过 API 获取 GPU 信息
printf("Use GPU is: %s\n", PD_UseGpu(config) ? "True" : "False"); // False
```

## CUDNN 设置

**注意：** 启用 CUDNN 的前提为已经启用 GPU，否则启用 CUDNN 无法生效。

API定义如下：

```c
// 启用 CUDNN 进行预测加速
// 参数：config - AnalysisConfig 对象指针
// 返回：None
void PD_EnableCUDNN(PD_AnalysisConfig* config);

// 判断是否启用 CUDNN 
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 是否启用 CUDNN
bool PD_CudnnEnabled(const PD_AnalysisConfig* config);
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
PD_EnableUseGpu(config, 100, 0);

// 启用 CUDNN 进行预测加速
PD_EnableCUDNN(config);

// 通过 API 获取 CUDNN 启用结果 - True
printf("Enable GPU is: %s\n", PD_CudnnEnabled(config) ? "True" : "False");

// 禁用 GPU 进行预测
PD_DisableGpu(config);
// 启用 CUDNN 进行预测加速 - 因为 GPU 被禁用，因此 CUDNN 启用不生效
PD_EnableCUDNN(config);
// 通过 API 获取 CUDNN 启用结果 - False
printf("Enable CUDNN is: %s\n", PD_CudnnEnabled(config) ? "True" : "False");
```

## TensorRT 设置

**注意：** 
1. 启用 TensorRT 的前提为已经启用 GPU，否则启用 TensorRT 无法生效
2. 对存在LoD信息的模型，如Bert, Ernie等NLP模型，必须使用动态 Shape
3. 启用 TensorRT OSS 可以支持更多 plugin，详细参考 [TensorRT OSS](https://news.developer.nvidia.com/nvidia-open-sources-parsers-and-plugins-in-tensorrt/)

更多 TensorRT 详细信息，请参考 [使用Paddle-TensorRT库预测](../../../optimize/paddle_trt)。

API定义如下：

```c
// 启用 TensorRT 进行预测加速
// 参数：config - AnalysisConfig 对象指针
//      workspace_size     - 指定 TensorRT 使用的工作空间大小
//      max_batch_size     - 设置最大的 batch 大小，运行时 batch 大小不得超过此限定值
//      min_subgraph_size  - Paddle-TRT 是以子图的形式运行，为了避免性能损失，当子图内部节点个数
//                           大于 min_subgraph_size 的时候，才会使用 Paddle-TRT 运行
//      precision          - 指定使用 TRT 的精度，支持 FP32(kFloat32)，FP16(kHalf)，Int8(kInt8)
//      use_static         - 若指定为 true，在初次运行程序的时候会将 TRT 的优化信息进行序列化到磁盘上，
//                           下次运行时直接加载优化的序列化信息而不需要重新生成
//      use_calib_mode     - 若要运行 Paddle-TRT INT8 离线量化校准，需要将此选项设置为 true
// 返回：None
PD_EnableTensorRtEngine(PD_AnalysisConfig* config, int workspace_size, int max_batch_size,
                        int min_subgraph_size, Precision precision, bool use_static,
                        bool use_calib_mode);

// 判断是否启用 TensorRT 
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 是否启用 TensorRT
bool PD_TensorrtEngineEnabled(const PD_AnalysisConfig* config);
```

代码示例 (1)：使用 TensorRT FP32 / FP16 / INT8 进行预测

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
PD_EnableUseGpu(config, 100, 0);

// 启用 TensorRT 进行预测加速 - FP32
PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, kFloat32, false, false);

// 启用 TensorRT 进行预测加速 - FP16
PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, kHalf, false, false);

// 启用 TensorRT 进行预测加速 - Int8
PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, kInt8, false, false);

// 通过 API 获取 TensorRT 启用结果 - true
printf("Enable TensorRT is: %s\n", PD_TensorrtEngineEnabled(config) ? "True" : "False");
```
