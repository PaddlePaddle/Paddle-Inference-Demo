# 使用 GPU 进行预测

**注意：**
1. Config 默认使用 CPU 进行预测，需要通过 `PD_ConfigEnableUseGpu` 来启用 GPU 预测
2. 可以尝试启用 TensorRT 进行 GPU 预测加速

## GPU 设置

API定义如下：

```c
// 启用 GPU 进行预测
// 参数：pd_config                - Config 对象指针
//      memory_pool_init_size_mb - 初始化分配的 GPU 显存，以MB为单位
//      device_id                - 设备 id
// 返回：None
PD_ConfigEnableUseGpu(PD_Config* pd_config, uint64_t memory_pool_init_size_mb, int32_t device_id);

// 禁用 GPU 进行预测
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigDisableGpu(PD_Config* pd_config);

// 判断是否启用 GPU 
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 GPU 
PD_Bool PD_ConfigUseGpu(PD_Config* pd_config);

// 获取 GPU 的device id
// 参数：pd_config - Config 对象指针
// 返回：int32_t -  GPU 的device id
int32_t PD_ConfigGpuDeviceId(PD_Config* pd_config);

// 获取 GPU 的初始显存大小
// 参数：pd_config - Config 对象指针
// 返回：int32_t -  GPU 的初始的显存大小
int32_t PD_ConfigMemoryPoolInitSizeMb(PD_Config* pd_config);

// 初始化显存占总显存的百分比
// 参数：pd_config - Config 对象指针
// 返回：float - 初始的显存占总显存的百分比
float PD_ConfigFractionOfGpuMemoryForPool(PD_Config* pd_config);

// 开启线程流，目前的行为是为每一个线程绑定一个流，在将来该行为可能改变
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigEnableGpuMultiStream(PD_Config* pd_config);

// 判断是否开启线程流
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否是否开启线程流
PD_Bool PD_ConfigThreadLocalStreamEnabled(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 启用 GPU 进行预测 - 初始化 GPU 显存 100 MB, Deivce_ID 为 0
PD_ConfigEnableUseGpu(config, 100, 0);

// 通过 API 获取 GPU 信息
printf("Use GPU is: %s\n", PD_ConfigUseGpu(config) ? "True" : "False"); // True
printf("GPU deivce id is: %d\n", PD_ConfigGpuDeviceId(config));
printf("GPU memory size is: %d\n", PD_ConfigMemoryPoolInitSizeMb(config));
printf("GPU memory frac is: %f\n", PD_ConfigFractionOfGpuMemoryForPool(config));

// 开启线程流
PD_ConfigEnableGpuMultiStream(config);

// 判断是否开启线程流 - True
printf("Thread local stream enabled: %s\n", PD_ConfigThreadLocalStreamEnabled(config) ? "True" : "False");

// 禁用 GPU 进行预测
PD_ConfigDisableGpu(config);

// 通过 API 获取 GPU 信息
printf("Use GPU is: %s\n", PD_ConfigUseGpu(config) ? "True" : "False"); // False

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

## TensorRT 设置

**注意：** 
1. 启用 TensorRT 的前提为已经启用 GPU，否则启用 TensorRT 无法生效
2. 对存在 LoD 信息的模型，如 ERNIE / BERT 等 NLP 模型，必须使用动态 Shape
3. 启用 TensorRT OSS 可以支持更多 plugin，详细参考 [TensorRT OSS](https://news.developer.nvidia.com/nvidia-open-sources-parsers-and-plugins-in-tensorrt/)

更多 TensorRT 详细信息，请参考 [使用Paddle-TensorRT库预测](../../../guides/nv_gpu_infer/gpu_trt_infer)。

API定义如下：
```c
// 启用 TensorRT 进行预测加速
// 参数：pd_config          - Config 对象指针
//      workspace_size     - 指定 TensorRT 在网络编译阶段进行kernel选择时使用的工作空间大小，不影响运
//                           行时显存占用。该值设置过小可能会导致选不到最佳kernel，设置过大时会增加初始
//                           化阶段的显存使用，请根据实际情况调整，建议值256MB
//      max_batch_size     - 设置最大的 batch 大小，运行时 batch 大小不得超过此限定值
//      min_subgraph_size  - Paddle 内 TensorRT 是以子图的形式运行，为了避免性能损失，当 TensorRT 
//                           子图内部节点个数大于 min_subgraph_size 的时候，才会使用 TensorRT 运行
//      precision          - 指定使用 TensorRT 的精度，支持 FP32(kFloat32)，FP16(kHalf)，
//                           Int8(kInt8)
//      use_static         - 若指定为 true，在初次运行程序退出Predictor析构的时候会将 TensorRT 的优
//                           化信息进行序列化到磁盘上。下次运行时直接加载优化的序列化信息而不需要重新生
//                           成，以加速启动时间（需要在同样的硬件和相同 TensorRT 版本的情况下）
//      use_calib_mode     - 若要运行 TensorRT INT8 离线量化校准，需要将此选项设置为 true
// 返回：None
void PD_ConfigEnableTensorRtEngine(PD_Config* pd_config,
                                   int32_t workspace_size,
                                   int32_t max_batch_size,
                                   int32_t min_subgraph_size,
                                   PD_PrecisionType precision,
                                   PD_Bool use_static,
                                   PD_Bool use_calib_mode);

// 判断是否启用 TensorRT 
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 TensorRT
PD_Bool PD_ConfigTensorRtEngineEnabled(PD_Config* pd_config);

// 设置 TensorRT 的动态 Shape
// 参数：pd_config               - Config 对象指针
//      tensor_num              - TensorRT 子图支持动态 shape 的 Tensor 数量
//      tensor_name             - TensorRT 子图支持动态 shape 的 Tensor 名称
//      shapes_num              - TensorRT 子图支持动态 shape 的 Tensor 对应的 shape 的长度
//      min_shape               - TensorRT 子图支持动态 shape 的 Tensor 对应的最小 shape
//      max_shape               - TensorRT 子图支持动态 shape 的 Tensor 对应的最大 shape
//      optim_shape             - TensorRT 子图支持动态 shape 的 Tensor 对应的最优 shape
//      disable_trt_plugin_fp16 - 设置 TensorRT 的 plugin 不在 fp16 精度下运行
// 返回：None
void PD_ConfigSetTrtDynamicShapeInfo(PD_Config* pd_config,
                                     size_t tensor_num,
                                     const char** tensor_name,
                                     size_t* shapes_num,
                                     int32_t** min_shape,
                                     int32_t** max_shape,
                                     int32_t** optim_shape,
                                     PD_Bool disable_trt_plugin_fp16);

// 启用 TensorRT OSS 进行预测加速
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigEnableTensorRtOSS(PD_Config* pd_config);

// 判断是否启用 TensorRT OSS
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 TensorRT OSS
PD_Bool PD_ConfigTensorRtOssEnabled(PD_Config* pd_config);

// 启用 TensorRT DLA 进行预测加速
// 参数：pd_config - Config 对象指针
//      dla_core - DLA 设备的 id，可选 0，1，...，DLA 设备总数 - 1
// 返回：None
void PD_ConfigEnableTensorRtDla(PD_Config* pd_config, int32_t dla_core);

// 判断是否已经开启 TensorRT DLA 加速
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否已开启 TensorRT DLA 加速
PD_Bool PD_ConfigTensorRtDlaEnabled(PD_Config* pd_config);
```

代码示例 (1)：使用 TensorRT FP32 / FP16 / INT8 进行预测

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 启用 GPU 进行预测 - 初始化 GPU 显存 100MB, Deivce_ID 为 0
PD_ConfigEnableUseGpu(config, 100, 0);

// 启用 TensorRT 进行预测加速 - FP32
PD_ConfigEnableTensorRtEngine(config, 1 << 20, 1, 3,
                              PD_PRECISION_FLOAT32, FALSE, FALSE);

// 启用 TensorRT 进行预测加速 - FP16
PD_ConfigEnableTensorRtEngine(config, 1 << 20, 1, 3,
                              PD_PRECISION_HALF, FALSE, FALSE);

// 启用 TensorRT 进行预测加速 - Int8
PD_ConfigEnableTensorRtEngine(config, 1 << 20, 1, 3,
                              PD_PRECISION_INT8, FALSE, FALSE);

// 通过 API 获取 TensorRT 启用结果 - True
printf("Enable TensorRT is: %s\n", PD_ConfigTensorRtEngineEnabled(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```
代码示例 (2)：使用 TensorRT 动态 Shape 进行预测

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径
const char* model_path  = "./model/inference.pdmodel";  
const char* params_path = "./model/inference.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);

// 启用 GPU 进行预测 - 初始化 GPU 显存 100MB, Deivce_ID 为 0
PD_ConfigEnableUseGpu(config, 100, 0);

// 启用 TensorRT 进行预测加速 - Int8
PD_ConfigEnableTensorRtEngine(config, 1 << 30, 1, 2, PD_PRECISION_INT8, FALSE, TRUE);

// 设置模型输入的动态 Shape 范围
const char * tensor_name[1] = {"image"};
size_t shapes_num[1] = {4};
int32_t image_min_shape[4] = {1, 1, 3, 3};
int32_t image_max_shape[4] = {1, 1, 10, 10};
int32_t image_opt_shape[4] = {1, 1, 3, 3};
int32_t* min_shape[1] = {image_min_shape};
int32_t* max_shape[1] = {image_max_shape};
int32_t* opt_shape[1] = {image_opt_shape};
PD_ConfigSetTrtDynamicShapeInfo(config, 1, tensor_name, shapes_num, 
                                min_shape, max_shape, opt_shape, FALSE); 

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

代码示例 (3)：使用 TensorRT OSS 进行预测

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径，这里为 Combined 模型
const char* model_path  = "./model/ernie.pdmodel";  
const char* params_path = "./model/ernie.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);

// 启用 GPU 进行预测 - 初始化 GPU 显存 100MB, Deivce_ID 为 0
PD_ConfigEnableUseGpu(config, 100, 0);

// 启用 TensorRT 进行预测加速 - FP32
PD_ConfigEnableTensorRtEngine(config, 1 << 20, 1, 3, PD_PRECISION_FLOAT32, FALSE, TRUE);

// 启用 TensorRT OSS 进行预测加速
PD_ConfigEnableTensorRtOSS(config);

// 通过 API 获取 TensorRT OSS 启用结果 - True
printf("Enable TensorRT is: %s\n", PD_ConfigTensorRtOssEnabled(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```
