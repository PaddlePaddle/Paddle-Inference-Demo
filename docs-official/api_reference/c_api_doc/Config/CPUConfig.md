# 使用 CPU 进行预测

**注意：**
1. 在 CPU 型号允许的情况下，进行预测库下载或编译试尽量使用带 AVX 和 MKL 的版本
2. 可以尝试使用 Intel 的 MKLDNN 进行 CPU 预测加速，默认 CPU 不启用 MKLDNN
3. 在 CPU 可用核心数足够时，可以通过设置 `PD_ConfigSetCpuMathLibraryNumThreads` 将线程数调高一些，默认线程数为 1

## CPU 设置

API定义如下：

```c
// 设置 CPU 加速库计算线程数
// 参数：config - Config 对象指针
//      cpu_math_library_num_threads - CPU 加速库计算线程数
// 返回：None
void PD_ConfigSetCpuMathLibraryNumThreads(PD_Config* pd_config, int32_t cpu_math_library_num_threads);

// 获取 CPU 加速库计算线程数
// 参数：pd_config - Config 对象指针
// 返回：int32_t - CPU 加速库计算线程数
int32_t PD_ConfigGetCpuMathLibraryNumThreads(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置 CPU 加速库线程数为 10
PD_ConfigSetCpuMathLibraryNumThreads(config, 10);

// 通过 API 获取 CPU 信息
printf("CPU Math Lib Thread Num is: %d\n", PD_ConfigGetCpuMathLibraryNumThreads(config));

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

## MKLDNN 设置

**注意：** 
1. 启用 MKLDNN 的前提为已经使用 CPU 进行预测，否则启用 MKLDNN 无法生效
2. 启用 MKLDNN BF16 要求 CPU 型号可以支持 AVX512，否则无法启用 MKLDNN BF16
3. `PD_ConfigSetMkldnnCacheCapacity` 请参考 <a class="reference external" href="https://github.com/PaddlePaddle/docs/blob/923d0dc161e54b424b8b163b6ff72c73ef10a43f/docs/design/mkldnn/caching/caching.md">MKLDNN cache设计文档</a>

API定义如下：

```c
// 启用 MKLDNN 进行预测加速
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigEnableMKLDNN(PD_Config* pd_config);

// 判断是否启用 MKLDNN 
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 MKLDNN
PD_Bool PD_ConfigMkldnnEnabled(PD_Config* pd_config);

// 设置 MKLDNN 针对不同输入 shape 的 cache 容量大小
// 参数：pd_config - Config 对象指针
//      capacity  - cache 容量大小
// 返回：None
void PD_ConfigSetMkldnnCacheCapacity(PD_Config* pd_config, int32_t capacity);

// 指定使用 MKLDNN 加速的 OP 列表
// 参数：pd_config - Config 对象指针
//      ops_num   - 使用 MKLDNN 加速的 OP 数量
//      op_list   - 使用 MKLDNN 加速的 OP 列表
// 返回：None
void PD_ConfigSetMkldnnOp(PD_Config* pd_config, size_t ops_num, const char** op_list);

// 启用 MKLDNN BFLOAT16
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigEnableMkldnnBfloat16(PD_Config* pd_config);

// 判断是否启用 MKLDNN BFLOAT16
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 MKLDNN BFLOAT16
PD_Bool PD_ConfigMkldnnBfloat16Enabled(PD_Config* pd_config);

// 指定使用 MKLDNN BFLOAT16 加速的 OP 列表
// 参数：pd_config - Config 对象指针
//      ops_num   - 使用 MKLDNN BFLOAT16 加速的 OP 数量
//      op_list   - 使用 MKLDNN BFLOAT16 加速的 OP 列表
// 返回：None
PD_ConfigSetBfloat16Op(PD_Config* pd_config, size_t ops_num, const char** op_list);
```

代码示例 (1)：使用 MKLDNN 进行预测

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 启用 MKLDNN 进行预测
PD_ConfigEnableMKLDNN(config);

// 通过 API 获取 MKLDNN 启用结果 - True
printf("Enable MKLDNN is: %s\n", PD_MkldnnEnabled(config) ? "True" : "False");

// 设置 MKLDNN 的 cache 容量大小
PD_ConfigSetMkldnnCacheCapacity(config, 1);

// 设置启用 MKLDNN 进行加速的 OP 列表
const char* op_list[3] = {"softmax", "elementwise_add", "relu"};
PD_ConfigSetMkldnnOp(config, 3, op_list);

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

代码示例 (2)：使用 MKLDNN BFLOAT16 进行预测

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 启用 MKLDNN 进行预测
PD_ConfigEnableMKLDNN(config);

// 启用 MKLDNN BFLOAT16 进行预测
PD_EnableMkldnnBfloat16(config);

// 设置启用 MKLDNN BFLOAT16 进行加速的 OP 列表
const char* op_list[1] = {"conv2d"};
PD_ConfigSetBfloat16Op(config, 1, op_list);

// 通过 API 获取 MKLDNN 启用结果 - True
printf("Enable MKLDNN BF16 is: %s\n", PD_ConfigMkldnnBfloat16Enabled(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```