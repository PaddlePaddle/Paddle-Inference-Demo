# 使用 CPU 进行预测

**注意：**
1. 在 CPU 型号允许的情况下，进行预测库下载或编译试尽量使用带 AVX 和 MKL 的版本
2. 可以尝试使用 Intel 的 MKLDNN 进行 CPU 预测加速，默认 CPU 不启用 MKLDNN
3. 在 CPU 可用核心数足够时，可以通过设置 `PD_SetCpuMathLibraryNumThreads` 将线程数调高一些，默认线程数为 1

## CPU 设置

API定义如下：

```c
// 设置 CPU Blas 库计算线程数
// 参数：config - AnalysisConfig 对象指针
//      cpu_math_library_num_threads - blas库计算线程数
// 返回：None
void PD_SetCpuMathLibraryNumThreads(PD_AnalysisConfig* config, 
                                    int cpu_math_library_num_threads)

// 获取 CPU Blas 库计算线程数
// 参数：config - AnalysisConfig 对象指针
// 返回：int - cpu blas 库计算线程数
int PD_CpuMathLibraryNumThreads(const PD_AnalysisConfig* config);
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置 CPU Blas 库线程数为 10
PD_SetCpuMathLibraryNumThreads(config, 10);

// 通过 API 获取 CPU 信息
printf("CPU Math Lib Thread Num is: %d\n", PD_CpuMathLibraryNumThreads(config));
```

## MKLDNN 设置

**注意：** 
1. 启用 MKLDNN 的前提为已经使用 CPU 进行预测，否则启用 MKLDNN 无法生效
2. 启用 MKLDNN BF16 要求 CPU 型号可以支持 AVX512，否则无法启用 MKLDNN BF16
3. `PD_SetMkldnnCacheCapacity` 请参考 <a class="reference external" href="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md">MKLDNN cache设计文档</a>

API定义如下：

```c
// 启用 MKLDNN 进行预测加速
// 参数：config - AnalysisConfig 对象指针
// 返回：None
void PD_EnableMKLDNN(PD_AnalysisConfig* config);

// 判断是否启用 MKLDNN 
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 是否启用 MKLDNN
bool PD_MkldnnEnabled(const PD_AnalysisConfig* config);

// 设置 MKLDNN 针对不同输入 shape 的 cache 容量大小
// 参数：config - AnalysisConfig 对象指针
//      capacity - cache 容量大小
// 返回：None
void PD_SetMkldnnCacheCapacity(PD_AnalysisConfig* config, int capacity);

// 启用 MKLDNN BFLOAT16
// 参数：config - AnalysisConfig 对象指针
// 返回：None
void PD_EnableMkldnnBfloat16(PD_AnalysisConfig* config);

// 判断是否启用 MKLDNN BFLOAT16
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 是否启用 MKLDNN BFLOAT16
bool PD_MkldnnBfloat16Enabled(const PD_AnalysisConfig* config);
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 启用 MKLDNN 进行预测
PD_EnableMKLDNN(config);

// 通过 API 获取 MKLDNN 启用结果 - true
printf("Enable MKLDNN is: %s\n", PD_MkldnnEnabled(config) ? "True" : "False");

// 启用 MKLDNN BFLOAT16 进行预测
PD_EnableMkldnnBfloat16(config);

// 通过 API 获取 MKLDNN BFLOAT16 启用结果
// 如果当前CPU支持AVX512，则返回 true, 否则返回 false
printf("Enable MKLDNN BF16 is: %s\n", PD_MkldnnBfloat16Enabled(config) ? "True" : "False");

// 设置 MKLDNN 的 cache 容量大小
PD_SetMkldnnCacheCapacity(config, 1);
```