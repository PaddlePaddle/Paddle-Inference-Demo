
# 使用 ONNXRuntime 进行预测

API定义如下：

```c
// 启用 ONNXRuntime 进行预测
// 参数：None
// 返回：None
void PD_ConfigEnableONNXRuntime(PD_Config* pd_config);

// 禁用 ONNXRuntime 进行预测
// 参数：None
// 返回：None
void PD_ConfigDisableONNXRuntime(PD_Config* pd_config);

// 判断是否启用 ONNXRuntime 
// 参数：None
// 返回：bool - 是否启用 ONNXRuntime 
PD_Bool PD_ConfigONNXRuntimeEnabled(PD_Config* pd_config);

// 启用 ONNXRuntime 预测时开启优化
// 参数：None
// 返回：None
void PD_ConfigEnableORTOptimization(PD_Config* pd_config);
```

ONNXRuntime设置代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 启用 ONNXRuntime
PD_ConfigEnableONNXRuntime(config);
// 通过 API 获取 ONNXRuntime 信息
printf("Use ONNXRuntime is: %s\n", PD_ConfigONNXRuntimeEnabled(config) ? "True" : "False"); // True

// 开启ONNXRuntime优化
PD_ConfigEnableORTOptimization(config);

// 设置 ONNXRuntime 算子计算线程数为 10
PD_ConfigSetCpuMathLibraryNumThreads(config, 10);

// 禁用 ONNXRuntime 进行预测
PD_ConfigDisableONNXRuntime(config);
// 通过 API 获取 ONNXRuntime 信息
printf("Use ONNXRuntime is: %s\n", PD_ConfigONNXRuntimeEnabled(config) ? "True" : "False"); // False
```
