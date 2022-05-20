
# 使用 ONNXRuntime 进行预测

API定义如下：

```c++
// 启用 ONNXRuntime 进行预测
// 参数：None
// 返回：None
void EnableONNXRuntime();

// 禁用 ONNXRuntime 进行预测
// 参数：None
// 返回：None
void DisableONNXRuntime();

// 判断是否启用 ONNXRuntime 
// 参数：None
// 返回：bool - 是否启用 ONNXRuntime 
bool use_onnxruntime() const;

// 启用 ONNXRuntime 预测时开启优化
// 参数：None
// 返回：None
void EnableORTOptimization();
```

ONNXRuntime设置代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_model_file, FLAGS_params_file);

// 启用 ONNXRuntime
config.EnableONNXRuntime();
// 通过 API 获取 ONNXRuntime 信息
std::cout << "Use ONNXRuntime is: " << config.use_onnxruntime() << std::endl; // true

// 开启ONNXRuntime优化
config.EnableORTOptimization();

// 设置 ONNXRuntime 算子计算线程数为 10
config.SetCpuMathLibraryNumThreads(10);

// 禁用 ONNXRuntime 进行预测
config.DisableONNXRuntime();
// 通过 API 获取 ONNXRuntime 信息
std::cout << "Use ONNXRuntime is: " << config.use_onnxruntime() << std::endl; // false
```
