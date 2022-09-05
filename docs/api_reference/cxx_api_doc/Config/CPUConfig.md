# 使用 CPU 进行预测

**注意：**
1. 在 CPU 型号允许的情况下，进行预测库下载或编译试尽量使用带 AVX 和 MKL 的版本
2. 可以尝试使用 Intel 的 MKLDNN 进行 CPU 预测加速，默认 CPU 不启用 MKLDNN
3. 在 CPU 可用核心数足够时，可以通过设置 `SetCpuMathLibraryNumThreads` 将线程数调高一些，默认线程数为 1

## CPU 设置

API定义如下：

```c++
// 设置 CPU Blas 库计算线程数
// 参数：cpu_math_library_num_threads - blas库计算线程数
// 返回：None
void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);

// 获取 CPU Blas 库计算线程数
// 参数：None
// 返回：int - cpu blas库计算线程数。
int cpu_math_library_num_threads() const;
```

代码示例：

```c++
// 创建默认 Config 对象
paddle_infer::Config config();

// 设置 CPU Blas 库线程数为 10
config.SetCpuMathLibraryNumThreads(10);

// 通过 API 获取 CPU 信息
int num_thread = config.cpu_math_library_num_threads();
std::cout << "CPU blas thread number is: " << num_thread << std::endl; // 10
```

## MKLDNN 设置

**注意：** 
1. 启用 MKLDNN 的前提为已经使用 CPU 进行预测，否则启用 MKLDNN 无法生效
2. 启用 MKLDNN BF16 要求 CPU 型号可以支持 AVX512，否则无法启用 MKLDNN BF16
3. `SetMkldnnCacheCapacity` 请参考 <a class="reference external" href="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md">MKLDNN cache设计文档</a>

API定义如下：

```c++
// 启用 MKLDNN 进行预测加速
// 参数：None
// 返回：None
void EnableMKLDNN();

// 判断是否启用 MKLDNN 
// 参数：None
// 返回：bool - 是否启用 MKLDNN
bool mkldnn_enabled() const;

// 设置 MKLDNN 针对不同输入 shape 的 cache 容量大小
// 参数：int - cache 容量大小
// 返回：None
void SetMkldnnCacheCapacity(int capacity);

// 指定使用 MKLDNN 加速的 OP 列表
// 参数：std::unordered_set<std::string> - 使用 MKLDNN 加速的 OP 列表
// 返回：None
void SetMKLDNNOp(std::unordered_set<std::string> op_list);

// 启用 MKLDNN BFLOAT16
// 参数：None
// 返回：None
void EnableMkldnnBfloat16();

// 设置新版本量化模型的 calibration file 路径
// 参数：calibration_file_path - 新版本量化模型的 calibration file 路径
// 返回：None
void SetCalibrationFilePath(const std::string& calibration_file_path);

// 启用 MKLDNN INT8
// 参数：op_list - 使用 MKLDNN INT8 加速的 OP 列表
// 返回：None
void EnableMkldnnInt8(const std::unordered_set<std::string>& op_list);

// 判断是否启用 MKLDNN INT8
// 参数：None
// 返回：bool - 是否启用 MKLDNN INT8
bool mkldnn_int8_enabled() const;

// 判断是否启用 MKLDNN BFLOAT16
// 参数：None
// 返回：bool - 是否启用 MKLDNN BFLOAT16
bool mkldnn_bfloat16_enabled() const;

// 指定使用 MKLDNN BFLOAT16 加速的 OP 列表
// 参数：std::unordered_set<std::string> - 使用 MKLDNN BFLOAT16 加速的 OP 列表
// 返回：None
void SetBfloat16Op(std::unordered_set<std::string> op_list);
```

代码示例 (1)：使用 MKLDNN 进行预测

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_infer_model + "/mobilenet");

// 启用 MKLDNN 进行预测
config.EnableMKLDNN();
// 通过 API 获取 MKLDNN 启用结果 - true
std::cout << "Enable MKLDNN is: " << config.mkldnn_enabled() << std::endl;

// 设置 MKLDNN 的 cache 容量大小
config.SetMkldnnCacheCapacity(1);

// 设置启用 MKLDNN 进行加速的 OP 列表
std::unordered_set<std::string> op_list = {"softmax", "elementwise_add", "relu"};
config.SetMKLDNNOp(op_list);
```

代码示例 (2)：使用 MKLDNN BFLOAT16 进行预测

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_infer_model + "/mobilenet");

// 启用 MKLDNN 进行预测
config.EnableMKLDNN();

// 启用 MKLDNN BFLOAT16 进行预测
config.EnableMkldnnBfloat16();
// 设置启用 MKLDNN BFLOAT16 的 OP 列表
config.SetBfloat16Op({"conv2d"});

// 通过 API 获取 MKLDNN BFLOAT16 启用结果 - true
std::cout << "Enable MKLDNN BF16 is: " << config.mkldnn_bfloat16_enabled() << std::endl;
```

代码示例 (3)：使用 MKLDNN INT8 进行预测

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_infer_model + "/mobilenet");

// 启用 MKLDNN 进行预测
config.EnableMKLDNN();

// 设置新版本量化模型的量化标定文件路径
config.SetCalibrationFilePath(FLAGS_infer_model + "/mobilenet/calibration_table.txt")

// 启用 MKLDNN INT8 进行预测
config.EnableMkldnnInt8();

// 通过 API 获取 MKLDNN INT8 启用结果 - true
std::cout << "Enable MKLDNN INT8 is: " << config.mkldnn_int8_enabled() << std::endl;
```
