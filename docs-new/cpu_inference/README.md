# CPU上部署模型

## 简介

Paddle Inference在CPU上有：原生CPU、MKLDNN和ONNX Runtime后端三种推理方式。还支持量化和低精度推理，加快模型推理速度。

本文档主要介绍使用Paddle Inference原生CPU、MKLDNN和ONNX Runtime后端进行推理时，如何调用API进行配置。详细代码请参考:[X86 Linux上预测部署示例](../demo_tutorial/x86_linux_demo)和[X86 Windows上预测部署示例](../demo_tutorial/x86_windows_demo)

## 使用原生CPU推理

原生CPU推理使用Paddle Inference原生的高性能Kernel进行计算，不依赖第三方加速库。使用部署简单，但性能无优势。

### 配置文件开发说明

使用原生CPU推理，不同的地方只在配置文件。

C++示例：
```c++
// 创建默认配置对象
paddle_infer::Config config;

// 设置预测模型路径
config.SetModel(FLAGS_model_file, FLAGS_params_file);

// 设置 CPU Blas 库线程数为 10
config.SetCpuMathLibraryNumThreads(10);

// 通过 API 获取 CPU 信息
int num_thread = config.cpu_math_library_num_threads();
```

python示例:
```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config()

# 设置模型的文件夹路径
config.set_model("model")

# 设置 CPU Blas 库线程数为 10
config.set_cpu_math_library_num_threads(10)

# 通过 API 获取 CPU 信息 - 10
print(config.cpu_math_library_num_threads())
```

## 使用MKLDNN推理

MKLDNN是Intel发布的开源的深度学习软件包，Paddle Inference除了有大量的算子支持MKLDNN加速，还针对MKLDNN进行了图优化。

### 配置文件开发说明

使用MKLDNN推理，只需修改配置文件。

C++示例：
```c++
// 创建默认配置对象
paddle_infer::Config config;

// 设置预测模型路径
config.SetModel(FLAGS_model_file, FLAGS_params_file);

// 启用 MKLDNN 进行预测
config.EnableMKLDNN();

// 通过 API 获取 MKLDNN 启用结果 - true
std::cout << "Enable MKLDNN is: " << config.mkldnn_enabled() << std::endl;

// 设置 MKLDNN 的 cache 容量大小
config.SetMkldnnCacheCapacity(1);
```

python示例:
```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config()

# 设置模型的文件夹路径
config.set_model("model")

# 启用 MKLDNN 进行预测
config.enable_mkldnn()

# 通过 API 获取 MKLDNN 启用结果 - true
print(config.mkldnn_enabled())

# 设置 MKLDNN 的 cache 容量大小
config.set_mkldnn_cache_capacity(1)
```

## 使用ONNX Runtime后端推理

ONNX Runtime是一个跨平台的机器学习模型加速器，对ONNX标准支持最全最广泛的的推理引擎。Paddle Inference从2.3开始新增了ONNX Runtime后端，能将Paddle模型运行在该后端上。

### 配置文件开发说明

使用ONNX Runtime推理，只需修改配置文件。

C++示例：
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
```

python示例:
```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("model.pdmodel", "model.pdiparams")

# 启用 ONNXRuntime 进行预测
config.enable_onnxruntime()

# 通过 API 获取 ONNXRuntime 信息
print("Use ONNXRuntime is: {}".format(config.onnxruntime_enabled())) # True

# 开启ONNXRuntime优化
config.enable_ort_optimization();

// 设置 ONNXRuntime 算子计算线程数为 10
config.set_cpu_math_library_num_threads(10)

# 禁用 ONNXRuntime 进行预测
config.DisableONNXRuntime();
# 通过 API 获取 ONNXRuntime 信息

print("Use ONNXRuntime is: {}".format(config.onnxruntime_enabled())) # false
```

