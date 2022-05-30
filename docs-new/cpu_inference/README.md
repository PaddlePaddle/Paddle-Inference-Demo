# CPU上部署模型<!-- omit in toc -->

<!-- omit in toc -->
## 目录
- [CPU原生推理](#cpu原生推理)
- [MKLDNN推理加速](#mkldnn推理加速)
- [ONNX Runtime推理](#onnx-runtime推理)

<!-- omit in toc -->
## 简介

Paddle Inference在CPU上有：原生CPU、MKLDNN和ONNX Runtime后端三种推理方式。还支持量化和低精度推理，加快模型推理速度。

本文档主要介绍使用Paddle Inference原生CPU、MKLDNN和ONNX Runtime后端进行推理时，如何调用API进行配置。详细代码请参考:[X86 Linux上预测部署示例](../demo_tutorial/x86_linux_demo)和[X86 Windows上预测部署示例](../demo_tutorial/x86_windows_demo)

## CPU原生推理

原生CPU推理在推理时，使用飞桨核心框架的标准OP实现进行推理计算，不依赖第三方计算库，推理时也无需额外配置。

<!-- omit in toc -->
### 配置文件开发说明

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
config.set_model("model.pdmodel", "model.pdiparam")

# 设置 CPU Blas 库线程数为 10
config.set_cpu_math_library_num_threads(10)

# 通过 API 获取 CPU 信息 - 10
print(config.cpu_math_library_num_threads())
```

## MKLDNN推理加速

MKLDNN(现OneDNN)是由英特尔开发的开源深度学习软件包，支持神经网络在CPU上的高性能计算，在Paddle Inference中可通过一行配置打开MKLDNN加速。

<!-- omit in toc -->
### 配置文件开发说明

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
config.set_model("model.pdmodel", "model.pdiparam")

# 启用 MKLDNN 进行预测
config.enable_mkldnn()

# 通过 API 获取 MKLDNN 启用结果 - true
print(config.mkldnn_enabled())

# 设置 MKLDNN 的 cache 容量大小
config.set_mkldnn_cache_capacity(1)
```

## ONNX Runtime推理

ONNX Runtime是由微软开源的一款推理引擎，Paddle Inference通过Paddle2ONNX集成ONNX Runtime作为推理的后端之一，开发者在使用时，只需一行配置代码即可让模型通过ONNX Runtime进行推理。

<!-- omit in toc -->
### 配置文件开发说明

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

# 设置 ONNXRuntime 算子计算线程数为 10
config.set_cpu_math_library_num_threads(10)

# 禁用 ONNXRuntime 进行预测
config.DisableONNXRuntime();
# 通过 API 获取 ONNXRuntime 信息

print("Use ONNXRuntime is: {}".format(config.onnxruntime_enabled())) # false
```

