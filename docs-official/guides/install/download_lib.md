# 下载安装 Linux 推理库

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

|硬件后端|是否打开 avx|数学库|gcc 版本|CUDA/cuDNN/TensorRT 版本|推理库(3.2.1 版本)|
|--------------|--------------|--------------|--------------|--------------|:-----------------|
|CPU|是|MKL|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.2.1/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz)|
|GPU|是|MKL|11.2|CUDA11.8/cuDNN8.9/TensorRT8.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.2.1/cxx_c/Linux/GPU/x86-64_gcc11.2_avx_mkl_cuda11.8_cudnn8.9.7-trt8.6.1.6/paddle_inference.tgz)|
|GPU|是|MKL|11.2|CUDA12.6/cuDNN9.5/TensorRT10.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.2.1/cxx_c/Linux/GPU/x86-64_gcc11.2_avx_mkl_cuda12.6_cudnn9.5.1-trt10.5.0.18/paddle_inference.tgz)|
|GPU|是|MKL|11.2|CUDA12.9/cuDNN9.9/TensorRT10.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.2.1/cxx_c/Linux/GPU/x86-64_gcc11.2_avx_mkl_cuda12.9_cudnn9.9.0-trt10.5.0.18/paddle_inference.tgz)|


# 下载安装 Windows 推理库

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

| 硬件后端 | 是否使用 avx |     编译器     |  CUDA/cuDNN/TensorRT 版本  | 数学库  |推理库(3.2.1 版本)   |
|--------------|--------------|:----------------|:--------|:-------------|:-----------------|
| CPU | 是 |  MSVC 2019 | - |MKL|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/3.2.1/cxx_c/Windows/CPU/x86-64_avx-mkl-vs2019/paddle_inference.zip)|


# 下载安装 Mac 推理库

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

|硬件后端 |是否打开 avx |数学库 |推理库(3.2.1 版本)   |
|----------|----------|----------|:----------------|
|m1 | 否 |Accelerate BLAS |[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.2.1/cxx_c/MacOS/m1_clang_noavx_accelerate_blas/paddle_inference.tgz)|
