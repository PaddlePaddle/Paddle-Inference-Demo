# 下载安装 Linux 推理库

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

|硬件后端|是否打开 avx|数学库|gcc 版本|CUDA/cuDNN/TensorRT 版本|推理库(3.0.0-rc0 版本)|
|--------------|--------------|--------------|--------------|--------------|:-----------------|
|CPU|是|MKL|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz)|
|CPU|是|OpenBLAS|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA11.8/cuDNN8.6/TensorRT8.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.8_cudnn8.6.0-trt8.5.1.7/paddle_inference.tgz)|
|GPU|是|MKL|12.2|CUDA12.3/cuDNN9.0/TensorRT8.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Linux/GPU/x86-64_gcc12.2_avx_mkl_cuda12.3_cudnn9.0.0-trt8.6.1.6/paddle_inference.tgz)|
|Jetson(all)|-|-|9.4|Jetpack 5.1.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Jetson/jetpack5.1.2_gcc9.4/all/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|9.4|Jetpack 5.1.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Jetson/jetpack5.1.2_gcc9.4/xavier/paddle_inference_install_dir.tgz)|
|Jetson(orin)|-|-|9.4|Jetpack 5.1.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Jetson/jetpack5.1.2_gcc9.4/orin/paddle_inference_install_dir.tgz)|


## Python 推理库

- 预编译包使用方式见：[推理示例（Python）](../quick_start/python_demo.md)

| 版本说明                                                                     | python3.8                                                                                                                                                                                                    |
|:-------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Jetpack5.1.2: nv-jetson-cuda11.4-cudnn8.6.0-trt8.5.2-jetpack5.1.2-all    | [paddlepaddle_gpu-3.0.0rc0-cp38-cp38-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/python/Jetson/jetpack5.1.2_gcc9.4/all/paddlepaddle_gpu-3.0.0rc0-cp38-cp38-linux_aarch64.whl)    |
| Jetpack5.1.2: nv-jetson-cuda11.4-cudnn8.6.0-trt8.5.2-jetpack5.1.2-xavier | [paddlepaddle_gpu-3.0.0rc0-cp38-cp38-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/python/Jetson/jetpack5.1.2_gcc9.4/xavier/paddlepaddle_gpu-3.0.0rc0-cp38-cp38-linux_aarch64.whl) |
| Jetpack5.1.2: nv-jetson-cuda11.4-cudnn8.6.0-trt8.5.2-jetpack5.1.2-orin   | [paddlepaddle_gpu-3.0.0rc0-cp38-cp38-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/python/Jetson/jetpack5.1.2_gcc9.4/orin/paddlepaddle_gpu-3.0.0rc0-cp38-cp38-linux_aarch64.whl)      |


# 下载安装 Windows 推理库

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

| 硬件后端 | 是否使用 avx |     编译器     |  CUDA/cuDNN/TensorRT 版本  | 数学库  |推理库(3.0.0-rc0 版本)   |
|--------------|--------------|:----------------|:--------|:-------------|:-----------------|
| CPU | 是 |  MSVC 2019 | - |MKL|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Windows/CPU/x86-64_avx-mkl-vs2019/paddle_inference.zip)|
| CPU | 是 | MSVC 2019 | - |OpenBLAS|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Windows/CPU/x86-64_avx-openblas-vs2019/paddle_inference.zip)|
| GPU | 是 | MSVC 2019  | CUDA11.8/cuDNN8.6/TensorRT8.5 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Windows/GPU/x86-64_cuda11.8_cudnn8.6.0_trt8.5.1.7_mkl_avx_vs2019/paddle_inference.zip)|
| GPU | 是 | MSVC 2019  | CUDA12.3/cuDNN9.0/TensorRT8.6 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/Windows/GPU/x86-64_cuda12.3_cudnn9.0.0_trt8.6.1.6_mkl_avx_vs2019/paddle_inference.zip)|


# 下载安装 Mac 推理库

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

|硬件后端 |是否打开 avx |数学库 |推理库(3.0.0-rc0 版本)   |
|----------|----------|----------|:----------------|
|X86_64 |是 |Accelerate BLAS |[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/MacOS/x86-64_clang_avx_accelerate_blas/paddle_inference.tgz)|
|m1 | 否 |Accelerate BLAS |[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/3.0.0-rc0/cxx_c/MacOS/m1_clang_noavx_accelerate_blas/paddle_inference.tgz)|
