# 下载安装 Linux 推理库
## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo)

|硬件后端|是否打开avx|数学库|gcc版本|CUDA/cuDNN/TensorRT版本|推理库(2.3.0版本)|
|--------------|--------------|--------------|--------------|--------------|:-----------------|
|CPU|是|MKL|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz)|
|CPU|是|MKL|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference.tgz)|
|CPU|是|OpenBLAS|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference.tgz)|
|CPU|否|OpenBLAS|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference.tgz)|
|CPU|否|OpenBLAS|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference.tgz)|
|CPU|否|OpenBLAS|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|MKL|5.4|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA11.1/cuDNN8.1/TensorRT7.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|MKL|5.4|CUDA11.1/cuDNN8.1/TensorRT7.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)|
|GPU|是|MKL|5.4|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.5_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.5_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.5_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.5_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/xavier/paddle_inference_install_dir.tgz)|

## C 推理库

- 预编译包使用方式见：[推理示例（C）](../quick_start/c_demo)

|硬件后端| 是否打开avx | 数学库   | gcc版本 | CUDA/cuDNN/TensorRT版本 |推理库(2.3.0版本)|
|----------|----------|----------|----------|:---------|:--------------|
|CPU|是| MKL      |8.2|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference_c.tgz)|
|CPU|是|MKL|5.4|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference_c.tgz)|
|CPU|是|OpenBLAS|8.2|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference_c.tgz)|
|CPU|否|OpenBLAS|5.4| - |[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference_c.tgz)|
|CPU|否|OpenBLAS|8.2| - |[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference_c.tgz)|
|CPU|否|OpenBLAS|5.4|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|GPU|是|是|5.4|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA11.1/cuDNN8.2/TensorRT7.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|GPU|是|是|5.4|CUDA11.1/cuDNN8.1/TensorRT7.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference_c.tgz)|
|GPU|是|是|5.4|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference_c.tgz)|

## Python 推理库

- 预编译包使用方式见：[推理示例（Python）](../quick_start/python_demo)

| 版本说明   |     python3.6  |   python3.7   |     python3.8     |     python3.9   |
|:---------|:----------------|:-------------|:-------------------|:----------------|
|linux-cuda10.2-cudnn8.1-trt7-gcc8.2|[paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle-cp37m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_x86_64.whl)|[paddlepaddle-cp38.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0-cp38-cp38-linux_x86_64.whl)|[paddlepaddle-cp39.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0-cp39-cp39-linux_x86_64.whl)|
|linux-cuda11.1-cudnn8.1-trt7-gcc8.2|[paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0.post111-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle-cp37m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0.post111-cp37-cp37m-linux_x86_64.whl)|[paddlepaddle-cp38.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0.post111-cp38-cp38-linux_x86_64.whl)|[paddlepaddle-cp39.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.3.0.post111-cp39-cp39-linux_x86_64.whl)|
|linux-cuda11.2-cudnn8.2-trt8-gcc8.2|[paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.3.0.post112-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle-cp37m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.3.0.post112-cp37-cp37m-linux_x86_64.whl)|[paddlepaddle-cp38.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.3.0.post112-cp38-cp38-linux_x86_64.whl)|[paddlepaddle-cp39.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.3.0.post112-cp39-cp39-linux_x86_64.whl)|
|Jetpack4.5/4.4(all)|[paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.5_gcc7.5/all/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5/4.4(Nano)|[paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.5_gcc7.5/nano/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5/4.4(TX2)|[paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.5_gcc7.5/tx2/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5/4.4(Xavier)|[paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.5_gcc7.5/xavier/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6(all)|[paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.6_gcc7.5/all/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6(Nano)|[paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.6_gcc7.5/nano/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6(TX2)|[paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.6_gcc7.5/tx2/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6(Xavier)|[paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.6_gcc7.5/xavier/paddlepaddle_gpu-2.3.0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6.1(all)||[paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.6.1_gcc7.5/all/paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1(Nano)||[paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.6.1_gcc7.5/nano/paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1(TX2)||[paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.6.1_gcc7.5/tx2/paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1(Xavier)||[paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Jetson/jetpack4.6.1_gcc7.5/xavier/paddlepaddle_gpu-2.3.0-cp37-cp37m-linux_aarch64.whl)|||


# 下载安装 Windows 推理库

环境硬件配置：

| 操作系统      |    win10 家庭版本      |
|:---------|:-------------------|
| CPU      |      I7-8700K      |
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo)

| 硬件后端 | 是否使用avx |     编译器     |  CUDA/cuDNN/TensorRT版本  | 数学库  |推理库(2.3.0版本)   |
|--------------|--------------|:----------------|:--------|:-------------|:-----------------|
| CPU | 是 |  MSVC 2017 | - |MKL|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/CPU/x86-64_vs2017_avx_mkl/paddle_inference.zip)| - |
| CPU | 是 | MSVC 2017 | - |OpenBLAS|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/CPU/x86-64_vs2017_avx_openblas/paddle_inference.zip)| - |
| GPU | 是 | MSVC 2017  | CUDA10.1/cuDNN7.6/no_trt | MKL                                          |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7/paddle_inference_notrt.zip)|  10.1 |
| GPU | 是 | MSVC 2017  | CUDA10.2/cuDNN7.6/TensorRT7.0 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7/paddle_inference.zip)|  10.2 |
| GPU | 是 | MSVC 2017  | CUDA11.0/cuDNN8.0/TensorRT7.2 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.0_cudnn8/paddle_inference.zip)| 11.0 |
| GPU | 是 | MSVC 2017  | CUDA11.2/cuDNN8.2/TensorRT8.0 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.2_cudnn8/paddle_inference.zip)| 11.2 |

## C 推理库

- 预编译包使用方式见：[推理示例（C）](../quick_start/c_demo)

| 硬件后端 |是否打开avx | 数学库  |     编译器版本     | CUDA/cuDNN/TensorRT版本  |推理库(2.3.0版本)   |
|----------|:--------|:---------|:--------------|:---------|:-----------------|
| CPU |是 |MKL|  MSVC 2017 | - | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/CPU/x86-64_vs2017_avx_mkl/paddle_inference_c.zip)|
| CPU |是 |OpenBLAS| MSVC 2017 | - | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/CPU/x86-64_vs2017_avx_openblas/paddle_inference_c.zip)|
| GPU |是 |MKL | MSVC 2017|CUDA10.1/cuDNN7.6/no_trt | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7/paddle_inference_c_notrt.zip) |
| GPU |是 |MKL | MSVC 2017 |CUDA10.2/cuDNN7.6/TensorRT7.0| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7/paddle_inference_c.zip) |
| GPU |是 |MKL | MSVC 2017 |CUDA11.0/cuDNN8.0/TensorRT7.2| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.0_cudnn8/paddle_inference_c.zip) |
| GPU |是 |MKL | MSVC 2017 |CUDA11.2/cuDNN8.2/TensorRT8.2| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.2_cudnn8/paddle_inference_c.zip) |

## python 推理

- 预编译包使用方式见：[推理示例（Python）](../quick_start/python_demo)

| 版本说明  |python3.8   |
|:---------|:-----------------|
|cuda10.2_cudnn7.6.5_avx_mkl-trt7.0.0.11|[paddlepaddle-cp38m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddlepaddle_gpu-2.3.0-cp38-cp38-win_amd64.whl)|
|cuda11.0_cudnn8.0_avx_mkl-trt7.2.1.6|[paddlepaddle-cp38m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.0_cudnn8.0.2_trt7.2/paddlepaddle_gpu-2.3.0.post110-cp38-cp38-win_amd64.whl)|
|cuda11.2_cudnn8.2_avx_mkl-trt8.0.1.6|[paddlepaddle-cp38m.whl](https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.1.6/paddlepaddle_gpu-2.3.0.post112-cp38-cp38-win_amd64.whl)|

# 下载安装 Mac 推理库

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo)

|硬件后端 |是否打开avx |数学库 |推理库(2.3.0版本)   |
|----------|----------|----------|:----------------|
|CPU |是 |OpenBLAS |[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/MacOS/CPU/x86-64_clang_avx_openblas/paddle_inference_install_dir.tgz)|

## C 推理库

- 预编译包使用方式见：[推理示例（C）](../quick_start/c_demo)

|硬件后端 |是否打开avx | 数学库      |推理库(2.3.0版)   |
|----------|----------|:---------|:----------------|
|CPU |是 |OpenBLAS|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/MacOS/CPU/x86-64_clang_avx_openblas/paddle_inference_c_install_dir.tgz)|
