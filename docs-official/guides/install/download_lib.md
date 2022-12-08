# 下载安装 Linux 推理库
## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

|硬件后端|是否打开 avx|数学库|gcc 版本|CUDA/cuDNN/TensorRT 版本|推理库(2.4.1 版本)|
|--------------|--------------|--------------|--------------|--------------|:-----------------|
|CPU|是|MKL|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz)|
|CPU|是|MKL|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference.tgz)|
|CPU|是|OpenBLAS|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference.tgz)|
|CPU|是|OpenBLAS|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference.tgz)|
|CPU|否|OpenBLAS|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference.tgz)|
|CPU|否|OpenBLAS|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA10.2/cuDNN7.6/TensorRT7.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|MKL|5.4|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)|
|GPU|是|MKL|5.4|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA11.6/cuDNN8.4/TensorRT8.4|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.6_cudnn8.4.0-trt8.4.0.6/paddle_inference.tgz)|
|GPU|是|MKL|8.2|CUDA11.7/cuDNN8.4/TensorRT8.4|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.7_cudnn8.4.1-trt8.4.2.4/paddle_inference.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.5_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.5_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.5_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.5|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.5_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.6_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.6_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.6_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.6_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|Jetson(all)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.6.1_gcc7.5/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.6.1_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.6.1_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 4.6.1|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack4.6.1_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|Jetson(all)|-|-|-|Jetpack 5.0.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack5.0.2_gcc9.4/all/paddle_inference_install_dir.tgz)|
|Jetson(Nano)|-|-|-|Jetpack 5.0.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack5.0.2_gcc9.4/nano/paddle_inference_install_dir.tgz)|
|Jetson(TX2)|-|-|-|Jetpack 5.0.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack5.0.2_gcc9.4/tx2/paddle_inference_install_dir.tgz)|
|Jetson(Xavier)|-|-|-|Jetpack 5.0.2|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Jetson/jetpack5.0.2_gcc9.4/xavier/paddle_inference_install_dir.tgz)|

## C 推理库

- 预编译包使用方式见：[推理示例（C）](../quick_start/c_demo.md)

|硬件后端| 是否打开 avx | 数学库   | gcc 版本 | CUDA/cuDNN/TensorRT 版本 |推理库(2.4.1 版本)|
|----------|----------|----------|----------|:---------|:--------------|
|CPU|是|MKL|8.2|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference_c.tgz)|
|CPU|是|MKL|5.4|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference_c.tgz)|
|CPU|是|OpenBLAS|8.2|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference_c.tgz)|
|CPU|是|OpenBLAS|5.4| - |[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference_c.tgz)|
|CPU|否|OpenBLAS|8.2| - |[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference_c.tgz)|
|CPU|否|OpenBLAS|5.4|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA10.2/cuDNN7.6/TensorRT7.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddle_inference_c.tgz)|
|GPU|是|是|5.4|CUDA10.2/cuDNN8.1/TensorRT7.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference_c.tgz)|
|GPU|是|是|5.4|CUDA11.2/cuDNN8.2/TensorRT8.0|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA11.6/cuDNN8.4/TensorRT8.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.6_cudnn8.4.0-trt8.4.0.6/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA11.7/cuDNN8.4/TensorRT8.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.7_cudnn8.4.1-trt8.4.2.4/paddle_inference_c.tgz)|

## Python 推理库

- 预编译包使用方式见：[推理示例（Python）](../quick_start/python_demo.md)

| 版本说明   |     python3.6  |   python3.7   |     python3.8     |     python3.9   |     python3.10   |
|:---------|:----------------|:-------------|:-------------------|:----------------|:----------------|
|Jetpack4.5(4.4): nv_jetson-cuda10.2-trt7-all|[paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.5_gcc7.5/all/paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5(4.4): nv_jetson-cuda10.2-trt7-nano|[paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.5_gcc7.5/nano/paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5(4.4): nv_jetson-cuda10.2-trt7-tx2|[paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.5_gcc7.5/tx2/paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.5(4.4): nv_jetson-cuda10.2-trt7-xavier|[paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.5_gcc7.5/xavier/paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8.0-all|[paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.6_gcc7.5/all/paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8.0-nano|[paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.6_gcc7.5/nano/paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8.0-tx2|[paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.6_gcc7.5/tx2/paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8.0-xavier|[paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.6_gcc7.5/xavier/paddlepaddle_gpu-2.4.1-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6.1：nv_jetson-cuda10.2-trt8.2-all||[paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.6.1_gcc7.5/all/paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1：nv_jetson-cuda10.2-trt8.2-nano||[paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.6.1_gcc7.5/nano/paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1：nv_jetson-cuda10.2-trt8.2-tx2||[paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.6.1_gcc7.5/tx2/paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack4.6.1：nv_jetson-cuda10.2-trt8.2-xavier||[paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack4.6.1_gcc7.5/xavier/paddlepaddle_gpu-2.4.1-cp37-cp37m-linux_aarch64.whl)|||
|Jetpack5.0.2：nv-jetson-cuda11.4-cudnn8.4.1-trt8.4.1-jetpack5.0.2-all|||[paddlepaddle_gpu-2.4.1-cp38-cp38m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack5.0.2_gcc9.4/all/paddlepaddle_gpu-2.4.1-cp38-cp38-linux_aarch64.whl)||
|Jetpack5.0.2：nv-jetson-cuda11.4-cudnn8.4.1-trt8.4.1-jetpack5.0.2-nano|||[paddlepaddle_gpu-2.4.1-cp38-cp38m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack5.0.2_gcc9.4/nano/paddlepaddle_gpu-2.4.1-cp38-cp38-linux_aarch64.whl)||
|Jetpack5.0.2：nv-jetson-cuda11.4-cudnn8.4.1-trt8.4.1-jetpack5.0.2-tx2|||[paddlepaddle_gpu-2.4.1-cp38-cp38m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack5.0.2_gcc9.4/tx2/paddlepaddle_gpu-2.4.1-cp38-cp38-linux_aarch64.whl)||
|Jetpack5.0.2：nv-jetson-cuda11.4-cudnn8.4.1-trt8.4.1-jetpack5.0.2-xavier|||[paddlepaddle_gpu-2.4.1-cp38-cp38m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.4.1/python/Jetson/jetpack5.0.2_gcc9.4/xavier/paddlepaddle_gpu-2.4.1-cp38-cp38-linux_aarch64.whl)||


# 下载安装 Windows 推理库

环境硬件配置：

| 操作系统      |    win10 家庭版本      |
|:---------|:-------------------|
| CPU      |      I7-8700K      |
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

| 硬件后端 | 是否使用 avx |     编译器     |  CUDA/cuDNN/TensorRT 版本  | 数学库  |推理库(2.4.1 版本)   |
|--------------|--------------|:----------------|:--------|:-------------|:-----------------|
| CPU | 是 |  MSVC 2017 | - |MKL|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/CPU/x86-64_avx-mkl-vs2017/paddle_inference.zip)|
| CPU | 是 | MSVC 2017 | - |OpenBLAS|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/CPU/x86-64_avx-openblas-vs2017/paddle_inference.zip)|
| GPU | 是 | MSVC 2017  | CUDA10.2/cuDNN7.6/TensorRT7.0 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/GPU/x86-64_cuda10.2_cudnn7.6.5_trt7.0.0.11_mkl_avx_vs2017/paddle_inference.zip)|
| GPU | 是 | MSVC 2019  | CUDA11.2/cuDNN8.2/TensorRT8.0 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/GPU/x86-64_cuda11.2_cudnn8.2.1_trt8.0.1.6_mkl_avx_vs2019/paddle_inference.zip)|
| GPU | 是 | MSVC 2019  | CUDA11.6/cuDNN8.4/TensorRT8.4 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/GPU/x86-64_cuda11.6_cudnn8.4.0_trt8.4.1.5_mkl_avx_vs2019/paddle_inference.zip)|
| GPU | 是 | MSVC 2019  | CUDA11.7/cuDNN8.4/TensorRT8.4 |MKL |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/GPU/x86-64_cuda11.7_cudnn8.4.1_trt8.4.2.4_mkl_avx_vs2019/paddle_inference.zip)|

## C 推理库

- 预编译包使用方式见：[推理示例（C）](../quick_start/c_demo.md)

| 硬件后端 |是否打开 avx | 数学库  |     编译器版本     | CUDA/cuDNN/TensorRT 版本  |推理库(2.4.1 版本)   |
|----------|:--------|:---------|:--------------|:---------|:-----------------|
| CPU |是 |MKL|  MSVC 2017 | - | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/CPU/x86-64_avx-mkl-vs2017/paddle_inference_c.zip)|
| CPU |是 |OpenBLAS| MSVC 2017 | - | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/CPU/x86-64_avx-openblas-vs2017/paddle_inference_c.zip)|
| GPU |是 |MKL | MSVC 2017 |CUDA10.2/cuDNN7.6/TensorRT7.0| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/GPU/x86-64_cuda10.2_cudnn7.6.5_trt7.0.0.11_mkl_avx_vs2017/paddle_inference_c.zip) |
| GPU |是 |MKL | MSVC 2019 |CUDA11.2/cuDNN8.2/TensorRT8.2| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/GPU/x86-64_cuda11.2_cudnn8.2.1_trt8.0.1.6_mkl_avx_vs2019/paddle_inference_c.zip) |
| GPU |是 |MKL | MSVC 2019 |CUDA11.6/cuDNN8.4/TensorRT8.4| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/GPU/x86-64_cuda11.6_cudnn8.4.0_trt8.4.1.5_mkl_avx_vs2019/paddle_inference_c.zip) |
| GPU |是 |MKL | MSVC 2019 |CUDA11.7/cuDNN8.4/TensorRT8.4| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/Windows/GPU/x86-64_cuda11.7_cudnn8.4.1_trt8.4.2.4_mkl_avx_vs2019/paddle_inference_c.zip) |

# 下载安装 Mac 推理库

## C++ 推理库

- 预编译包使用方式见：[推理示例（C++）](../quick_start/cpp_demo.md)

|硬件后端 |是否打开 avx |数学库 |推理库(2.4.1 版本)   |
|----------|----------|----------|:----------------|
|X86_64 |是 |OpenBLAS |[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/MacOS/x86-64_clang_avx_openblas/paddle_inference_install_dir.tgz)|
|m1 | 否 |OpenBLAS |[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/MacOS/m1_clang_noavx_openblas/paddle_inference_install_dir.tgz)|

## C 推理库

- 预编译包使用方式见：[推理示例（C）](../quick_start/c_demo.md)

|硬件后端 |是否打开 avx | 数学库      |推理库(2.4.1 版)   |
|----------|----------|:---------|:----------------|
|X86_64 |是 |OpenBLAS|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/MacOS/x86-64_clang_avx_openblas/paddle_inference_c_install_dir.tgz)|
|m1 |否 |OpenBLAS|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.4.1/cxx_c/MacOS/m1_clang_noavx_openblas/paddle_inference_c_install_dir.tgz)|
