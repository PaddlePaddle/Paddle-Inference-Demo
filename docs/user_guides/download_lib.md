# 下载安装Linux预测库
## C++预测库

| 版本说明      |     预测库(1.8.5版本)  |预测库(2.2.0-rc0版本)   |     预测库(develop版本)     |  
|:-------------|:---------------------|:-----------------|:---------------------------|
|manylinux_cpu_avx_mkl_gcc8.2|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-avx-mkl/fluid_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-mkl/paddle_inference.tgz)|
|manylinux_cpu_avx_mkl_gcc5.4||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference.tgz)||
|manylinux_cpu_avx_openblas_gcc8.2|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-avx-openblas/fluid_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-openblas/paddle_inference.tgz)|
|manylinux_cpu_avx_openblas_gcc5.4||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference.tgz)||
|manylinux_cpu_noavx_openblas_gcc8.2|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-noavx-openblas/fluid_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-noavx-openblas/paddle_inference.tgz)|
|manylinux_cpu_noavx_openblas_gcc5.4||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference.tgz)||
|manylinux_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc8.2||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz)||
|manylinux_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc5.4||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz)||
|manylinux_cuda10.2_cudnn7.6_avx_mkl_trt7_gcc5.4||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz)||
|manylinux_cuda10.2_cudnn8.1_avx_mkl_trt7_gcc8.2||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)||
|manylinux_cuda10.2_cudnn8.1_avx_mkl_trt7_gcc5.4||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)||
|manylinux_cuda11.1_cudnn8.1_avx_mkl_trt7_gcc8.2||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)||
|manylinux_cuda11.1_cudnn8.1_avx_mkl_trt7_gcc5.4||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)||
|manylinux_cuda11.2_cudnn8.2_avx_mkl_trt8_gcc8.2||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)||
|manylinux_cuda11.2_cudnn8.2_avx_mkl_trt8_gcc5.4||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)||
|Jetpack4.4(4.5): nv-jetson-cuda10.2-cudnn8-trt7(all)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Jetson/jetpack4.4_gcc7.5/all/paddle_inference_install_dir.tgz)||
|Jetpack4.4(4.5): nv-jetson-cuda10.2-cudnn8-trt7(nano)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Jetson/jetpack4.4_gcc7.5/nano/paddle_inference_install_dir.tgz)||
|Jetpack4.4(4.5): nv-jetson-cuda10.2-cudnn8-trt7(tx2)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Jetson/jetpack4.4_gcc7.5/tx2/paddle_inference_install_dir.tgz)||
|Jetpack4.4(4.5): nv-jetson-cuda10.2-cudnn8-trt7(xavier)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Jetson/jetpack4.4_gcc7.5/xavier/paddle_inference_install_dir.tgz)||
|Jetpack4.6: nv-jetson-cuda10.2-cudnn8-trt7(all)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Jetson/jetpack4.6_gcc7.5/all/paddle_inference_install_dir.tgz)||
|Jetpack4.6: nv-jetson-cuda10.2-cudnn8-trt7(nano)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Jetson/jetpack4.6_gcc7.5/nano/paddle_inference_install_dir.tgz)||
|Jetpack4.6: nv-jetson-cuda10.2-cudnn8-trt7(tx2)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Jetson/jetpack4.6_gcc7.5/tx2/paddle_inference_install_dir.tgz)||
|Jetpack4.6: nv-jetson-cuda10.2-cudnn8-trt7(xavier)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Jetson/jetpack4.6_gcc7.5/xavier/paddle_inference_install_dir.tgz)||


## C预测库

|  版本说明 |预测库(2.2.0-rc0版本)|
|:---------|:--------------|
|manylinux_cpu_avx_mkl_gcc8.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference_c.tgz)||
|manylinux_cpu_avx_mkl_gcc5.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference_c.tgz)|
|manylinux_cpu_avx_openblas_gcc8.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference_c.tgz)|
|manylinux_cpu_avx_openblas_gcc5.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference_c.tgz)|
|manylinux_cpu_noavx_openblas_gcc8.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference_c.tgz)|
|manylinux_cpu_noavx_openblas_gcc5.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference_c.tgz)|
|manylinux_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc8.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference_c.tgz)|
|manylinux_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc5.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference_c.tgz)|
|manylinux_cuda10.2_cudnn7.6_avx_mkl_trt6_gcc5.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddle_inference_c.tgz)|
|manylinux_cuda10.2_cudnn8.1_avx_mkl_trt7_gcc8.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|manylinux_cuda10.2_cudnn8.1_avx_mkl_trt7_gcc5.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|manylinux_cuda11.1_cudnn8.1_avx_mkl_trt7_gcc8.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|manylinux_cuda11.1_cudnn8.1_avx_mkl_trt7_gcc5.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference_c.tgz)|
|manylinux_cuda11.2_cudnn8.2_avx_mkl_trt8_gcc8.2|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference_c.tgz)|
|manylinux_cuda11.2_cudnn8.2_avx_mkl_trt8_gcc5.4|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference_c.tgz)|


## Python预测库


| 版本说明   |     python3.6  |   python3.7   |     python3.8     |     python3.9   |  
|:---------|:----------------|:-------------|:-------------------|:----------------|
|linux-cuda10.1-cudnn7.6-trt6-gcc8.2|[paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0rc0.post101-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle-cp37m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0rc0.post101-cp37-cp37m-linux_x86_64.whl)|[paddlepaddle-cp38.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0rc0.post101-cp38-cp38-linux_x86_64.whl)|[paddlepaddle-cp39.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0rc0.post101-cp39-cp39-linux_x86_64.whl)|
|linux-cuda10.2-cudnn8.1-trt7-gcc8.2|[paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle-cp37m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0rc0-cp37-cp37m-linux_x86_64.whl)|[paddlepaddle-cp38.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0rc0-cp38-cp38-linux_x86_64.whl)|[paddlepaddle-cp39.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0rc0-cp39-cp39-linux_x86_64.whl)|
|linux-cuda11.1-cudnn8.1-trt7-gcc8.2|[paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0rc0.post111-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle-cp37m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0rc0.post111-cp37-cp37m-linux_x86_64.whl)|[paddlepaddle-cp38.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0rc0.post111-cp38-cp38-linux_x86_64.whl)|[paddlepaddle-cp39.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0rc0.post111-cp39-cp39-linux_x86_64.whl)|
|linux-cuda11.2-cudnn8.2-trt8-gcc8.2|[paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.2.0rc0.post112-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle-cp37m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.2.0rc0.post112-cp37-cp37m-linux_x86_64.whl)|[paddlepaddle-cp38.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.2.0rc0.post112-cp38-cp38-linux_x86_64.whl)|[paddlepaddle-cp39.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.2.0rc0.post112-cp39-cp39-linux_x86_64.whl)|
|Jetpack4.4(4.5): nv_jetson-cuda10.2-trt7-all|[paddlepaddle_gpu-2.2.0-rc0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Jetson/jetpack4.4_gcc7.5/all/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.4(4.5): nv_jetson-cuda10.2-trt7-nano|[paddlepaddle_gpu-2.2.0-rc0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Jetson/jetpack4.4_gcc7.5/nano/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.4(4.5): nv_jetson-cuda10.2-trt7-tx2|[paddlepaddle_gpu-2.2.0-rc0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Jetson/jetpack4.4_gcc7.5/tx2/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.4(4.5): nv_jetson-cuda10.2-trt7-xavier|[paddlepaddle_gpu-2.2.0-rc0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Jetson/jetpack4.4_gcc7.5/xavier/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8-all|[paddlepaddle_gpu-2.2.0-rc0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Jetson/jetpack4.6_gcc7.5/all/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8-nano|[paddlepaddle_gpu-2.2.0-rc0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Jetson/jetpack4.6_gcc7.5/nano/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8-tx2|[paddlepaddle_gpu-2.2.0-rc0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Jetson/jetpack4.6_gcc7.5/tx2/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_aarch64.whl)||||
|Jetpack4.6：nv_jetson-cuda10.2-trt8-xavier|[paddlepaddle_gpu-2.2.0-rc0-cp36-cp36m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Jetson/jetpack4.6_gcc7.5/xavier/paddlepaddle_gpu-2.2.0rc0-cp36-cp36m-linux_aarch64.whl)||||



# 下载安装Windows预测库

**请确认您的VS版本与下载链接对应的版本完全一致，目前暂不保证在其它VS版本的可用性**

环境硬件配置：

| 操作系统      |    win10 家庭版本      |
|:---------|:-------------------|
| CPU      |      I7-8700K      |
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |

## C++预测库

| 版本说明      |     预测库(1.8.4版本)  |预测库(2.2.0-rc0版本)   |     编译器     |  cuDNN  |  CUDA  |
|:-------------|:---------------------|:-----------------|:----------------|:--------|:-------|
|cpu_avx_mkl| [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/CPU/x86-64_vs2017_avx_mkl/paddle_inference.zip)|  MSVC 2017 | - | - |
|cpu_avx_openblas| [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/cpu/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/CPU/x86-64_vs2017_avx_openblas/paddle_inference.zip)| MSVC 2017 | - | - |
|cuda10.1_cudnn7.6_avx_mkl_no_trt | |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference_notrt.zip)| MSVC 2017  | 7.6|  10.1 |
|cuda10.1_cudnn7.6_avx_mkl_trt6 | |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.zip)| MSVC 2017  | 7.6|  10.1 |
|cuda10.2_cudnn7.6_avx_mkl_trt7 | |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddle_inference.zip)| MSVC 2017  | 7.6 | 10.2 |
|cuda11.0_cudnn8.0_avx_mkl_trt7 | |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11_cudnn8_trt7.2/paddle_inference.zip)| MSVC 2017  | 8.0 | 11.0 |

## C预测库

| 版本说明  |预测库(2.2.0-rc0版本)   |     编译器     |   cuDNN  |  CUDA  |
|:---------|:-----------------|:--------------|:---------|:--------|
|cpu_avx_mkl| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/CPU/x86-64_vs2017_avx_mkl/paddle_inference_c.zip)|  MSVC 2017 | - | - |
|cpu_avx_openblas| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/CPU/x86-64_vs2017_avx_openblas/paddle_inference_c.zip)| MSVC 2017 | - | - |
|cuda10.1_cudnn7.6_avx_mkl_no_trt | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference_c_notrt.zip) | MSVC 2017|7.6|   10.0    |
|cuda10.1_cudnn7.6_avx_mkl_trt6 | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference_c.zip) | MSVC 2017|7.6|   10.0    |
|cuda10.2_cudnn7.6_avx_mkl_trt7 | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddle_inference_c.zip) | MSVC 2017 |7.6|   10.2    |
|cuda11.0_cudnn8.0_avx_mkl_trt7 | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11_cudnn8_trt7.2/paddle_inference_c.zip) | MSVC 2017 |8.0|   11.0    |

## python预测库
| 版本说明   |     python3.8     |  
|:---------|:-------------------|
|cuda10.1-cudnn7.6-trt6-gcc8.2|[paddlepaddle-cp38m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0rc0.post101-cp38-cp38-win_amd64.whl)|
|cuda10.2-cudnn8.1-trt7-gcc8.2|[paddlepaddle-cp38m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddlepaddle_gpu-2.2.0rc0-cp38-cp38-win_amd64.whl)|
|cuda11.0-cudnn8.1-trt7-gcc8.2|[paddlepaddle-cp38m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.0_cudnn8_trt7.2/paddlepaddle_gpu-2.2.0rc0.post110-cp38-cp38-win_amd64.whl)|

# 下载安装Mac预测库

## C++预测库

| 版本说明       |预测库(2.2.0-rc0版本)   |
|:---------|:----------------|
|cpu_avx_openblas|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/MacOS/CPU/x86-64_clang_avx_openblas/paddle_inference_install_dir.tgz)|

## C预测库

| 版本说明       |预测库(2.2.0-rc0版本)   |
|:---------|:----------------|
|cpu_avx_openblas|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/MacOS/CPU/x86-64_clang_avx_openblas/paddle_inference_c_install_dir.tgz)|




