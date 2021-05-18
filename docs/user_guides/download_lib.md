# 下载安装Linux预测库

| 版本说明      |     预测库(1.8.5版本)  |预测库(2.1.0版本)   |     预测库(develop版本)     |  
|:---------|:-------------------|:-------------------|:----------------|
|manylinux_cpu_avx_mkl_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddle_inference.tgz)||
|manylinux_cpu_avx_openblas_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-cpu-avx-openblas/paddle_inference.tgz)||
|manylinux_cpu_noavx_openblas_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-cpu-noavx-openblas/paddle_inference.tgz)||
|manylinux_cuda9.0_cudnn7_avx_mkl_gcc54|||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz)|
|manylinux_cuda10.0_cudnn7_avx_mkl_gcc54|||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz)|
|manylinux_cuda10.1_cudnn7.6_avx_mkl_gcc54||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc5.4/paddle_inference.tgz)||
|manylinux_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddle_inference.tgz)|||
|manylinux_cuda10.2_cudnn8.0_avx_mkl_trt7_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddle_inference.tgz)|||
|manylinux_cuda11.0_cudnn8.0_avx_mkl_trt7_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddle_inference.tgz)|||
|nv_jetson_cuda10.2_cudnn8_trt7_all(jetpack4.4)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.1.0-nv-jetson-jetpack4.4-all/paddle_inference.tgz)|||




# 下载安装Windows预测库


| 版本说明      |     预测库(1.8.4版本)  |预测库(2.1.0版本)   |     编译器     |    构建工具      |  cuDNN  |  CUDA  |
|:---------|:-------------------|:-------------------|:----------------|:--------|:-------|:-------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.1.0win/win-infer/mkl/cpu/paddle_inference.zip)|  MSVC 2017 |  CMake v3.17.0  | - | - |
|    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/cpu/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.1.0win/win-infer/open/cpu/paddle_inference.zip)| MSVC 2017 |  CMake v3.17.0  | - | - |
|    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post97/fluid_inference_install_dir.zip) | | MSVC 2015 update 3 |  CMake v3.17.0  |  7.3.1  |   9.0    |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/post97/fluid_inference_install_dir.zip) | - | MSVC 2015 update 3 |  CMake v3.17.0  |  7.3.1  |   9.0    |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post107/fluid_inference_install_dir.zip) | |MSVC 2015 update 3 |  CMake v3.17.0  |  7.4.1  |   10.0    |
|    cuda10.1_cudnn7_avx_mkl_trt6 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.1.0win/win-infer/mkl/post101/paddle_inference.zip)| MSVC 2017 |  CMake v3.17.0  |  7.6  |   10.1    |
|    cuda10.2_cudnn7_avx_mkl_trt7 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.1.0win/win-infer/mkl/post102/paddle_inference.zip)| MSVC 2017  |  CMake v3.17.0  |  7.6  |   10.2    |
|    cuda11.0_cudnn8_avx_mkl_trt7 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.1.0win/win-infer/mkl/post110/paddle_inference.zip)| MSVC 2017  |  CMake v3.17.0  |  8.0  |   11.0    |

# 下载安装Mac预测库

| 版本说明       |预测库(2.1.0版本)   |
|:---------|:----------------|
|cpu_avx_openblas||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/mac/2.1.0/cpu_avx_openblas/paddle_inference.tgz)||




环境硬件配置：

| 操作系统      |    win10 家庭版本      |
|:---------|:-------------------|
| CPU      |      I7-8700K      |
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |
