# 下载安装Linux预测库

| 版本说明      |     预测库(1.8.5版本)  |预测库(2.0.0-rc0版本)   |     预测库(develop版本)     |  
|:---------|:-------------------|:-------------------|:----------------|
|manylinux_cpu_avx_mkl_gcc482|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-avx-mkl/fluid_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc0-cpu-avx-mkl/paddle_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-mkl/paddle_inference.tgz)|
|manylinux_cpu_avx_openblas_gcc482|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-avx-openblas/fluid_inference.tgz)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-openblas/paddle_inference.tgz)|
|manylinux_cpu_noavx_openblas_gcc482|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-noavx-openblas/fluid_inference.tgz)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-noavx-openblas/paddle_inference.tgz)|
|manylinux_cuda9.0_cudnn7_avx_mkl_gcc482|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc0-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz)|
|manylinux_cuda10.0_cudnn7_avx_mkl_gcc482|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc0-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz)|
|manylinux_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc482|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-gpu-cuda10.1-cudnn7.6-avx-mkl-trt6/fluid_inference.tgz)||||
|manylinux_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc0-gpu-cuda10.1-cudnn7-avx-mkl/paddle_inference.tgz)|||
|manylinux_cuda10.2_cudnn8.0_avx_mkl_trt7_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc0-gpu-cuda10.2-cudnn8-avx-mkl/paddle_inference.tgz)|||
|nv_jetson_cuda10_cudnn7.5_trt5(jetpack4.2)|[fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.8.5-nv-jetson-cuda10-cudnn7.5-trt5/fluid_inference.tgz)||||
|nv_jetson_cuda10_cudnn7.6_trt6(jetpack4.3)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc0-nv-jetson-cuda10-cudnn7.6-trt6/paddle_inference.tgz)|||


# 下载安装Windows预测库


| 版本说明      |     预测库(1.8.4版本)  |预测库(2.0.0-beta0版本)   |     编译器     |    构建工具      |  cuDNN  |  CUDA  |
|:---------|:-------------------|:-------------------|:----------------|:--------|:-------|:-------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.0-beta0/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | MSVC 2015 update 3|  CMake v3.16.0  |
|    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/cpu/fluid_inference_install_dir.zip) || MSVC 2015 update 3|  CMake v3.16.0  |
|    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post97/fluid_inference_install_dir.zip) ||  MSVC 2015 update 3 |  CMake v3.16.0  |  7.3.1  |   9.0    |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/post97/fluid_inference_install_dir.zip) || MSVC 2015 update 3 |  CMake v3.16.0  |  7.3.1  |   9.0    |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post107/fluid_inference_install_dir.zip) | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.3/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.16.0  |  7.4.1  |   10.0    |


环境硬件配置：

| 操作系统      |    win10 家庭版本      |
|:---------|:-------------------|
| CPU      |      I7-8700K      |
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |
