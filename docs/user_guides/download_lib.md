# 下载安装Linux预测库

| 版本说明      |     预测库(1.8.5版本)  |预测库(2.0.1版本)   |     预测库(develop版本)     |  
|:---------|:-------------------|:-------------------|:----------------|
|manylinux_cpu_avx_mkl_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-cpu-avx-mkl/paddle_inference.tgz)||
|manylinux_cpu_avx_openblas_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-cpu-avx-openblas/paddle_inference.tgz)||
|manylinux_cpu_noavx_openblas_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-cpu-noavx-openblas/paddle_inference.tgz)||
|manylinux_cuda9.0_cudnn7_avx_mkl_gcc54||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz)|
|manylinux_cuda10.0_cudnn7_avx_mkl_gcc54||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz)|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz)|
|manylinux_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda10.1-cudnn7-avx-mkl/paddle_inference.tgz)|||
|manylinux_cuda10.2_cudnn8.0_avx_mkl_trt7_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda10.2-cudnn8-avx-mkl/paddle_inference.tgz)|||
|manylinux_cuda11.0_cudnn8.0_avx_mkl_trt7_gcc82||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda11-cudnn8-avx-mkl/paddle_inference.tgz)|||
|nv_jetson_cuda10_cudnn7.6_trt6_all(jetpack4.3)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-all/paddle_inference.tgz)|||
|nv_jetson_cuda10_cudnn7.6_trt6_nano(jetpack4.3)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-nano/paddle_inference.tgz)|||
|nv_jetson_cuda10_cudnn7.6_trt6_tx2(jetpack4.3)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-tx2/paddle_inference.tgz)|||
|nv_jetson_cuda10_cudnn7.6_trt6_xavier(jetpack4.3)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-xavier/paddle_inference_install_dir.tgz)|||
|nv_jetson_cuda10.2_cudnn8_trt7_all(jetpack4.4)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.4-all/paddle_inference.tgz)|||
|nv_jetson_cuda10.2_cudnn8_trt7_nano(jetpack4.4)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.4-nano/paddle_inference.tgz)|||
|nv_jetson_cuda10.2_cudnn8_trt7_tx2(jetpack4.4)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.4-tx2/paddle_inference.tgz)|||
|nv_jetson_cuda10.2_cudnn8_trt7_xavier(jetpack4.4)||[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.4-xavier/paddle_inference.tgz)|||



# 下载安装Windows预测库


| 版本说明      |     预测库(1.8.4版本)  |预测库(2.0.1版本)   |     编译器     |    构建工具      |  cuDNN  |  CUDA  |
|:---------|:-------------------|:-------------------|:----------------|:--------|:-------|:-------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/mkl/cpu/paddle_inference_install_dir.zip)|  MSVC 2015 update 3|  CMake v3.17.0  | - | - |
|    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/cpu/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/open/cpu/paddle_inference_install_dir.zip)| MSVC 2015 update 3|  CMake v3.17.0  | - | - |
|    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post97/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/mkl/post100/paddle_inference_install_dir.zip)| MSVC 2015 update 3 |  CMake v3.17.0  |  7.3.1  |   9.0    |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/post97/fluid_inference_install_dir.zip) | - | MSVC 2015 update 3 |  CMake v3.17.0  |  7.3.1  |   9.0    |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post107/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/mkl/post100/paddle_inference_install_dir.zip) |MSVC 2015 update 3 |  CMake v3.17.0  |  7.4.1  |   10.0    |
|    cuda10.0_cudnn7_avx_mkl_trt6 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/trt_mkl/post100/paddle_inference.zip)| MSVC 2015 update 3 |  CMake v3.17.0  |  7.4.1  |   10.0    |
|    cuda10.1_cudnn7_avx_mkl_trt6 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/trt_mkl/post101/paddle_inference.zip)| MSVC 2015 update 3 |  CMake v3.17.0  |  7.6  |   10.1    |
|    cuda10.2_cudnn7_avx_mkl_trt7 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/trt_mkl/post102/paddle_inference.zip)| MSVC 2015 update 3 |  CMake v3.17.0  |  7.6  |   10.2    |
|    cuda11.0_cudnn8_avx_mkl_trt7 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/trt_mkl/post11/paddle_inference.zip)| MSVC 2015 update 3 |  CMake v3.17.0  |  8.0  |   11.0    |


环境硬件配置：

| 操作系统      |    win10 家庭版本      |
|:---------|:-------------------|
| CPU      |      I7-8700K      |
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |
