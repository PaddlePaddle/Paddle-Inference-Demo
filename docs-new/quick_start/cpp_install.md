# C++ 推理部署

本文主要介绍 Paddle Inference C++ API 的安装。主要分为以下三个章节：环境准备、安装步骤和验证安装。三个章节分别说明了安装前的环境要求、安装的具体流程和成功安装后的验证方法。

## 环境准备

- G++ 8.2
- CMake 3.0+
- Visual Studio 2015 Update 3 （仅在使用 Windows 版本的预测库时需要，根据 Paddle 预测库所使用的 VS 版本选择，请参考 [Visual Studio 不同版本二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=msvc-170&viewFallbackFrom=vs-2019) )
- cuda 10.1 / cuda 10.2 / cuda 11.0 / cuda 11.2, cudnn7.6+, tensorrt （仅在使用 gpu 版本的预测库时需要）

（您可参考 nvidia 官方文档了解 cuda 和 cudnn 的安装流程和配置方法，请见 [cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cudnn](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，版本对应关系如下表所示)

|CUDA 版本|cudnn 版本| TensorRt 版本|
|---|---|---|
|10.1|7.6|6|
|10.2|7.6|7|
|11.0|8.0|7|
|11.2|8.2|8|

## 开始安装

Paddle Inference 提供了 Linux/Windows/MacOS 平台的官方 Release 预测库下载，如果您使用的是以上平台，我们优先推荐您通过以下链接直接下载，或者您也可以参照文档进行[源码编译](../user_guides/source_compile.html)。

- [下载安装 Linux 预测库](../user_guides/download_lib.html#linux)

|硬件后端|是否打开avx|数学库|gcc版本|cuda/cudnn版本|预测库(2.3.0版本)|
|--------------|--------------|--------------|--------------|--------------|:-----------------|
|CPU|是|mkl|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference.tgz)|
|CPU|是|mkl|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference.tgz)|
|CPU|是|openblas|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference.tgz)|
|CPU|否|openblas|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference.tgz)|
|CPU|否|openblas|8.2|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference.tgz)|
|CPU|否|openblas|5.4|-|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference.tgz)|
|GPU|是|mkl|8.2|cuda10.1/cudnn7.6/trt6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz)|
|GPU|是|mkl|5.4|cuda10.1/cudnn7.6/trt6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz)|
|GPU|是|mkl|5.4|cuda10.2/cudnn7.6/trt6|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz)|
|GPU|是|mkl|8.2|cuda10.2/cudnn8.1/trt7|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|mkl|5.4|cuda10.2/cudnn8.1/trt7|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|mkl|8.2|cuda11.1/cudnn8.1/trt7|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|mkl|5.4|cuda11.1/cudnn8.1/trt7|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.1_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz)|
|GPU|是|mkl|8.2|cuda11.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)|
|GPU|是|mkl|5.4|cuda11.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz)|
|nv-jetson(all)|-|-|-|cuda10.2/cudnn8.0/trt7|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.5_gcc7.5/all/paddle_inference_install_dir.tgz)|
|nv-jetson(nano)|-|-|-|cuda10.2/cudnn8.0/trt7|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.5_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|nv-jetson(tx2)|-|-|-|cuda10.2/cudnn8.0/trt7|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.5_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|nv-jetson(xavier)|-|-|-|cuda10.2/cudnn8.0/trt7|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.5_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|nv-jetson(all)|-|-|-|cuda10.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6_gcc7.5/all/paddle_inference_install_dir.tgz)|
|nv-jetson(nano)|-|-|-|cuda10.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|nv-jetson(tx2)|-|-|-|cuda10.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6_gcc7.5/tx2/paddle_inference_install_dir.tgz)|
|nv-jetson(xavier)|-|-|-|cuda10.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6_gcc7.5/xavier/paddle_inference_install_dir.tgz)|
|nv-jetson(all)|-|-|-|cuda10.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/all/paddle_inference_install_dir.tgz)|
|nv-jetson(nano)|-|-|-|cuda10.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/nano/paddle_inference_install_dir.tgz)|
|nv-jetson(tx2)|-|-|-|cuda10.2/cudnn8.2/trt8|[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Jetson/jetpack4.6.1_gcc7.5/tx2/paddle_inference_install_dir.tgz)|

- [下载安装 Windows 预测库](../user_guides/download_lib.html#windows)

| 硬件后端 | 是否使用avx |     编译器     |  cu da/cudnn版本  | 数学库  |预测库(2.3.0版本)   |  CUDA  |
|--------------|--------------|:----------------|:--------|:-------------|:-----------------|:-------|
| CPU | 是 |  MSVC 2017 | - |mkl|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/CPU/x86-64_vs2017_avx_mkl/paddle_inference.zip)| - |
| CPU | 是 | MSVC 2017 | - |openblas|[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/CPU/x86-64_vs2017_avx_openblas/paddle_inference.zip)| - |
| GPU | 是 | MSVC 2017  | cuda10.1/cudnn7.6/no_trt | mkl                                          |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7/paddle_inference_notrt.zip)|  10.1 |
| GPU | 是 | MSVC 2017  | cuda10.1/cudnn7.6/trt6 |mkl |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7/paddle_inference.zip)|  10.1 |
| GPU | 是 | MSVC 2017  | cuda10.2/cudnn7.6/trt7 |mkl |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7/paddle_inference.zip)|  10.2 |
| GPU | 是 | MSVC 2017  | cuda11.0/cudnn8.0/trt7 |mkl |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.0_cudnn8/paddle_inference.zip)| 11.0 |
| GPU | 是 | MSVC 2017  | cuda11.2/cudnn8.2/trt8 |mkl |[paddle_inference.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.2_cudnn8/paddle_inference.zip)| 11.2 |

- [下载安装 MacOs预测库](../user_guides/download_lib.html#mac)

|硬件后端 |是否打开avx |数学库 |预测库(2.3.0版本)   |
|----------|----------|----------|:----------------|
|CPU |是 |openblas |[paddle_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/MacOS/CPU/x86-64_clang_avx_openblas/paddle_inference_install_dir.tgz)|

下载完成并解压之后，目录下的 `paddle_inference_install_dir` 即为 C++ 预测库，目录结构如下：

```bash
paddle_inference/paddle_inference_install_dir/
├── CMakeCache.txt
├── paddle
│   ├── include                                    C++ 预测库头文件目录
│   │   ├── crypto
│   │   ├── internal
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h                 C++ 预测库头文件
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   └── paddle_pass_builder.h
│   └── lib
│       ├── libpaddle_inference.a                      C++ 静态预测库文件
│       └── libpaddle_inference.so                     C++ 动态态预测库文件
├── third_party
│   ├── install                                    第三方链接库和头文件
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── mkldnn
│   │   ├── mklml
│   │   ├── protobuf
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt
```

include 目录下包括了使用飞桨预测库需要的头文件，lib 目录下包括了生成的静态库和动态库，third_party 目录下包括了预测库依赖的其它库文件。

其中 `version.txt` 文件中记录了该预测库的版本信息，包括 Git Commit ID、使用 OpenBlas 或 MKL 数学库、CUDA/CUDNN 版本号，如：

```bash
GIT COMMIT ID: 1bf4836580951b6fd50495339a7a75b77bf539f6
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
CUDA version: 9.0
CUDNN version: v7.6
CXX compiler version: 4.8.5
WITH_TENSORRT: ON
TensorRT version: v6
```

## 验证安装

您可以编写应用代码，与预测库联合编译并测试结果。请参考 [预测示例(C++)](../quick_start/cpp_demo) 一节。

## 开始使用

请参考 [预测示例(C++)](../quick_start/cpp_demo) 和 [C++ API 文档](../api_reference/cpp_api_index)。
