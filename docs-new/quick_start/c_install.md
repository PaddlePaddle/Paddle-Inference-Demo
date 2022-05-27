# C 推理部署

本文主要介绍 Paddle Inference C API 的安装。主要分为以下三个章节：环境准备、安装步骤和验证安装。三个章节分别说明了安装前的环境要求、安装的具体流程和成功安装后的验证方法。

## 环境准备

- G++ 8.2
- CMake 3.0+
- Visual Studio 2017 Update 3 （仅在使用 Windows 版本的预测库时需要，根据 Paddle 预测库所使用的 VS 版本选择，请参考 [Visual Studio 不同版本二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=msvc-170&viewFallbackFrom=vs-2019) )
- CUDA 10.1 / CUDA 10.2 / CUDA 11.0 / CUDA 11.2, cuDNN7.6+, TensorRT （仅在使用 gpu 版本的预测库时需要）

（您可参考 nvidia 官方文档了解 CUDA 和 cuDNN 的安装流程和配置方法，请见 [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，版本对应关系如下表所示)

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.1|7.6|6|
|10.2|7.6|7|
|11.0|8.0|7|
|11.2|8.2|8|

## 开始安装

Paddle Inference 提供了 Linux/Windows/MacOS 平台的官方 Release 预测库下载，如果您使用的是以上平台，我们优先推荐您通过以下链接直接下载，或者您也可以参照文档进行[源码编译](../user_guides/source_compile.html)。

- [下载安装 Linux C 预测库](../user_guides/download_lib.html#id1)

|硬件后端| 是否打开avx | 数学库   | gcc版本 | CUDA/cuDNN版本 |预测库(2.3.0版本)|
|----------|----------|----------|----------|:---------|:--------------|
|CPU|是| mkl      |8.2|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_avx_mkl/paddle_inference_c.tgz)|
|CPU|是|mkl|5.4|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_avx_mkl/paddle_inference_c.tgz)|
|CPU|是|openblas|8.2|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_avx_openblas/paddle_inference_c.tgz)|
|CPU|否|openblas|5.4| - |[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference_c.tgz)|
|CPU|否|openblas|8.2| - |[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc8.2_openblas/paddle_inference_c.tgz)|
|CPU|否|openblas|5.4|-|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/CPU/gcc5.4_openblas/paddle_inference_c.tgz)|
|GPU|是|是|8.2|CUDA10.1\cu7.6\trt6|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference_c.tgz)|
|GPU|是|是|5.4|CUDA10.1\cu7.6\trt6|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference_c.tgz)|
|GPU|是|是|5.4|CUDA10.2\cu7.6\trt6|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddle_inference_c.tgz)|


- [下载安装 Windows C 预测库](../user_guides/download_lib.html#id3)

| 硬件后端 |是否打开avx | 数学库  |     编译器版本     | CUDA/cuDNN版本  |预测库(2.3.0版本)   |
|----------|:--------|:---------|:--------------|:---------|:-----------------|
| CPU |是 |mkl|  MSVC 2017 | - | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/CPU/x86-64_vs2017_avx_mkl/paddle_inference_c.zip)|
| CPU |是 |openblas| MSVC 2017 | - | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/CPU/x86-64_vs2017_avx_openblas/paddle_inference_c.zip)|
| GPU |是 |mkl | MSVC 2017|CUDA10.1\cuDNN7.6\no_trt | [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7/paddle_inference_c_notrt.zip) |
| GPU |是 |mkl| MSVC 2017|CUDA10.1\cuDNN7.6\trt6|[paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.1_cudnn7/paddle_inference_c.zip)|
| GPU |是 |mkl | MSVC 2017 |CUDA10.2\cuDNN7.6\trt7| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7/paddle_inference_c.zip) |
| GPU |是 |mkl | MSVC 2017 |CUDA11.0\cuDNN8.0\trt7| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.0_cudnn8/paddle_inference_c.zip) |
| GPU |是 |mkl | MSVC 2017 |CUDA11.2\cuDNN8.2\trt8| [paddle_inference_c.zip](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Windows/GPU/x86-64_vs2017_avx_mkl_cuda11.2_cudnn8/paddle_inference_c.zip) |

- [下载安装 MacOs C 预测库](../user_guides/download_lib.html#id6)

|硬件后端 |是否打开avx | 数学库      |预测库(2.3.0版)   |
|----------|----------|:---------|:----------------|
|CPU |是 |openblas|[paddle_inference_c.tgz](https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/MacOS/CPU/x86-64_clang_avx_openblas/paddle_inference_c_install_dir.tgz)|

下载完成并解压之后，目录下的 `paddle_inference_c_install_dir` 即为 C 预测库，目录结构如下：

```
paddle_inference_c_install_dir
├── paddle
│   ├── include               C 预测库头文件目录
│   │   └── pd_common.h
│   │   └── pd_config.h
│   │   └── pd_inference_api.h         C 预测库头文件
│   │   └── pd_predictor.h
│   │   └── pd_tensor.h
│   │   └── pd_types.h
│   │   └── pd_utils.h
│   └── lib
│       ├── libpaddle_inference_c.a          C 静态预测库文件
│       └── libpaddle_inference_c.so         C 动态预测库文件
├── third_party
│   └── install                          第三方链接库和头文件
│       ├── cryptopp
│       ├── gflags
│       ├── glog
│       ├── mkldnn
│       ├── mklml
│       ├── protobuf
│       └── xxhash
└── version.txt                          版本信息与编译选项信息
```

include 目录下包括了使用飞桨预测库需要的头文件，lib 目录下包括了生成的静态库和动态库，third_party 目录下包括了预测库依赖的其它库文件。

其中 `version.txt` 文件中记录了该预测库的版本信息，包括 Git Commit ID、使用 OpenBlas 或 MKL 数学库、CUDA/cuDNN 版本号，如：

```bash
GIT COMMIT ID: 1bf4836580951b6fd50495339a7a75b77bf539f6
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
CUDA version: 9.0
cuDNN version: v7.6
CXX compiler version: 4.8.5
WITH_TENSORRT: ON
TensorRT version: v6
```

## 验证安装

您可以编写应用代码，与预测库联合编译并测试结果。请参考 [预测示例(C)](../quick_start/c_demo) 一节。

## 开始使用

请参考 [预测示例(C)](../quick_start/c_demo) 和 [C API 文档](../api_reference/c_api_index)。
