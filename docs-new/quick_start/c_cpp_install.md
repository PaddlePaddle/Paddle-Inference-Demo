# C/C++ 推理部署

本文主要介绍 Paddle Inferrence C/C++ API 的安装。主要分为以下三个章节：环境准备，安装步骤，和验证安装。

## 环境准备

- G++ 8.2
- CMake 3.0+
- CUDA 10.1 / CUDA 10.2 / CUDA 11.0 / CUDA 11.1 / CUDA 11.2, cudnn 7+ （仅在使用GPU版本的预测库时需要）
- Visual Studio 2019 (仅在使用Windows版本的预测库时需要，根据Paddle预测库所使用的VS版本选择，请参考 [Visual Studio 不同版本二进制兼容性]() )


## 开始安装

Paddle Inference 提供了 Ubuntu/Windows/MacOS 平台的官方 Release 预测库下载，如果您使用的是以上平台，我们优先推荐您通过以下链接直接下载，或者您也可以参照文档进行[源码编译](../user_guides/source_compile.html)。

- [下载安装 Linux 预测库](../user_guides/download_lib.html#linux)
- [下载安装 Windows 预测库](../user_guides/download_lib.html#windows)
- [下载安装 MacOs预测库]（）
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

