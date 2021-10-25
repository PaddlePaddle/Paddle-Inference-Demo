## 使用 Paddle-TRT 运行 Ernie 变长输入样例

该文档为使用 Paddle-TRT 运行 Ernie 变长输入的样例，重点关注 TensorRT8 的稀疏特性带来的性能提升。如果你想获取更多 **TensorRT8稀疏特性** 相关的信息，请访问[这里](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)。

### 获取 paddle_inference 预测库

下载 paddle_inference 预测库并解压存储到 `Paddle-Inference-Demo/c++/lib` 目录，lib 目录结构如下所示

```
Paddle-Inference-Demo/c++/lib/
├── CMakeLists.txt
└── paddle_inference
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
    │       ├── libpaddle_inference.a                  C++ 静态预测库文件
    │       └── libpaddle_inference.so                 C++ 动态态预测库文件
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

本目录下，

- 文件 `ernie_varlen_test.cc` 为预测的样例程序。
- 脚本 `compile.sh` 包含了第三方库、预编译库的信息配置。
- 脚本 `run.sh` 一键运行脚本。

### 获取模型和数据
首先，我们从下列链接下载所需模型和数据：

[模型](https://drive.google.com/file/d/1RJeWVfbsXRt6a8gMb86zuhCty0GJ5biK/view?usp=sharing)

[数据](https://drive.google.com/file/d/1Q_SOngP1qMGt7j5nJvmaRxEQDufrwugm/view?usp=sharing)

模型中包含 `ERNIE 2.0 Base` 与 `ERNIE 2.0 Large` 的普通模型和稀疏模型。解压模型和数据，此时，目录下的文件如下：
```
Paddle-Inference-Demo/c++/sparsity/ernie_varlen
├── models
├── data
├── ernie_varlen_test.cc
├── compile.sh                                   
└── run.sh
```

### 样例编译与运行

1）**修改`compile.sh`**

打开 `compile.sh`，我们对以下的几处信息进行修改：

```shell
DEMO_NAME=ernie_varlen_test

# 配置预测库的根目录
LIB_DIR=${work_path}/../../lib/paddle_inference

# 设置 CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-8.0.3.4
```

运行 `bash compile.sh`， 会在目录下产生build目录。


2） **运行样例**

通过 `bash run.sh`，运行样例。

### 测试结果
本样例通过 QPS 计算性能提升。测试环境如下：
```
机器显卡：A100
显卡版本：470.57.02
CUDA版本：11.3
trt版本：8.0.3.4
cudnn版本：8.2.1
```
| batch_size |    1   |    2   |    4   |    8   |   16   |   32   |   64   |  128   |  256   |
|   :----:   | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   dense    |297.977 |529.827 |978.298 |1331.87 |2670.03 |4571.65 |4571.65 |4992.79 |5312.89 |
|   sparse   |297.977 |529.827 |978.298 |1331.87 |2670.03 |4571.65 |4571.65 |4992.79 |5312.89 |
|  speedup   |1.0967  |1.0962  |1.0919  |1.4028  |1.1012  |1.2811  |1.3431  |1.3499  |1.3332  |
