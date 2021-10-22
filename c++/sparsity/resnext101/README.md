## 使用Paddle-TRT运行ResNexXt101样例

该文档为使用Paddle-TRT运行ResNexXt101的样例，重点关注TensorRT8的稀疏特性带来的性能提升。运行环境如下：
```
机器显卡：A100
显卡版本：470.57.02
CUDA版本：11.3
trt版本：8.0.3.4
cudnn版本：8.2.1
```
如果你想获取更多**TensorRT8稀疏特性**相关的信息，请访问[这里](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)。

### 获取paddle_inference预测库

下载paddle_inference预测库并解压存储到`Paddle-Inference-Demo/c++/lib`目录，lib目录结构如下所示

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

- 文件`trt_fp16_test.cc` 为预测的样例程序。
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
- 脚本`run.sh` 一键运行脚本。

### 获取模型
首先，我们从下列链接下载所需模型和数据：

[模型](https://drive.google.com/file/d/1RJeWVfbsXRt6a8gMb86zuhCty0GJ5biK/view?usp=sharing)

解压模型，此时，目录下的文件如下：
```
Paddle-Inference-Demo/c++/sparsity/resnext101
├── models
├── trt_fp16_test.cc
├── compile.sh                                   
└── run.sh
```

### 样例编译与运行

1）**修改`compile.sh`**

打开`compile.sh`，我们对以下的几处信息进行修改：

```shell
DEMO_NAME=trt_fp16_test

# 配置预测库的根目录
LIB_DIR=${work_path}/../../lib/paddle_inference

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-8.0.3.4
```

运行 `bash compile.sh`， 会在目录下产生build目录。


2） **运行样例**

通过 `bash run.sh`，运行样例。

### 测试结果
| batch_size |    1   |    2   |    4   |    8   |   16   |   32   |   64   |  128   |  256   |
|   :----:   | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   dense    |297.977 |529.827 |978.298 |1331.87 |2670.03 |4571.65 |4571.65 |4992.79 |5312.89 |
|   sparse   |297.977 |529.827 |978.298 |1331.87 |2670.03 |4571.65 |4571.65 |4992.79 |5312.89 |
|  speedup   |1.0967  |1.0962  |1.0919  |1.4028  |1.1012  |1.2811  |1.3431  |1.3499  |1.3332  |
