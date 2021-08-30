## 子图运行ASCEND-310 

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


### 获取mobilenet_v1_fp32_224模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/ascend310_clas_assets.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。
### **编译样例**

文件`demo.cc` 为预测的样例程序。    
脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
脚本`run.sh` 一键运行脚本。

1）**修改`compile.sh`**

打开`compile.sh`，我们对以下信息进行修改：

```shell
# 配置预测库的根目录
LIB_DIR=${work_path}/../../lib/paddle_inference
```

运行 `bash compile.sh`， 会在目录下产生build目录。


2） **运行样例**

```shell
bash run.sh
```

运行成功后，程序会将模型结果打印到屏幕上，说明运行成功。模型结果的关键信息如下：
```shell
warmup: 1 repeat: 1, average: 1.555000 ms, max: 1.555000 ms, min: 1.555000 ms
    results: 3
    Top0  tabby, tabby cat - 0.529785
    Top1  Egyptian cat - 0.418945
    Top2  tiger cat - 0.045227
    Preprocess time: 0.605000 ms
    Prediction time: 1.555000 ms
    Postprocess time: 0.093000 ms
```


### 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
