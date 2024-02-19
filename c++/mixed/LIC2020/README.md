# 运行C++ LIC2020关系抽取 demo

该工程是[2020语言与智能技术竞赛：关系抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/31)竞赛提供的基线模型的c++预测demo，基线模型的训练相关信息可参考[LIC2020-关系抽取任务基线系统](https://aistudio.baidu.com/aistudio/projectdetail/357344)。

## 获取paddle_inference预测库

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

## 获取基线模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/lic_model.tgz)下载模型，如果你想获取更多的**模型训练信息**，请访问[LIC2020-关系抽取任务基线系统](https://aistudio.baidu.com/aistudio/projectdetail/357344)。

```
wget https://paddle-inference-dist.bj.bcebos.com/lic_model.tgz
tar xzf lic_model.tgz
```

## **样例编译**
 
文件`demo.cc` 为预测的样例程序（程序中的输入为固定值，如果您有其他方式进行数据读取的需求，需要对程序进行一定的修改）。
脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
脚本`run.sh` 一键运行脚本。

编译demo样例，我们首先需要对脚本`compile.sh` 文件中的配置进行修改。

1）**修改`compile.sh`**

打开`compile.sh`，我们对以下的几处信息进行修改：

```shell
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-7.2.3.4
```

运行 `bash compile.sh`， 会在目录下产生build目录。

2） **运行样例**

```shell
bash run.sh
# 或者
bash compile.sh
./build/demo -model_file lic_model//model --params_file lic_model//params
```

运行结束后，程序会将模型输出个数打印到屏幕，说明运行成功。

## 更多链接
- [Paddle Inference使用Quick Start！](https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/index_quick_start.html)
- [Paddle Inference C++ Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/python_api_index.html)
