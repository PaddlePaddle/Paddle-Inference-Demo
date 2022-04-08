## 运行Paddle-IPU进行ResNet50图像分类样例

该文档为使用Paddle-IPU预测在ResNet50分类模型上的实践demo。

### 获取paddle_inference预测库

通过源码编译的方式安装，源码编译方式参考官网文档，注意这里cmake编译时打开`-DON_INFER=ON`,在编译目录下得到`paddle_inference_install_dir`，将其存储到`Paddle-Inference-Demo/c++/lib`目录，并更名为`paddle_inference`，lib目录结构如下所示

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
    │   │   ├── paddle_infer_contrib.h
    │   │   ├── paddle_infer_declare.h
    │   │   ├── paddle_inference_api.h                 C++ 预测库头文件
    │   │   ├── paddle_mkldnn_quantizer_config.h
    │   │   ├── paddle_pass_builder.h
    │   │   └── paddle_tensor.h
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
    │   │   ├── utf8proc
    │   │   └── xxhash
    │   └── threadpool
    │       └── ThreadPool.h
    └── version.txt
```


### 获取Resnet50模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。
### **样例编译**
 
文件`resnet50_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。   
脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
脚本`run.sh` 一键运行脚本。

编译Resnet50样例，我们首先需要对脚本`compile.sh` 文件中的配置进行修改。

1）**修改`compile.sh`**

打开`compile.sh`，我们对以下的几处信息进行修改：

```shell
# 编译的 demo 名称，resnet50_test
DEMO_NAME=resnet50_test

# 根据预编译库中的version.txt信息判断是否将以下标记打开
WITH_MKL=ON

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference
```

运行 `bash compile.sh`， 会在目录下产生build目录。


2） **运行样例**

```shell
# 如果使用IPU运行程序，请source对应PopLAR SDK中的 PopLAR 和 PopART

# 修改 run.sh，增加运行参数 --use_ipu true
bash run.sh

# 或者

bash compile.sh
# 增加运行参数 --use_ipu true
./build/resnet50_test -model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --use_ipu true
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

### 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
