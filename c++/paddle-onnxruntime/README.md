## 使用Paddle-ONNXRuntime进行MobileNetV2图像分类样例

该文档为使用Paddle-ONNXRuntime预测在MobileNetV2分类模型上的实践demo，使用Paddle-ONNXRuntime预测跟使用Paddle MKLDNN推理类似，详细代码可参考本目录里的onnxruntime_mobilenet_demo.cc代码。

**注意**:当前Paddle-ONNXRuntime只支持CPU上推理

### 获取paddle_inference预测库

下载paddle_inference预测库并解压存储到`Paddle-Inference-Demo/c++/lib`目录。lib目录结构如下所示

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
    │   │   ├── paddle2onnx
    │   │   ├── onnxruntime
    │   │   └── xxhash
    │   └── threadpool
    │       └── ThreadPool.h
    └── version.txt
```

本目录下，

- `onnxruntime_mobilenet_demo.cc` 为使用Paddle-ONNXRuntime进行预测的样例程序源文件（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
- 脚本`run.sh` 一键运行脚本。

### 获取模型
首先，我们从下列链接下载所需模型：

[MobileNetV2 模型](http://paddle-inference-dist.bj.bcebos.com/MobileNetV2.inference.model.tar.gz)

### 运行样例

```shell
bash run.sh
```

该脚本会自动编译、下载模型并运行程序进行推理。运行结束后，程序会将模型预测输出的前20个结果打印到屏幕，说明运行成功。

