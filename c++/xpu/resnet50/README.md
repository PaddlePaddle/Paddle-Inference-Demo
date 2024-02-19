# 昆仑 XPU 运行 ResNet50 图像分类样例

ResNet50 样例展示了单输入模型在昆仑 XPU 下的推理过程。运行步骤如下：

## 一、获取 Paddle Inference 预测库

当前仅支持通过源码编译的方式安装，源码编译方式参考 Paddle Inference 官网文档的硬件支持部分。

编译完成之后在编译目录下将会生成 Paddle Inference 的 C++ 预测库，即为编译目录下的 `paddle_inference_install_dir` 文件夹。将其软链接或重命名为 `paddle_inference`，并置于 `Paddle-Inference-Demo/c++/lib` 目录下，目录结构如下：

```bash
Paddle-Inference-Demo/c++/lib/
├── CMakeLists.txt
└── paddle_inference
    ├── CMakeCache.txt
    ├── paddle
    │   ├── include                                    # C++ 预测库头文件目录
    │   │   ├── crypto
    │   │   ├── experimental
    │   │   ├── internal
    │   │   ├── paddle_analysis_config.h
    │   │   ├── paddle_api.h
    │   │   ├── paddle_infer_contrib.h
    │   │   ├── paddle_infer_declare.h
    │   │   ├── paddle_inference_api.h                 # C++ 预测库头文件
    │   │   ├── paddle_mkldnn_quantizer_config.h
    │   │   ├── paddle_pass_builder.h
    │   │   └── paddle_tensor.h
    │   └── lib
    │       ├── libpaddle_inference.a                  # C++ 静态预测库文件
    │       └── libpaddle_inference.so                 # C++ 动态态预测库文件
    ├── third_party
    │   ├── install
    │   │   ├── cryptopp
    │   │   ├── gflags
    │   │   ├── glog
    │   │   ├── openblas
    │   │   ├── protobuf
    │   │   ├── utf8proc
    │   │   ├── xpu
    │   │   └── xxhash
    │   └── threadpool
    │       └── ThreadPool.h
    └── version.txt                                    # 预测库版本信息
```

## 二：获取 Resnet50 模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。

## 三：编译样例
 
- 文件`resnet50_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
- 脚本`run.sh` 为一键运行脚本。

编译前，需要根据自己的环境修改 `compile.sh` 中的相关代码配置依赖库：

```bash
# 根据预编译库中的version.txt信息判断是否将以下标记打开
WITH_MKL=ON  # 这里如果是 Aarch64 环境，则改为 OFF
WITH_ARM=OFF # 这里如果是 Aarch64 环境，则改为 ON
WITH_XPU=ON
```

运行 `bash compile.sh` 编译样例。

## 四：运行样例

```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```
运行结束后，程序会将模型结果打印到屏幕，说明运行成功，预期得到如下的输出结果：

```bash
W0525 20:56:43.035851 95178 xpu_context.cc:89] Please NOTE: xpu device: 0
W0525 20:56:43.035950 95178 device_context.cc:310] Please NOTE: xpu device: 0
I0525 20:56:43.083045 95178 resnet50_test.cc:79] run avg time is 46.773 ms
I0525 20:56:43.083097 95178 resnet50_test.cc:116] 0 : 6.93194e-15
I0525 20:56:43.083169 95178 resnet50_test.cc:116] 100 : 6.93194e-15
... ...
I0525 20:56:43.083432 95178 resnet50_test.cc:116] 800 : 6.93194e-15
I0525 20:56:43.083436 95178 resnet50_test.cc:116] 900 : 6.93194e-15
```

## 更多链接
- [Paddle Inference使用Quick Start！](https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/index_quick_start.html)
- [Paddle Inference C++ Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/python_api_index.html)
