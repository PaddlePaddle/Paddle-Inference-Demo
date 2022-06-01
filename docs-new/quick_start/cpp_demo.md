# 推理示例 (C++)

本章节包含2部分内容
- 运行 C++ 示例程序
- C++ 推理程序开发说明

## 运行 C++ 示例程序

在此环节中，共包含以下3个步骤，
- 1. 下载预编译 C++ 推理库
- 2. 获取推理示例代码并编译
- 3. 执行推理程序

### 1. 下载预编译 C++ 推理库

Paddle Inference 提供了 Ubuntu/Windows/MacOS/Jetson 平台的官方 Release 推理库下载，如果使用的是以上平台，推荐通过以下链接直接下载，或者也可以参考[源码编译](../user_guides/source_compile.html)文档自行编译。

- [下载安装 Linux 推理库](../user_guides/download_lib.html#linux)
- [下载安装 Windows 推理库](../user_guides/download_lib.html#windows)
- [下载安装 Mac 推理库](../user_guides/download_lib.html#mac)

下载完成并解压之后，目录下的 `paddle_inference_install_dir` 即为 C++ 推理库，目录结构如下：

```bash
paddle_inference
├── CMakeCache.txt
├── paddle
│   ├── include                              C++ 推理库头文件目录
│   │   ├── crypto
│   │   ├── experimental
│   │   ├── internal
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_contrib.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h           C++ 推理库头文件
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   ├── paddle_pass_builder.h
│   │   └── paddle_tensor.h
│   └── lib
│       ├── libpaddle_inference.a
│       └── libpaddle_inference.so
├── third_party                              第三方链接库和头文件
│   ├── install
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── mkldnn
│   │   ├── mklml
│   │   ├── onnxruntime
│   │   ├── paddle2onnx
│   │   ├── protobuf
│   │   ├── utf8proc
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt
```

其中 `version.txt` 文件中记录了该推理库的版本信息，包括 Git Commit ID、使用 OpenBlas 或 MKL 数学库、CUDA/CUDNN 版本号，如：

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

### 2. 获取推理示例代码并编译

本章节 C++ 推理示例代码位于 [Paddle-Inference-Demo/c++/resnet50](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++/resnet50)。

```
# 获取部署 Demo 代码库
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git
```

其中以ResNet50为例，Linux Demo 代码在如下路径

```bash
Paddle-Inference-Demo/c++/resnet50/
├── resnet50_test.cc         推理 C++ 源码程序
├── README.md                README 说明
├── compile.sh               编译脚本
└── run.sh                   运行脚本 
```

编译运行推理样例之前，先下载对应推理库并解压，将解压后的 paddle_inference 目录移至 Paddle-Inference-Demo/c++/lib 目录下。

编译时，根据部署的环境和硬件，编辑`compile.sh`，其中各参数含义如下所示，

```bash
# 根据预编译库中的 version.txt 信息判断是否将以下三个标记打开
WITH_MKL=ON       
WITH_GPU=ON         
USE_TENSORRT=OFF

# 配置推理库的根目录，即为本章节第1步中下载/编译的 C++ 推理库，可重命名为 paddle_inference 后置于 ../lib 目录下
LIB_DIR=${work_path}/../lib/paddle_inference

# 如果上述的 WITH_GPU 或 USE_TENSORRT 设为ON，请设置对应的 CUDA, CUDNN, TENSORRT 的路径，例如
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-6.0.1.5
```

运行脚本进行编译，会在目录下产生 `build` 目录，并生成 `build/resnet50_test` 可执行文件

```bash
bash compile.sh
```

### 3. 执行推理程序

**注意**：Paddle Inference 提供下载的 C++ 推理库对应的 GCC 版本与您电脑中GCC版本需要一致，如果不一致可能出现未知错误。

运行脚本 `run.sh` 执行推理程序。

```bash
bash run.sh
```

脚本说明：
```bash
# 脚本 run.sh 会首先下载推理部署模型，如需查看模型结构，可将 `inference.pdmodel` 加载到可视化工具 Netron 中打开。
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 加载下载的模型，执行推理程序
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```

成功执行之后，得到的推理输出结果如下：

```bash
# 程序输出结果如下
I1202 06:53:18.979496  3411 resnet50_test.cc:73] run avg time is 257.678 ms
I1202 06:53:18.979645  3411 resnet50_test.cc:88] 0 : 0
I1202 06:53:18.979676  3411 resnet50_test.cc:88] 100 : 2.04164e-37
I1202 06:53:18.979728  3411 resnet50_test.cc:88] 200 : 2.12382e-33
I1202 06:53:18.979768  3411 resnet50_test.cc:88] 300 : 0
I1202 06:53:18.979779  3411 resnet50_test.cc:88] 400 : 1.68493e-35
I1202 06:53:18.979794  3411 resnet50_test.cc:88] 500 : 0
I1202 06:53:18.979802  3411 resnet50_test.cc:88] 600 : 1.05767e-19
I1202 06:53:18.979810  3411 resnet50_test.cc:88] 700 : 2.04094e-23
I1202 06:53:18.979820  3411 resnet50_test.cc:88] 800 : 3.85254e-25
I1202 06:53:18.979828  3411 resnet50_test.cc:88] 900 : 1.52393e-30
```

## C++ 推理程序开发说明

使用 Paddle Inference 开发 C++ 推理程序仅需以下五个步骤：


(1) 引用头文件

```c++
#include "paddle/include/paddle_inference_api.h"
```

(2) 创建配置对象，并根据需求配置，详细可参考 [C++ API 文档 - Config](../api_reference/cxx_api_doc/Config_index)

```c++
// 创建默认配置对象
paddle_infer::Config config;

// 设置推理模型路径，即为本小节第2步中下载的模型
config.SetModel(FLAGS_model_file, FLAGS_params_file);

// 启用 GPU 和 MKLDNN 推理
config.EnableUseGpu(100, 0);
config.EnableMKLDNN();

// 开启 内存/显存 复用
config.EnableMemoryOptim();
```

(3) 根据 Config 创建推理对象，详细可参考 [C++ API 文档 - Predictor](../api_reference/cxx_api_doc/Predictor)

```c++
auto predictor = paddle_infer::CreatePredictor(config);
```

(4) 设置模型输入 Tensor，详细可参考 [C++ API 文档 - Tensor](../api_reference/cxx_api_doc/Tensor)

```c++
// 获取输入 Tensor
auto input_names = predictor->GetInputNames();
auto input_tensor = predictor->GetInputHandle(input_names[0]);

// 设置输入 Tensor 的维度信息
std::vector<int> INPUT_SHAPE = {1, 3, 224, 224};
input_tensor->Reshape(INPUT_SHAPE);

// 准备输入数据
int input_size = 1 * 3 * 224 * 224;
std::vector<float> input_data(input_size, 1);
// 设置输入 Tensor 数据
input_tensor->CopyFromCpu(input_data.data());
```

(5) 执行推理，详细可参考 [C++ API 文档 - Predictor](../api_reference/cxx_api_doc/Predictor)

```c++
// 执行推理
predictor->Run();
```

(6) 获得推理结果，详细可参考 [C++ API 文档 - Tensor](../api_reference/cxx_api_doc/Tensor)

```c++
// 获取 Output Tensor
auto output_names = predictor->GetOutputNames();
auto output_tensor = predictor->GetOutputHandle(output_names[0]);

// 获取 Output Tensor 的维度信息
std::vector<int> output_shape = output_tensor->shape();
int output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());

// 获取 Output Tensor 的数据
std::vector<float> output_data;
output_data.resize(output_size);
output_tensor->CopyToCpu(output_data.data());
```
