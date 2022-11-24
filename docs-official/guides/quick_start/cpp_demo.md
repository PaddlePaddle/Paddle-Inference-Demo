# 快速上手C++推理

本章节包含2部分内容,
- [运行 C++ 示例程序](#id1)
- [C++ 推理程序开发说明](#id2)

注意本章节文档和代码仅适用于Linux系统。

## 运行 C++ 示例程序

在此环节中，共包含以下5个步骤，
- 环境准备
- 模型准备
- 推理代码
- 编译代码
- 执行程序

### 1. 环境准备 

Paddle Inference 提供了 Ubuntu/Windows/MacOS/Jetson 平台的官方 Release 推理库，用户需根据开发环境和硬件自行下载安装，具体可参阅[C++推理环境安装](../install/cpp_install.html)。

### 2. 模型准备

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 推理格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可参考[模型结构可视化文档](../export_model/visual_model.rst)。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得模型目录即文件如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

### 3. 推理代码
本章节 C++ 推理示例代码位于[Paddle-Inference-Demo/c++/cpu/resnet50](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/cpu/resnet50)。
```
# 获取部署 Demo 代码库
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git
cd Paddle-Inference-Demo/c++/cpu/resnet50
```
其中示例代码目录结构如下所示
```
Paddle-Inference-Demo/c++/resnet50/
├── resnet50_test.cc         推理 C++ 源码程序
├── README.md                README 说明
├── compile.sh               编译脚本
└── run.sh                   运行脚本 
```

### 4. 编译代码

在编译前，
- 将**第1步环境准备**下载解压后的预测库`paddle_inference`目录(如解压后的目录名称不同，也需重命名为`paddle_inference`)拷贝至`Paddle-Inference-Demo/c++/lib`目录下
- 将**第2步模型准备**下载解压后的模型目录`resnet50`目录拷贝至`Paddle-Inference-Demo/c++/cpu/resnet50`目录下

执行如下命令进行编译
```
bash compile.sh
```
编译后的二进制即在`Paddle-Inference-Demo/c++/cpu/resnet50/build`目录下

编译前，可根据部署的环境和硬件编辑`compile.sh`，配置推理方式。其中各参数含义如下所示，

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

### 5. 执行程序

使用如下命令执行推理程序
```
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```
如若推理时，提示找不到`.so`的问题，可将各个依赖动态库拷贝到执行路径（也可以将依赖库路径加入到环境变量中），再执行上述命令
```
# 将paddle inference中的动态库拷贝到执行路径下
find ../../lib/paddle_inference/ -name "*.so*" | xargs -i cp {} .
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

(2) 创建配置对象，并根据需求配置，详细可参考 [C++ API 文档 - Config](../../api_reference/cxx_api_doc/Config_index.rst)

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

(3) 根据 Config 创建推理对象，详细可参考 [C++ API 文档 - Predictor](../../api_reference/cxx_api_doc/Predictor.md)

```c++
auto predictor = paddle_infer::CreatePredictor(config);
```

(4) 设置模型输入 Tensor，详细可参考 [C++ API 文档 - Tensor](../../api_reference/cxx_api_doc/Tensor.md)

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

(5) 执行推理，详细可参考 [C++ API 文档 - Predictor](../../api_reference/cxx_api_doc/Predictor.md)

```c++
// 执行推理
predictor->Run();
```

(6) 获得推理结果，详细可参考 [C++ API 文档 - Tensor](../../api_reference/cxx_api_doc/Tensor.md)

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

至此 Paddle Inference 推理已跑通，如果想更进一步学习 Paddle Inference，可以根据硬件情况选择学习 GPU 推理、CPU 推理、进阶使用等章节。
