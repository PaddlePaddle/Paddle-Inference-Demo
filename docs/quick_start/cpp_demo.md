# 预测示例 (C++)

本章节包含2部分内容：(1) [运行 C++ 示例程序](#id1)；(2) [C++ 预测程序开发说明](#id6)。

## 运行 C++ 示例程序

### 1. 下载预编译 C++ 预测库

Paddle Inference 提供了 Ubuntu/Windows/MacOS 平台的官方Release预测库下载，如果您使用的是以上平台，我们优先推荐您通过以下链接直接下载，或者您也可以参照文档进行[源码编译](../user_guides/source_compile.html)。

- [下载安装Linux预测库](../user_guides/download_lib.html#linux)
- [下载安装Windows预测库](../user_guides/download_lib.html#windows)

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

其中 `version.txt` 文件中记录了该预测库的版本信息，包括Git Commit ID、使用OpenBlas或MKL数学库、CUDA/CUDNN版本号，如：

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

### 2. 准备预测部署模型

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 预测格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可将 `inference.pdmodel` 文件重命名为 `__model__`，然后通过模型可视化工具 Netron 打开。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得模型目录即文件如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

### 3. 获取预测示例代码并编译

本章节 C++ 预测示例代码位于 [Paddle-Inference-Demo/c++/resnet50](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++/resnet50)。目录包含以下文件：

```bash
-rw-r--r-- 1 root root 2.6K Dec 11 07:26 resnet50_test.cc    预测 C++ 源码程序
-rw-r--r-- 1 root root 7.4K Dec 11 07:26 CMakeLists.txt      CMAKE 文件
-rwxr-xr-x 1 root root  650 Dec 11 07:26 run_impl.sh         编译脚本
-rw-r--r-- 1 root root 2.2K Dec 11 07:26 README.md           README 说明
```

编译运行预测样例之前，需要根据运行环境配置编译脚本 `run_impl.sh`。

```bash
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON       
WITH_GPU=ON         
USE_TENSORRT=OFF

# 配置预测库的根目录，即为本章节第1步中下载/编译的 C++ 预测库
LIB_DIR=${YOUR_LIB_DIR}/paddle_inference_install_dir

# 如果上述的 WITH_GPU 或 USE_TENSORRT 设为ON，请设置对应的 CUDA, CUDNN, TENSORRT的路径，例如
CUDNN_LIB=/usr/lib/x86_64-linux-gnu
CUDA_LIB=/usr/local/cuda-10.2/lib64
```
运行脚本进行编译，会在目录下产生 `build` 目录，并生成 `build/resnet50_test` 可执行文件

```bash
bash run_impl.sh
```

### 3. 执行预测程序

**注意**：Paddle Inference 提供下载的C++预测库对应的 GCC 版本与您电脑中GCC版本需要一致，如果不一致可能出现未知错误。

运行脚本 `run.sh` 执行预测程序。

**注意**：执行预测之前，需要先将动态库文件 `libpaddle_inference.so` 所在路径加入 `LD_LIBRARY_PATH`，否则会出现无法找到库文件的错误。而且，Paddle Inference 提供下载的C++预测库对应GCC 4.8，所以请检查您电脑中GCC版本是否一致，如果不一致可能出现未知错误。

```bash
# 设置 LD_LIBRARY_PATH
LIB_DIR=${YOUR_LIB_DIR}/paddle_inference_install_dir
export LD_LIBRARY_PATH=${LIB_DIR}/paddle/lib:$LD_LIBRARY_PATH

# 参数输入为本章节第2步中下载的 ResNet50 模型
./build/resnet50_test --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams
```

成功执行之后，得到的预测输出结果如下：

```bash
# 程序输出结果如下
WARNING: Logging before InitGoogleLogging() is written to STDERR
E1211 08:29:39.840502 18792 paddle_pass_builder.cc:139] GPU not support MKLDNN yet
E1211 08:29:39.840647 18792 paddle_pass_builder.cc:139] GPU not support MKLDNN yet
E1211 08:29:40.997318 18792 paddle_pass_builder.cc:139] GPU not support MKLDNN yet
I1211 08:29:40.997367 18792 analysis_predictor.cc:139] Profiler is deactivated, and no profiling report will be generated.
I1211 08:29:41.016829 18792 analysis_predictor.cc:496] MKLDNN is enabled
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [conv_affine_channel_fuse_pass]
--- Running IR pass [conv_eltwiseadd_affine_channel_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
I1211 08:29:41.536377 18792 graph_pattern_detector.cc:101] ---  detected 53 subgraphs
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [embedding_eltwise_layernorm_fuse_pass]
--- Running IR pass [multihead_matmul_fuse_pass_v2]
--- Running IR pass [fc_fuse_pass]
I1211 08:29:41.577596 18792 graph_pattern_detector.cc:101] ---  detected 1 subgraphs
--- Running IR pass [fc_elementwise_layernorm_fuse_pass]
--- Running IR pass [conv_elementwise_add_act_fuse_pass]
I1211 08:29:41.599529 18792 graph_pattern_detector.cc:101] ---  detected 33 subgraphs
--- Running IR pass [conv_elementwise_add2_act_fuse_pass]
I1211 08:29:41.610285 18792 graph_pattern_detector.cc:101] ---  detected 16 subgraphs
--- Running IR pass [conv_elementwise_add_fuse_pass]
I1211 08:29:41.613446 18792 graph_pattern_detector.cc:101] ---  detected 4 subgraphs
--- Running IR pass [transpose_flatten_concat_fuse_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
I1211 08:29:41.620128 18792 ir_params_sync_among_devices_pass.cc:45] Sync params from CPU to GPU
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I1211 08:29:41.688971 18792 analysis_predictor.cc:541] ======= optimize end =======
I1211 08:29:41.689072 18792 naive_executor.cc:102] ---  skip [feed], feed -> image
I1211 08:29:41.689968 18792 naive_executor.cc:102] ---  skip [save_infer_model/scale_0.tmp_0], fetch -> fetch
W1211 08:29:41.690475 18792 device_context.cc:338] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 9.0
W1211 08:29:41.690726 18792 device_context.cc:346] device: 0, cuDNN Version: 7.6.
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1211 08:29:43.666896 18792 resnet50_test.cc:76] 0.000293902
I1211 08:29:43.667001 18792 resnet50_test.cc:76] 0.000453056
I1211 08:29:43.667009 18792 resnet50_test.cc:76] 0.000202802
I1211 08:29:43.667017 18792 resnet50_test.cc:76] 0.000109128
I1211 08:29:43.667255 18792 resnet50_test.cc:76] 0.000138924
...
```

## C++ 预测程序开发说明

使用 Paddle Inference 开发 C++ 预测程序仅需以下五个步骤：


(1) 引用头文件

```c++
#include "paddle/include/paddle_inference_api.h"
```

(2) 创建配置对象，并根据需求配置，详细可参考 [C++ API 文档 - Config](../api_reference/cxx_api_doc/Config_index)

```c++
// 创建默认配置对象
paddle_infer::Config config;

// 设置预测模型路径，即为本小节第2步中下载的模型
config.SetModel(FLAGS_model_file, FLAGS_params_file);

// 启用 GPU 和 MKLDNN 预测
config.EnableUseGpu(100, 0);
config.EnableMKLDNN();

// 开启 内存/显存 复用
config.EnableMemoryOptim();
```

(3) 根据Config创建预测对象，详细可参考 [C++ API 文档 - Predictor](../api_reference/cxx_api_doc/Predictor)

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

(5) 执行预测，详细可参考 [C++ API 文档 - Predictor](../api_reference/cxx_api_doc/Predictor)

```c++
// 执行预测
predictor->Run();
```

(6) 获得预测结果，详细可参考 [C++ API 文档 - Tensor](../api_reference/cxx_api_doc/Tensor)

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
