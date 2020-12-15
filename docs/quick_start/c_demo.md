# 预测示例 (C)

本章节包含2部分内容：(1) [运行 C 示例程序](#id1)；(2) [C 预测程序开发说明](#id7)。

## 运行 C 示例程序

### 1. 源码编译 C 预测库

Paddle Inference 的 C 预测库需要以源码编译的方式进行获取，请参照以下两个文档进行源码编译

- [安装与编译 Linux 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html) 
- [安装与编译 Windows 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/windows_cpp_inference.html)

编译完成后，在编译目录下的 `paddle_inference_c_install_dir` 即为 C 预测库，目录结构如下：

```bash
paddle_inference_c_install_dir
├── paddle
│   ├── include
│   │   └── paddle_c_api.h               C 预测库头文件
│   └── lib
│       ├── libpaddle_fluid_c.a          C 静态预测库文件
│       └── libpaddle_fluid_c.so         C 动态预测库文件
├── third_party
│   └── install                          第三方链接库和头文件
│       ├── cryptopp
│       ├── gflags
│       ├── glog
│       ├── mkldnn
│       ├── mklml
│       ├── protobuf
│       └── xxhash
└── version.txt                          版本信息与编译选项信息
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

下载 [resnet50](http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz) 模型后解压，得到 Paddle Combined 形式的模型，位于文件夹 model 下。如需查看模型结构，可将 `model` 文件重命名为 `__model__`，然后通过模型可视化工具 Netron 打开。

```bash
wget http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz
tar zxf resnet50_model.tar.gz

# 获得模型目录即文件如下
model/
├── model
└── params
```

### 3. 准备预测部署程序

将以下代码保存为 `c_demo.c` 文件：

```c
#include <stdbool.h>
#include "paddle_c_api.h"
#include <memory.h>
#include <malloc.h>

int main() {
  // 配置 PD_AnalysisConfig
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();

  // 设置预测模型路径，即为本小节第2步中下载的模型
  const char* model_path  = "./model/model";
  const char* params_path = "./model/params";
  PD_SetModel(config, model_path, params_path);

  // 创建输入 Tensor
  PD_Tensor* input_tensor = PD_NewPaddleTensor();

  // 创建输入 Buffer
  PD_PaddleBuf* input_buffer = PD_NewPaddleBuf();
  printf("PaddleBuf empty: %s\n", PD_PaddleBufEmpty(input_buffer) ? "true" : "false");
  int batch   = 1;
  int channel = 3;
  int height  = 318;
  int width   = 318;
  int input_shape[4] = {batch, channel, height, width};
  int input_size     = batch * channel * height * width;
  float* input_data  = malloc(sizeof(float) * input_size);
  int i = 0;
  for (i = 0; i < input_size ; i++){ 
      input_data[i] = 1.0f; 
  }
  PD_PaddleBufReset(input_buffer, (void*)(input_data), sizeof(float) * input_size);

  // 设置输入 Tensor 信息
  char* input_name = "data"; // 可通过 Netron 工具查看输入 Tensor 名字、形状、数据等
  PD_SetPaddleTensorName(input_tensor, input_name);
  PD_SetPaddleTensorDType(input_tensor, PD_FLOAT32);
  PD_SetPaddleTensorShape(input_tensor, input_shape, 4);
  PD_SetPaddleTensorData(input_tensor, input_buffer);

  // 设置输出 Tensor 和 数量
  PD_Tensor* output_tensor = PD_NewPaddleTensor();
  int output_size;

  // 执行预测
  PD_PredictorRun(config, input_tensor, 1, &output_tensor, &output_size, 1);

  // 获取预测输出 Tensor 信息
  printf("Output Tensor Size: %d\n", output_size);
  printf("Output Tensor Name: %s\n", PD_GetPaddleTensorName(output_tensor));
  printf("Output Tensor Dtype: %d\n", PD_GetPaddleTensorDType(output_tensor));

  // 获取预测输出 Tensor 数据
  PD_PaddleBuf* output_buffer = PD_GetPaddleTensorData(output_tensor);
  float* result = (float*)(PD_PaddleBufData(output_buffer));
  int result_length = PD_PaddleBufLength(output_buffer) / sizeof(float);
  printf("Output Data Length: %d\n", result_length);
  
  // 删除输入 Tensor 和 Buffer
  PD_DeletePaddleTensor(input_tensor);
  PD_DeletePaddleBuf(input_buffer);

  return 0;
}
```

### 4. 编译预测部署程序

将 `paddle_inference_c_install_dir/paddle` 目录下的头文件 `paddle_c_api.h` 和动态库文件 `libpaddle_fluid_c.so` 拷贝到与预测源码同一目录，然后使用 GCC 进行编译：

```bash
# GCC 编译命令
gcc c_demo.c libpaddle_fluid_c.so -o c_demo_prog

# 编译完成之后生成 c_demo_prog 可执行文件，编译目录内容如下
c_demo_dir/
│
├── c_demo.c                 预测 C 源码程序，内容如本小节第3步所示
├── c_demo_prog              编译后的预测可执行程序
│
├── paddle_c_api.h           C 预测库头文件
├── libpaddle_fluid_c.so     C 动态预测库文件
│
├── resnet50_model.tar.gz    本小节第2步中下载的预测模型
└── model                    本小节第2步中下载的预测模型解压后的模型文件
    ├── model
    └── params
```

### 5. 执行预测程序

**注意**：需要现将动态库文件 `libpaddle_fluid_c.so` 所在路径加入 `LD_LIBRARY_PATH`，否则会出现无法找到库文件的错误。

```bash
# 执行预测程序
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
./c_demo_prog
```

成功执行之后，得到的预测输出结果如下：

```bash
# 程序输出结果如下
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1211 05:57:48.939208 16443 pd_config.cc:43] ./model/model
I1211 05:57:48.939507 16443 pd_config.cc:48] ./model/model
PaddleBuf empty: true
W1211 05:57:48.941076 16443 analysis_predictor.cc:1052] Deprecated. Please use CreatePredictor instead.
I1211 05:57:48.941124 16443 analysis_predictor.cc:139] Profiler is deactivated, and no profiling report will be generated.
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [attention_lstm_fuse_pass]
--- Running IR pass [seqconv_eltadd_relu_fuse_pass]
--- Running IR pass [seqpool_cvm_concat_fuse_pass]
--- Running IR pass [mul_lstm_fuse_pass]
--- Running IR pass [fc_gru_fuse_pass]
--- Running IR pass [mul_gru_fuse_pass]
--- Running IR pass [seq_concat_fc_fuse_pass]
--- Running IR pass [fc_fuse_pass]
I1211 05:57:49.481595 16443 graph_pattern_detector.cc:101] ---  detected 1 subgraphs
--- Running IR pass [repeated_fc_relu_fuse_pass]
--- Running IR pass [squared_mat_sub_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
I1211 05:57:49.698067 16443 graph_pattern_detector.cc:101] ---  detected 53 subgraphs
--- Running IR pass [conv_transpose_bn_fuse_pass]
--- Running IR pass [conv_transpose_eltwiseadd_bn_fuse_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I1211 05:57:49.741832 16443 analysis_predictor.cc:541] ======= optimize end =======
Output Tensor Size: 1
Output Tensor Name: AddmmBackward190.fc.output.1.tmp_1
Output Tensor Dtype: 0
Output Data Length: 512
```

## C 预测程序开发说明

使用 Paddle Inference 开发 C 预测程序仅需以下六个步骤：


(1) 引用头文件

```c
#include "paddle_c_api.h"
```

(2) 创建配置对象，并指定预测模型路径，详细可参考 [C API 文档 - AnalysisConfig](../api_reference/c_api_doc/Config_index)

```c
// 配置 PD_AnalysisConfig
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置预测模型路径，即为本小节第2步中下载的模型
const char* model_path = "./model/model";
const char* params_path = "./model/params";
PD_SetModel(config, model_path, params_path);
```

(3) 设置模型输入和输出 Tensor，详细可参考 [C API 文档 - PaddleTensor](../api_reference/c_api_doc/PaddleTensor)

```c
// 创建输入 Tensor
PD_Tensor* input_tensor = PD_NewPaddleTensor();

// 创建输入 Buffer
PD_PaddleBuf* input_buffer = PD_NewPaddleBuf();
printf("PaddleBuf empty: %s\n", PD_PaddleBufEmpty(input_buffer) ? "true" : "false");
int batch = 1;
int channel = 3;
int height = 318;
int width = 318;
int input_shape[4] = {batch, channel, height, width};
int input_size = batch * channel * height * width;
float* data = malloc(sizeof(float) * input_size);
int i = 0;
for (i = 0; i < input_size ; i++){ 
    data[i] = 1.0f; 
}
PD_PaddleBufReset(input_buffer, (void*)(data), sizeof(float) * input_size);

// 设置输入 Tensor 信息
char* input_name = "data"; // 可通过 Netron 工具查看输入 Tensor 名字、形状、数据等
PD_SetPaddleTensorName(input_tensor, input_name);
PD_SetPaddleTensorDType(input_tensor, PD_FLOAT32);
PD_SetPaddleTensorShape(input_tensor, input_shape, 4);
PD_SetPaddleTensorData(input_tensor, input_buffer);

// 设置输出 Tensor 和 数量
PD_Tensor* output_tensor = PD_NewPaddleTensor();
int output_size;
```

(4) 执行预测引擎，，详细可参考 [C API 文档 - Predictor](../api_reference/c_api_doc/Predictor)

```c
// 执行预测
PD_PredictorRun(config, input_tensor, 1, &output_tensor, &output_size, 1);
```

(5) 获得预测结果，详细可参考 [C API 文档 - PaddleTensor](../api_reference/c_api_doc/PaddleTensor)

```c
// 获取预测输出 Tensor 信息
printf("Output Tensor Size: %d\n", output_size);
printf("Output Tensor Name: %s\n", PD_GetPaddleTensorName(output_tensor));
printf("Output Tensor Dtype: %d\n", PD_GetPaddleTensorDType(output_tensor));

// 获取预测输出 Tensor 数据
PD_PaddleBuf* output_buffer = PD_GetPaddleTensorData(output_tensor);
float* result = (float*)(PD_PaddleBufData(output_buffer));
int result_length = PD_PaddleBufLength(output_buffer) / sizeof(float);
printf("Output Data Length: %d\n", result_length);
```

(6) 删除输入 Tensor，Buffer 和 Config

```c
PD_DeletePaddleTensor(input_tensor);
PD_DeletePaddleBuf(input_buffer);
PD_DeleteAnalysisConfig(config);
```