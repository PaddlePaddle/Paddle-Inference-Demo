# 预测示例 (C)

本章节包含2部分内容：(1) [运行 C 示例程序](#id1)；(2) [C 预测程序开发说明](#id7)。

## 运行 C 示例程序

### 1. 源码编译 C 预测库

Paddle Inference 的 C 预测库需要以源码编译的方式进行获取，请参照以下两个文档进行源码编译

- [安装与编译 Linux 预测库](../user_guides/source_compile.html#ubuntu-18-04)
- [安装与编译 Windows 预测库](../user_guides/source_compile.html#windows-10)

编译完成后，在编译目录下的 `paddle_inference_c_install_dir` 即为 C 预测库，目录结构如下：

```bash
paddle_inference_c_install_dir
├── paddle
│   ├── include               C 预测库头文件目录
│   │   └── pd_common.h
│   │   └── pd_config.h
│   │   └── pd_inference_api.h         C 预测库头文件
│   │   └── pd_predictor.h
│   │   └── pd_tensor.h
│   │   └── pd_types.h
│   │   └── pd_utils.h
│   └── lib
│       ├── libpaddle_inference_c.a          C 静态预测库文件
│       └── libpaddle_inference_c.so         C 动态预测库文件
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

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 预测格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可将 `inference.pdmodel` 文件重命名为 `__model__`，然后通过模型可视化工具 Netron 打开。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50_model.tar.gz

# 获得模型目录即文件如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

### 3. 准备预测部署程序

将以下代码保存为 `c_demo.c` 文件：

```c
#include "pd_inference_api.h"
#include <memory.h>
#include <malloc.h>

int main() {
  // 创建 Config 对象
  PD_Config* config = PD_ConfigCreate();

  // 设置预测模型路径，即为本小节第2步中下载的模型
  const char* model_path  = "./resnet50/inference.pdmodel";
  const char* params_path = "./resnet50/inference.pdiparams";
  PD_ConfigSetModel(config, model_path, params_path);
  
  // 根据 Config 创建 Predictor, 并销毁 Config 对象
  PD_Predictor* predictor = PD_PredictorCreate(config);

  // 准备输入数据
  int32_t input_shape[4] = {1, 3, 244, 244};
  float* input_data = (float*)calloc(1 * 3 * 224 * 224, sizeof(float));

  // 获取输入 Tensor
  PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
  PD_Tensor* input_tensor = PD_PredictorGetInputHandle(predictor, input_names->data[0]);

  // 设置输入 Tensor 的维度信息及数据
  PD_TensorReshape(input_tensor, 4, input_shape);
  PD_TensorCopyFromCpuFloat(input_tensor, input_data);

  // 执行预测
  PD_PredictorRun(pd_predictor);

  // 获取预测输出 Tensor
  PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
  PD_Tensor* output_tensor = PD_PredictorGetOutputHandle(predictor, output_names->data[0]);

  // 获取预测输出 Tensor 信息
  PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
  int32_t out_size = 1;
  for (size_t i = 0; i < output_shape->size; ++i) {
    out_size = out_size * output_shape->data[i];
  }

  // 打印输出 Tensor 信息
  printf("Output Tensor Name: %s\n", output_names->data[0]);
  printf("Output Tensor Size: %d\n", out_size);

  // 获取预测输出 Tensor 数据
  out_data = (float*)malloc(out_size * sizeof(float));
  PD_TensorCopyToCpuFloat(output_tensor, out_data);


  // 销毁相关对象， 回收相关内存
  free(out_data);
  PD_OneDimArrayInt32Destroy(output_shape);
  PD_TensorDestroy(output_tensor);
  PD_OneDimArrayCstrDestroy(output_names);
  PD_TensorDestroy(input_tensor);
  PD_OneDimArrayCstrDestroy(input_names);
  free(input_data);
  PD_PredictorDestroy(predictor);

  return 0;
}
```

### 4. 编译预测部署程序

将 `paddle_inference_c_install_dir/paddle/include` 目录下的所有头文件和动态库文件 `paddle_inference_c_install_dir/paddle/lib/libpaddle_inference_c.so` 拷贝到与预测源码同一目录，然后使用 GCC 进行编译：

```bash
# GCC 编译命令
gcc c_demo.c libpaddle_inference_c.so -o c_demo_prog

# 编译完成之后生成 c_demo_prog 可执行文件，编译目录内容如下
c_demo_dir/
│
├── c_demo.c                 预测 C 源码程序，内容如本小节第3步所示
├── c_demo_prog              编译后的预测可执行程序
│
├── pd_inference_api.h         C 预测库头文件
├── pd_common.h
├── pd_config.h
├── pd_utils.h
├── pd_predictor.h
├── pd_tensor.h
├── pd_types.h
├── libpaddle_fluid_c.so     C 动态预测库文件
│
├── resnet50_model.tar.gz    本小节第2步中下载的预测模型
└── resnet50                 本小节第2步中下载的预测模型解压后的模型文件
    ├── inference.pdmodel
    ├── inference.pdiparams.info
    └── inference.pdiparams
```

### 5. 执行预测程序

**注意**：需要先将动态库文件 `libpaddle_inference_c.so` 所在路径加入 `LD_LIBRARY_PATH`，否则会出现无法找到库文件的错误。

```bash
# 执行预测程序
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
./c_demo_prog
```

成功执行之后，得到的预测输出结果如下：

```bash
# 程序输出结果如下
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1211 05:57:48.939208 16443 pd_config.cc:43] ./resnet50/inference.pdmodel
I1211 05:57:48.939507 16443 pd_config.cc:48] ./resnet50/inference.pdiparams
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
Output Tensor Name: AddmmBackward190.fc.output.1.tmp_1
Output Tensor Size: 1
```

## C 预测程序开发说明

使用 Paddle Inference 开发 C 预测程序仅需以下七个步骤：


(1) 引用头文件

```c
#include "pd_inference_api.h"
```

(2) 创建配置对象，并指定预测模型路径，详细可参考 [C API 文档 - Config 方法](../api_reference/c_api_doc/Config_index)

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径，即为本小节第2步中下载的模型
const char* model_path  = "./resnet50/inference.pdmodel";
const char* params_path = "./resnet50/inference.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);
```
(3) 根据Config创建预测对象，详细可参考 [C API 文档 - Predictor 方法](../api_reference/c_api_doc/Predictor)

```c
// 根据 Config 创建 Predictor, 并销毁 Config 对象
PD_Predictor* predictor = PD_PredictorCreate(config);
```
(4) 设置模型输入Tensor，详细可参考 [C API 文档 - Tensor 方法](../api_reference/c_api_doc/Tensor)

```c
// 准备输入数据
int32_t input_shape[4] = {1, 3, 244, 244};
float* input_data = (float*)calloc(1 * 3 * 224 * 224, sizeof(float));

// 获取输入 Tensor
PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
PD_Tensor* input_tensor = PD_PredictorGetInputHandle(predictor, input_names->data[0]);

// 设置输入 Tensor 的维度信息及数据
PD_TensorReshape(input_tensor, 4, input_shape);
PD_TensorCopyFromCpuFloat(input_tensor, input_data);
```

(5) 执行预测引擎，详细可参考 [C API 文档 - Predictor 方法](../api_reference/c_api_doc/Predictor)

```c
// 执行预测
PD_PredictorRun(pd_predictor);
```
(6) 获得预测结果，详细可参考 [C API 文档 - Tensor 方法](../api_reference/c_api_doc/Tensor)

```c
// 获取预测输出 Tensor
PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
PD_Tensor* output_tensor = PD_PredictorGetOutputHandle(predictor, output_names->data[0]);

// 获取预测输出 Tensor 信息
PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
int32_t out_size = 1;
for (size_t i = 0; i < output_shape->size; ++i) {
  out_size = out_size * output_shape->data[i];
}

// 打印输出 Tensor 信息
printf("Output Tensor Name: %s\n", output_names->data[0]);
printf("Output Tensor Size: %d\n", out_size);

// 获取预测输出 Tensor 数据
out_data = (float*)malloc(out_size * sizeof(float));
PD_TensorCopyToCpuFloat(output_tensor, out_data);
```

(7) 销毁相关对象，回收相关内存

```c
// 销毁相关对象， 回收相关内存
free(out_data);
PD_OneDimArrayInt32Destroy(output_shape);
PD_TensorDestroy(output_tensor);
PD_OneDimArrayCstrDestroy(output_names);
PD_TensorDestroy(input_tensor);
PD_OneDimArrayCstrDestroy(input_names);
free(input_data);
PD_PredictorDestroy(predictor);
```
