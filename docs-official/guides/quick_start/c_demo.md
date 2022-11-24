# 快速上手C推理

本章节包含2部分内容,
- [运行 C 示例程序](#id1)
- [C 推理程序开发说明](#id2)

注意本章节文档和代码仅适用于Linux系统。

## 运行 C 示例程序

在此环节中，共包含以下5个步骤，
- 环境准备
- 模型准备
- 推理代码
- 编译代码
- 执行程序

### 1. 环境准备
Paddle Inference 提供了 Ubuntu/Windows/MacOS 平台的官方 Release 推理库下载, 用户需根据开发环境和硬件自行下载安装，具体可参阅 [C 推理环境安装](../install/c_install.md)。

### 2. 模型准备

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 推理格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可参考[模型结构可视化文档](../export_model/visual_model.html)。

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

将以下代码保存为 `c_demo.c` 文件：

```c
#include "pd_inference_api.h"
#include <memory.h>
#include <malloc.h>

int main() {
  // 创建 Config 对象
  PD_Config* config = PD_ConfigCreate();

  // 设置推理模型路径，即为本小节第2步中下载的模型
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

  // 执行推理
  PD_PredictorRun(predictor);

  // 获取推理输出 Tensor
  PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
  PD_Tensor* output_tensor = PD_PredictorGetOutputHandle(predictor, output_names->data[0]);

  // 获取推理输出 Tensor 信息
  PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
  int32_t out_size = 1;
  for (size_t i = 0; i < output_shape->size; ++i) {
    out_size = out_size * output_shape->data[i];
  }

  // 打印输出 Tensor 信息
  printf("Output Tensor Name: %s\n", output_names->data[0]);
  printf("Output Tensor Size: %d\n", out_size);

  // 获取推理输出 Tensor 数据
  float* out_data = (float*)malloc(out_size * sizeof(float));
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

### 4. 编译代码

将 `paddle_inference_c/paddle/include` 目录下的所有头文件和动态库文件 `paddle_inference_c_install_dir/paddle/lib/libpaddle_inference_c.so` 拷贝到与推理源码同一目录，然后使用 GCC 进行编译：
将如下动态库文件拷贝到与推理源码同一目录，然后使用 gcc 进行编译，
- paddle_inference_c/third_party/install/paddle2onnx/lib/libpaddle2onnx.so
- paddle_inference_c/third_party/install/onnxruntime/lib/libonnxruntime.so.1.10.0
- paddle_inference_c/third_party/install/mklml/lib/libmklml_intel.so
- paddle_inference_c/third_party/install/mklml/lib/libiomp5.so
- paddle_inference_c/third_party/install/mkldnn/lib/libdnnl.so.2
- paddle_inference_c/paddle/lib/libpaddle_inference_c.so

```bash
# 拷贝所有的动态库到编译路径
find paddle_inference_c/ -name "*.so*" | xargs -i cp {} .

# GCC 编译命令
gcc c_demo.c -Ipaddle_inference_c/paddle/include \
    libpaddle2onnx.so libonnxruntime.so.1.10.0 \
    libiomp5.so libdnnl.so.2 libpaddle_inference_c.so \
    -o c_demo_prog
```

编译完成之后在当前目录生成 c_demo_prog 可执行文件

### 5. 执行程序

**注意**：需要先将动态库文件所在路径加入 `LD_LIBRARY_PATH`，否则会出现无法找到库文件的错误。

```bash
# 执行推理程序
export LD_LIBRARY_PATH=${PWD}:${LD_LIBRARY_PATH}
./c_demo_prog
```

成功执行之后，得到的推理输出结果如下：

```bash
# 程序输出结果如下
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [layer_norm_fuse_pass]
---    Fused 0 subgraphs into layer_norm op.
--- Running IR pass [attention_lstm_fuse_pass]
--- Running IR pass [seqconv_eltadd_relu_fuse_pass]
--- Running IR pass [seqpool_cvm_concat_fuse_pass]
--- Running IR pass [mul_lstm_fuse_pass]
--- Running IR pass [fc_gru_fuse_pass]
---    fused 0 pairs of fc gru patterns
--- Running IR pass [mul_gru_fuse_pass]
--- Running IR pass [seq_concat_fc_fuse_pass]
--- Running IR pass [squeeze2_matmul_fuse_pass]
--- Running IR pass [reshape2_matmul_fuse_pass]
WARNING: Logging before InitGoogleLogging() is written to STDERR
W1202 07:16:22.473459  3803 op_compat_sensible_pass.cc:219]  Check the Attr(transpose_Y) of Op(matmul) in pass(reshape2_matmul_fuse_pass) failed!
W1202 07:16:22.473500  3803 map_matmul_to_mul_pass.cc:668] Reshape2MatmulFusePass in op compat failed.
--- Running IR pass [flatten2_matmul_fuse_pass]
--- Running IR pass [map_matmul_v2_to_mul_pass]
--- Running IR pass [map_matmul_v2_to_matmul_pass]
--- Running IR pass [map_matmul_to_mul_pass]
I1202 07:16:22.476769  3803 fuse_pass_base.cc:57] ---  detected 1 subgraphs
--- Running IR pass [fc_fuse_pass]
I1202 07:16:22.478200  3803 fuse_pass_base.cc:57] ---  detected 1 subgraphs
--- Running IR pass [repeated_fc_relu_fuse_pass]
--- Running IR pass [squared_mat_sub_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
I1202 07:16:22.526548  3803 fuse_pass_base.cc:57] ---  detected 53 subgraphs
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [conv_transpose_bn_fuse_pass]
--- Running IR pass [conv_transpose_eltwiseadd_bn_fuse_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I1202 07:16:22.576740  3803 analysis_predictor.cc:717] ======= optimize end =======
I1202 07:16:22.579823  3803 naive_executor.cc:98] ---  skip [feed], feed -> inputs
I1202 07:16:22.581485  3803 naive_executor.cc:98] ---  skip [save_infer_model/scale_0.tmp_1], fetch -> fetch
Output Tensor Name: save_infer_model/scale_0.tmp_1
Output Tensor Size: 1000
```

## C 推理程序开发说明

使用 Paddle Inference 开发 C 推理程序仅需以下七个步骤：


(1) 引用头文件

```c
#include "pd_inference_api.h"
```

(2) 创建配置对象，并指定推理模型路径，详细可参考 [C API 文档 - Config 方法](../../api_reference/c_api_doc/Config_index.rst)

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置推理模型路径，即为本小节第2步中下载的模型
const char* model_path  = "./resnet50/inference.pdmodel";
const char* params_path = "./resnet50/inference.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);
```
(3) 根据Config创建推理对象，详细可参考 [C API 文档 - Predictor 方法](../../api_reference/c_api_doc/Predictor.md)

```c
// 根据 Config 创建 Predictor, 并销毁 Config 对象
PD_Predictor* predictor = PD_PredictorCreate(config);
```
(4) 设置模型输入Tensor，详细可参考 [C API 文档 - Tensor 方法](../../api_reference/c_api_doc/Tensor.md)

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

(5) 执行推理引擎，详细可参考 [C API 文档 - Predictor 方法](../../api_reference/c_api_doc/Predictor.md)

```c
// 执行推理
PD_PredictorRun(predictor);
```
(6) 获得推理结果，详细可参考 [C API 文档 - Tensor 方法](../../api_reference/c_api_doc/Tensor.md)

```c
// 获取推理输出 Tensor
PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
PD_Tensor* output_tensor = PD_PredictorGetOutputHandle(predictor, output_names->data[0]);

// 获取推理输出 Tensor 信息
PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
int32_t out_size = 1;
for (size_t i = 0; i < output_shape->size; ++i) {
  out_size = out_size * output_shape->data[i];
}

// 打印输出 Tensor 信息
printf("Output Tensor Name: %s\n", output_names->data[0]);
printf("Output Tensor Size: %d\n", out_size);

// 获取推理输出 Tensor 数据
float* out_data = (float*)malloc(out_size * sizeof(float));
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

至此 Paddle Inference 推理已跑通，如果想更进一步学习 Paddle Inference，可以根据硬件情况选择学习 GPU 推理、CPU 推理、进阶使用等章节。
