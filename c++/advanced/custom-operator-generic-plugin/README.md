# 自定义算子通用Plugin运行样例

## 一：背景介绍
自定义算子的完整用法样例参照[custom-operator](../custom-operator/README.md)。 [custom-operator](../custom-operator/README.md)中提供了一种自定义算子的实现方式，需要用户自行实现相应的plugin，优点是较为灵活，缺点是有较高的开发成本。本样例提供另一种更为简便和高效的实现方法。

## 二：运行样例

### 1. 获取样例中的自定义算子模型

下载地址：https://paddle-inference-dist.bj.bcebos.com/inference_demo/custom_operator/custom_gap_infer_model.tgz

执行 `tar zxvf custom_gap_infer_model.tgz` 将模型文件解压至当前目录。

### 2. 编译样例

文件 `custom_gap_op.cc`、`custom_gap_op.cu` 为自定义算子源文件，自定义算子编写方式请参考[飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)。
注意：自定义算子目前需要与飞桨预测库 `libpaddle_inference.so` 联合构建。

文件`custom_gap_op_test.cc` 为预测的样例程序。
文件`CMakeLists.txt` 为编译构建文件。
脚本`compile.sh` 包含了第三方库、预编译库的信息配置。

我们首先需要对脚本`compile.sh` 文件中的配置进行修改。
打开`compile.sh`，我们对以下的几处信息进行修改：

```shell
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6
```

运行 `bash compile.sh`， 会在目录下产生build目录。

### 3. 运行样例

```shell
# 运行样例
./build/custom_gap_op_test
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

> 注：确保路径配置正确后，也可执行执行 `bash run.sh` ，一次性完成以上两个步骤的执行

## 三：自定义算子通用Plugin使用说明

一般情况下，自定义算子的注册声明方式如下(推理仅需注册前向)：

\* 详细用法参考 [飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn) 

```c++
PD_BUILD_OP(op_type)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(cuda_forward))
    .Attrs({"attr1: std::vector<int>", "attr2: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype));
```

当模型运行在 TensorRT 后端时，为了充分发挥 TensorRT 的性能，期望模型全图跑在 TensorRT 上，Paddle Inference 推理并未默认支持将自定义算子映射 TensorRT 的 Layer 中。自定义算子映射至 TensorRT 的 Layer 除了依照 TensorRT 官方文档自行实现 Plugin 之外，更便捷的做法是只额外指定 getOutputDimensions 和通过简单配置实现 supportsFormatCombination，其他部分框架可以自动完成。

getOutputDimensions的实现方法和 TensorRT 官方文档类似，区别在于以下两点：

1. `int32_t outputIndex` 和 `int32_t nbInputs` 两个参数组合成 `std::pair<int_32_t, int_32_t> outputIndex_nbInputs`，作为 getOutputDimensions 的输入参数，函数体内使用时可以用 *.first 和 *.second 分别获取两个参数的值。
2. 支持Attr，Attr的类型和使用方法参考 [飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn) 。

比如本样例中，自定义算子 custom_gap 的 getOutputDimensions 实现如下：
```c++
nvinfer1::DimsExprs getOutputDimensions(
    std::pair<int32_t, int32_t> outputIndex_nbInputs,
    const nvinfer1::DimsExprs* inputs,
    nvinfer1::IExprBuilder& exprBuilder,
    const std::vector<int>& test_attr1,
    const int test_attr2) noexcept {
  nvinfer1::DimsExprs dimsOutput(inputs[0]);
  dimsOutput.d[dimsOutput.nbDims - 1] = exprBuilder.constant(test_attr2);
  dimsOutput.d[dimsOutput.nbDims - 2] = exprBuilder.constant(test_attr2);
  return dimsOutput;
}
```

supportsFormatCombination 通简单配置实现，配置格式：`{"<dtype1>[:format1]+<dtype2>[:format2]+...", "<dtype3>[:format3]+<dtype4>[:format4]+..."}`
依次按照输入输出数据类型和格式进行配置，支持多个输入输出数据类型和格式的配置。比如一个有2个输入1个输出，第1个输入支持fp32和fp16，第二个输入只支持int32的自定义算子，config可以写为：
```c++
{"float32:LINEAR+int32:LINEAR+float32:LINEAR", "float16:LINEAR+int32:LINEAR+float16:LINEAR"}
```
提供format时用 `:` 隔开，format可缺省，缺省时为LINEAR。注意，要严格输入、输出的顺序指定支持的格式。
当前支持的数据类型和format及其对应关系如下表。

| 数据类型 | format |
| :---: | :---: |
| float32 | LINEAR, CHW32 | 
| float16 | LINEAR, CHW2, HWC8, CHW4, <br> DHWC8(Tensor 7.2及以上), HWC16(Tensor 8.0及以上)| 
| int32 | LINEAR, CHW32, CHW4 | 
| int8 | LINEAR| 

本样例中，自定义算子 custom_gap 支持推理映射为 Tensor Layer 的算子注册方法如下：
```c++
PD_BUILD_OP(gap)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(paddle_gap_forward))
    .Attrs({"test_attr1: std::vector<int>", "test_attr2: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype))
    .SetTrtInferShapeFn(PD_TRT_INFER_SHAPE(getOutputDimensions))
    .SetTrtSupportsFormatConfig({"float32:LINEAR+float32:LINEAR"});
```
相比自定义算子的注册，仅增加了 `SetTrtInferShapeFn` 和 `SetTrtSupportsFormatConfig` 两个接口。

Demo 需要添加 `-DPADDLE_WITH_TENSORRT` 编译选项，同时将自定义算子编译的动态库文件链接到可执行文件上。`CMakeLists.txt` 中需要有如下代码：
```bash
if(USE_TENSORRT)
  add_definitions("-DPADDLE_WITH_TENSORRT")
endif()

cuda_add_library(custom_op ${CUSTOM_OPERATOR_FILES} SHARED)
set(DEPS ${DEPS} custom_op)
add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)
target_link_libraries(${DEMO_NAME} ${DEPS})
```


## 更多链接
- [Paddle Inference使用Quick Start！](https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/index_quick_start.html)
- [Paddle Inference C++ Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/python_api_index.html)
