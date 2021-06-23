# X86 Linux 上部署 GRU INT8 C++预测示例

## 1 流程解析


1.1 准备预测库

请参考[推理库下载文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html)下载Paddle Linux预测库。

1.2 准备预测模型

使用Paddle训练结束后，得到预测模型。通过quant-aware 或者 post-quantization 得到quant gru模型。

本示例准备了 gru_quant 模型，可以从[链接](https://paddle-inference-dist.cdn.bcebos.com/int8/QAT_models/GRU_quant_acc.tar.gz)下载，或者wget下载。然后使用 [save_quant_model.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py) 将 quant 模型转化为可以在 CPU 上获得加速的INT8模型

```
wget https://paddle-inference-dist.cdn.bcebos.com/int8/QAT_models/GRU_quant_acc.tar.gz
tar -xzvf GRU_quant_acc.tar.gz
gru_quant_model=/full/path/to/GRU_quant_acc
int8_gru_save_path=/full/path/to/new/folder
python3.6 path/to/Paddle/python/paddle/fluid/contrib/slim/tests/save_quant_model.py --quant_model_path ${gru_quant_model} --int8_model_save_path ${int8_gru_save_path} multi_gru
```
转化后的INT8 模型保存在了 `${int8_gru_save_path}`所在位置

1.3 准备数据

从链接下载binary gru模型预测数据[GRU data](https://paddle-inference-dist.cdn.bcebos.com/gru/GRU_eval_data.tar.gz)。

1.4 包含头文件

使用Paddle预测库，只需要包含 `paddle_inference_api.h` 头文件。

```
#include "paddle/include/paddle_inference_api.h"
```

1.4 设置Config

根据预测部署的实际情况，设置Config，用于后续创建Predictor。

Config默认是使用CPU预测，设置开启MKLDNN加速、设置CPU的线程数、开启IR优化、开启内存优化。

```
paddle_infer::Config config;
if (FLAGS_model_dir == "") {
config.SetModel(FLAGS_model_file, FLAGS_params_file); // Load combined model
} else {
config.SetModel(FLAGS_model_dir); // Load no-combined model
}
config.EnableMKLDNN();
config.SetCpuMathLibraryNumThreads(FLAGS_threads);
config.SwitchIrOptim();
config.EnableMemoryOptim();
```

1.5 创建Predictor

```
std::shared_ptr<paddle_infer::Predictor> predictor = paddle_infer::CreatePredictor(config);
```

## 2 编译运行示例


2.1 编译示例

文件`model_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
文件`CMakeLists.txt` 为编译构建文件。
脚本`run.sh` 包含了第三方库、预编译库的信息配置。

打开 `run.sh` 文件，设置 LIB_DIR 为准备的预测库路径，比如 `LIB_DIR=/work/Paddle/build/paddle_inference_install_dir`。

运行 `sh run.sh`,程序开始编译运行。

2.2 性能与精度

* Accuracy
  
|  Model  | FP32    | INT8   | Accuracy diff|
|---------|---------|--------|--------------|
|accuracy | 0.89326 |0.89323 |  -0.00007    |

* Performance of 6271

| Performance configuration  | Naive fp32        | int8 | Int8/Native fp32 |
|----------------------------|-------------------|------|------------------|
| bs 1, thread 1             | 1108              | 1393 | 1.26             |
| repeat 1, bs 50, thread 1  | 2175              | 3199 | 1.47             |
| repeat 10, bs 50, thread 1 | 2165              | 3334 | 1.54             |