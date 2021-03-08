# GPU上C++预测部署示例

## 1 流程解析

1.1 准备预测库

请参考[推理库下载文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html)下载Paddle预测库。

1.2 准备预测模型

使用Paddle训练结束后，得到预测模型，可以用于预测部署。

本示例准备了mobilenet_v1预测模型，可以从[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/mobilenetv1.tgz)下载，或者wget下载。

```
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/mobilenetv1.tgz
tar xzf mobilenetv1.tgz
```

1.3 包含头文件

使用Paddle预测库，只需要包含 `paddle_inference_api.h` 头文件。

```
#include "paddle/include/paddle_inference_api.h"
```

1.4 设置Config

根据预测部署的实际情况，设置Config，用于后续创建Predictor。

Config默认是使用CPU预测，若要使用GPU预测，需要手动开启，设置运行的GPU卡号和分配的初始显存。可以设置开启TensorRT加速、开启IR优化、开启内存优化。使用Paddle-TensorRT相关说明和示例可以参考[文档](https://paddle-inference.readthedocs.io/en/master/optimize/paddle_trt.html)。

```
paddle_infer::Config config;
if (FLAGS_model_dir == "") {
config.SetModel(FLAGS_model_file, FLAGS_params_file); // Load combined model
} else {
config.SetModel(FLAGS_model_dir); // Load no-combined model
}
config.EnableUseGpu(500, 0);
config.SwitchIrOptim(true);
config.EnableMemoryOptim();
config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, 10, PrecisionType::kFloat32, false, false);
```

1.5 创建Predictor

```
std::shared_ptr<paddle_infer::Predictor> predictor = paddle_infer::CreatePredictor(config);
```

1.6 设置输入

从Predictor中获取输入的names和handle，然后设置输入数据。

```
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputHandle(input_names[0]);
std::vector<int> input_shape = {1, 3, 224, 224};
std::vector<float> input_data(1 * 3 * 224 * 224, 1);
input_t->Reshape(input_shape);
input_t->CopyFromCpu(input_data.data());
```

1.7 执行Predictor

```
predictor->Run();
```

1.8 获取输出

```
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputHandle(output_names[0]);
std::vector<int> output_shape = output_t->shape();
int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                            std::multiplies<int>());
std::vector<float> out_data;
out_data.resize(out_num);
output_t->CopyToCpu(out_data.data());
```

## 2 编译运行示例

2.1 编译示例

文件`model_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
脚本`run.sh`为一键运行脚本。

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
TENSORRT_ROOT=/usr/local/TensorRT-7.0.0.11
```

运行 `bash compile.sh`， 会在目录下产生build目录。

2.2 运行示例

```shell
bash run.sh
# 或
./build/model_test --model_file mobilenetv1/inference.pdmodel --params_file mobilenetv1/inference.pdiparams
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。
