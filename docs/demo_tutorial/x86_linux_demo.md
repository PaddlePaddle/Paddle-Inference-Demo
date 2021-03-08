# X86 Linux上预测部署示例

## 1 C++预测部署示例

C++示例代码在[链接](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/x86_linux_demo)，下面从`流程解析`和`编译运行示例`两方面介绍。

### 1.1 流程解析

#### 1.1.1 准备预测库

请参考[下载安装预测库文档](../user_guides/download_lib)下载Paddle Inference C++预测库，或者参考[源码编译文档](../user_guides/source_compile)编译Paddle Inference C++预测库。

#### 1.1.2 准备预测模型

使用Paddle训练结束后，得到预测模型，可以用于预测部署。

本示例准备了mobilenet_v1预测模型，可以从[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/mobilenetv1.tgz)下载，或者wget下载。

```shell
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/mobilenetv1.tgz
```

#### 1.1.3 包含头文件

使用Paddle预测库，只需要包含 `paddle_inference_api.h` 头文件。

```cpp
#include "paddle/include/paddle_inference_api.h"
```

#### 1.1.4 设置Config

根据预测部署的实际情况，设置Config，用于后续创建Predictor。

Config默认是使用CPU预测，可以设置开启MKLDNN加速、设置CPU的线程数、开启IR优化、开启内存优化。

```cpp
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

#### 1.1.5 创建Predictor

```cpp
std::shared_ptr<paddle_infer::Predictor> predictor = paddle_infer::CreatePredictor(config);
```

#### 1.1.6 设置输入

从Predictor中获取输入的names和handle，然后设置输入数据。

```cpp
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputHandle(input_names[0]);
std::vector<int> input_shape = {1, 3, 224, 224};
std::vector<float> input_data(1 * 3 * 224 * 224, 1);
input_t->Reshape(input_shape);
input_t->CopyFromCpu(input_data.data());
```

#### 1.1.7 执行Predictor

```cpp
predictor->Run();
```

#### 1.1.8 获取输出

```cpp
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputHandle(output_names[0]);
std::vector<int> output_shape = output_t->shape();
int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                              std::multiplies<int>());
std::vector<float> out_data;
out_data.resize(out_num);
output_t->CopyToCpu(out_data.data());
```

### 1.2 编译运行示例

#### 1.2.1 编译示例

文件`model_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
文件`CMakeLists.txt` 为编译构建文件。   
脚本`run_impl.sh` 包含了第三方库、预编译库的信息配置。

根据前面步骤下载Paddle预测库和mobilenetv1模型。

打开 `run_impl.sh` 文件，设置 LIB_DIR 为下载的预测库路径，比如 `LIB_DIR=/work/Paddle/build/paddle_inference_install_dir`。

运行 `sh run_impl.sh`， 会在当前目录下编译产生build目录。

注意：Paddle Inference 提供下载的C++预测库对应GCC 4.8，所以请检查您电脑中GCC版本是否一致，如果不一致可能出现未知错误。

#### 1.2.2 运行示例

进入build目录，运行样例。

```shell
cd build
./model_test --model_file inference.pdmodel --params_file inference.pdiparams
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

## 2 Python预测部署示例

Python预测部署示例代码在[链接](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/x86_linux_demo)，下面从`流程解析`和`编译运行示例`两方面介绍。

### 2.1 流程解析

#### 2.1.1 准备环境

请参考[飞桨官网](https://www.paddlepaddle.org.cn/)安装2.0及以上版本的Paddle。

Python安装opencv：`pip install opencv-python`。

#### 2.1.2 准备预测模型

使用Paddle训练结束后，得到预测模型，可以用于预测部署。

本示例准备了mobilenet_v1预测模型，可以从[链接](https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz)下载，或者wget下载。

```shell
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz
tar zxf mobilenetv1_fp32.tar.gz
```

#### 2.1.3 Python导入

```
from paddle.inference import Config
from paddle.inference import create_predictor
```

#### 2.1.4 设置Config

根据预测部署的实际情况，设置Config，用于后续创建Predictor。

Config默认是使用CPU预测，可以设置开启MKLDNN加速、设置CPU的线程数、开启IR优化、开启内存优化。

```python
# args 是解析的输入参数
if args.model_dir == "":
    config = Config(args.model_file, args.params_file)
else:
    config = Config(args.model_dir)
config.enable_mkldnn()
config.set_cpu_math_library_num_threads(args.threads)
config.switch_ir_optim()
config.enable_memory_optim()
```

#### 2.1.5 创建Predictor

```python
predictor = create_predictor(config)
```

#### 2.1.6 设置输入

从Predictor中获取输入的names和handle，然后设置输入数据。

```python
img = cv2.imread(args.img_path)
img = preprocess(img)
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])
input_tensor.reshape(img.shape)
input_tensor.copy_from_cpu(img.copy())
```

#### 2.1.7 执行Predictor

```python
predictor.run();
```

#### 2.1.8 获取输出

```python
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])
output_data = output_tensor.copy_to_cpu()
```

### 2.2 编译运行示例

文件`img_preprocess.py`是对图像进行预处理。
文件`model_test.py`是示例程序。

参考前面步骤准备环境、下载预测模型。

下载预测图片。

```shell
wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
```

执行预测命令。

```
python model_test.py --model_dir mobilenetv1_fp32 --img_path ILSVRC2012_val_00000247.jpeg
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。
