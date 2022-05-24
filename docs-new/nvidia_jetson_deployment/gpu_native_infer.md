# GPU 原生推理

不论你用什么操作系统，使用 GPU 原生推理前必须确保你的机器上已经安装了 CUDA 和 cuDNN，并且你一定得知道它们的安装位置。
下面分别介绍在 Linux/Ubuntu 操作系统下和 Windows 操作系统下用 GPU 原生推理的示例。

## 1 C++ 示例


使用 PaddlePaddle 训练结束后，得到预测模型，可以用于预测部署。

本示例准备了 mobilenet_v1 预测模型，可以从[链接](https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz)下载，或者wget下载。

```shell
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz
```

C++ 示例代码在[链接](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/cuda_linux_demo)，下面从先介绍此示例代码的流程解析，然后介绍如何在 Linux/Ubuntu 系统下和 Windows 系统下编译和执行此示例代码。

&emsp;

下面介绍请参考示例代码中的`model_test.cc`，它包含了使用 Paddle Inference C++ API 的典型过程。

(1) 包含头文件

使用 Paddle Inference 预测库，只需要含 `paddle_inference_api.h`。

```cpp
#include "paddle/include/paddle_inference_api.h"
```

(2) 设置 Config

根据预测部署的实际情况，设置 Config，用于后续创建 Predictor。

Config 默认用 CPU 预测，若要用 GPU 预测，需手动开启，设置分配的初始显存 和 运行的 GPU 卡号。可以设置开启 TensorRT 加速、开启 IR 优化、开启内存优化。使用Paddle-TensorRT 相关说明和示例可以参考[文档](https://paddle-inference.readthedocs.io/en/master/optimize/paddle_trt.html)。

```cpp
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

如果你不想使用 TensorRT 加速，仅想使用 GPU 原生推理，请注释掉`config.EnableTensorRtEngine();` 这行。

(3) 创建 Predictor

```cpp
std::shared_ptr<paddle_infer::Predictor> predictor = paddle_infer::CreatePredictor(config);
```

(4) 设置输入

从 Predictor 中获取输入的 names 和 handle，然后设置输入的 shape 和输入的数据。

```cpp
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputHandle(input_names[0]);
std::vector<int> input_shape = {1, 3, 224, 224};
std::vector<float> input_data(1 * 3 * 224 * 224, 1);
input_t->Reshape(input_shape);
input_t->CopyFromCpu(input_data.data());
```

(5) 执行Predictor

```cpp
predictor->Run();
```

(6) 获取输出

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

### Linux/Ubuntu 部署示例

请参考[下载安装 Ubuntu 预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux)下载 Paddle Inference C++ 预测库，名称中带有 `cuda` 的为用于 GPU 的预测库。以 `manylinux_cuda11.2_cudnn8.2_avx_mkl_trt8_gcc8.2`为例，它要求你的系统上安装 CUDA 版本为11.2，cuDNN 版本为8.2， TensorRT 版本为8.x， gcc 版本为8.2，当然，如果你的上述版本不能完全对应，那么或许也是可以的。注意，如果你的机器上没有安装 TensorRT，你仍然可以下载这个库，只不过模型就只能用 GPU 原生推理，而不能使用 TensorRT 加速。


下面介绍请参考示例代码的在 Linux/Ubuntu 下的编译和执行。
你需要关心下面四个文件即可。
文件`model_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定修改）。    
文件`../lib/CMakeLists.txt` 为编译构建文件，
脚本 `compile.sh` 为编译脚本，它将复制`../lib/CMakeLists.txt`到当前目录，并编译生成可执行文件。
脚本`run.sh` 下载模型，并运行可执行程序。

先要把你下载好的Paddle Inference预测库放到`Paddle-Inference-Demo/c++/lib`中，然后在 `compile.sh` 里面进行如下设置。
如果你不想使用 TensorRT 或机器上没有安装TensorRT，那你要记得把`USE_TENSORRT`置于`OFF`。`LIB_DIR`就是你的Paddle Inference预测库的放置路径，`CUDNN_LIB`、`CUDA_LIB`、`TENSORRT_ROOT`分别为你的CUDNN的库路径，CUDA的库路径，以及TensorRT的根目录。

```shell
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

LIB_DIR=${work_path}/../lib/paddle_inference
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/tensorrt
```

最后只需简单的两个命令即可完成编译和执行。

```shell
bash compile.sh
bash run.sh
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

### Windows上 GPU 原生推理部署示例

请参考[下载安装 Windows 预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows)下载 Paddle Inference C++ 预测库。

Windows上部署的话，你需要下面几个图形界面的操作，此时你只需要关注两个文件即可。

文件`model_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
文件`../lib/CMakeLists.txt` 为编译构建文件，请把它手动复制到和`model_test.cc`相同目录。

打开cmake-gui程序生成vs工程：

- 选择源代码路径，及编译产物路径，如图所示

![win_x86_cpu_cmake_1](./images/win_x86_cpu_cmake_1.png)

- 点击Configure，选择Visual Studio且选择x64版本如图所示，点击Finish，由于我们没有加入必要的CMake Options，会导致configure失败，请继续下一步。

![win_x86_cpu_cmake_2](./images/win_x86_cpu_cmake_2.png)

- 设置CMake Options，点击Add Entry，新增PADDLE_LIB，CMAKE_BUILD_TYPE，DEMO_NAME等选项。具体配置项如下图所示，其中PADDLE_LIB为您下载的预测库路径。

![win_x86_cpu_cmake_3](./images/win_x86_cpu_cmake_3.png)

- 点击Configure，log信息显示Configure done代表配置成功，接下来点击Generate生成vs工程，log信息显示Generate done，代表生成成功，最后点击Open Project打开Visual Studio.

- 设置为Release/x64，编译，编译产物在build/Release目录下。

![win_x86_cpu_vs_1](./images/win_x86_cpu_vs_1.png)

&emsp;

运行示例

首先设置model_test工程为启动首选项。

![win_x86_cpu_vs_2](./images/win_x86_cpu_vs_2.png)

配置输入flags，即设置您之前下载的模型路径。点击Debug选项卡的`model_test Properities..`

![win_x86_cpu_vs_3](./images/win_x86_cpu_vs_3.png)

点击Debug选项卡下的Start Without Debugging选项开始执行程序。

![win_x86_cpu_vs_4](./images/win_x86_cpu_vs_4.png)


## 2 Python 示例

请参考[飞桨官网](https://www.paddlepaddle.org.cn/)安装2.0及以上版本的paddlepaddle-gpu。或者从[下载安装 Ubuntu 预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux)下载 Paddle Inference Python 预测库，名称中带有 `cuda` 的为用于 GPU 的预测库。上面两个地方安装的Python包都可以支持原生GPU推理。

此示例需要你在Python里安装opencv，命令为`python -m pip install opencv-python`。


Python 示例代码在[链接](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/cuda_linux_demo)，下面从先介绍此示例代码的流程解析，然后介绍如何在 Linux/Ubuntu 系统中 和 Windows 系统下执行此示例代码。

&emsp;

下面介绍请参考示例代码中的`model_test.py`，它包含了使用 Paddle Inference Python API 的典型过程。

（1） Python 导入

```
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
```

（2）设置 Config

根据预测部署的实际情况，设置 Config ，用于后续创建 Predictor。

Config 默认用 CPU 预测，若要用 GPU 预测，需手动开启，设置分配的初始显存 和 运行的 GPU 卡号。可以设置开启 TensorRT 加速、开启 IR 优化、开启内存优化。使用Paddle-TensorRT 相关说明和示例可以参考[文档](https://paddle-inference.readthedocs.io/en/master/optimize/paddle_trt.html)。


```python
# args 是解析的输入参数
# Init config
if args.model_dir == "":
    config = Config(args.model_file, args.params_file)
else:
    config = Config(args.model_dir)
config.enable_use_gpu(500, 0)
config.switch_ir_optim()
config.enable_memory_optim()
config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=PrecisionType.Float32,max_batch_size=1, min_subgraph_size=5, use_static=True, use_calib_mode=False)
```

如果你不想使用 TensorRT 加速，仅想使用 GPU 原生推理，请注释掉`config.enable_tensorrt_engine();` 这行。

（3）创建Predictor

```python
# Create predictor
predictor = create_predictor(config)
```

（4） 设置输入

从 Predictor 中获取输入的 names 和 handle，然后设置输入 shape 和 输入数据。

```python
img = cv2.imread(args.img_path)
img = preprocess(img)
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])
input_tensor.reshape(img.shape)
input_tensor.copy_from_cpu(img.copy())
```

（5） 执行 Predictor

```python
predictor.run()
```

（6） 获取输出

```python
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])
output_data = output_tensor.copy_to_cpu()
```

&emsp;

下面介绍请参考示例代码的编译和执行。

文件`img_preprocess.py`是对图像进行预处理。
文件`model_test.py`是示例程序。
脚本 `run.sh`负责下载模型和执行示例程序。

在 Linux/Ubuntu 下你只需要执行`bash run.sh`，就可以看到程序被执行。运行结束后，程序会将模型结果打印到屏幕，说明运行成功。
Windows 下，你需要手动下载模型，然后执行`run.sh` 里面的 Python 命令即可。
