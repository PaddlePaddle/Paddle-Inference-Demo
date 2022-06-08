# 运行 ResNet50 图像分类样例

ResNet50 样例展示了单输入模型在 GPU 下的推理过程。运行步骤如下：

## 一：获取 Paddle Inference 预测库

- [官网下载](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)
- 自行编译获取

将获取到的 Paddle Inference 预测库软链接或者重命名为 `paddle_inference`，并置于 `Paddle-Inference-Demo/c++/lib` 目录下。

## 二：获取 Resnet50 模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。

## 三：编译样例
 
- 文件`resnet50_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
- 脚本`run.sh` 为一键运行脚本。

编译前，需要根据自己的环境修改 `compile.sh` 中的相关代码配置依赖库：
```shell
# 编译的 demo 名称
DEMO_NAME=resnet50_test

# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-7.1.3.4
```

运行 `bash compile.sh` 编译样例。

## 四：运行样例

### 使用原生 GPU 运行样例

```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```

### 使用 TensorRT Fp32 运行样例

```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp32
```

### 使用 TensorRT Fp16 运行样例

```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp16
```

### 使用 TensorRT Int8 离线量化预测运行样例

在使用 TensorRT In8 离线量化预测运行样例时，相同的运行命令需要执行两次。第一次执行生成量化校准表，第二次加载校准表执行 Int8 预测。需要注意的是 TensorRT Int8 离线量化预测使用的仍然是 ResNet50 FP32 模型，是通过校准表中包含的量化 scale 在运行时将 FP32 转为 Int8 从而加速预测的。

#### 生成量化校准表

```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_int8
```

生成校准表的log：
```
I0623 08:40:49.386909 107053 tensorrt_engine_op.h:159] This process is generating calibration table for Paddle TRT int8...
I0623 08:40:49.387279 107057 tensorrt_engine_op.h:352] Prepare TRT engine (Optimize model structure, Select OP kernel etc). This process may cost a lot of time.
I0623 08:41:13.784473 107053 analysis_predictor.cc:791] Wait for calib threads done.
I0623 08:41:14.419198 107053 analysis_predictor.cc:793] Generating TRT Calibration table data, this may cost a lot of time...
```

执行后，模型文件夹`ResNet50`下的`_opt_cache`文件夹下会多出一个名字为`trt_calib_*`的文件，即校准表。

#### 加载校准表执行预测

```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_int8 --use_calib=true
```

加载校准表预测的log：
```
I0623 08:40:27.217701 107040 tensorrt_subgraph_pass.cc:258] RUN Paddle TRT int8 calibration mode...
I0623 08:40:27.217834 107040 tensorrt_subgraph_pass.cc:321] Prepare TRT engine (Optimize model structure, Select OP kernel etc). This process may cost a lot of time.
```

### 使用 TensorRT 加载 PaddleSlim Int8 量化模型预测
这里，我们首先下载 [ResNet50 PaddleSlim量化模型](https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ResNet50_quant.tar.gz)。

与加载离线量化校准表执行 Int8 预测的区别是，PaddleSlim 量化模型已经将 scale 保存在模型 op 的属性中，这里我们就不再需要校准表了，所以在运行样例时将 `use_calib` 配置为 false。

```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_int8 --use_calib=false
```

### 使用 TensorRT dynamic shape 运行样例（以 Fp32 为例）
```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp32 --use_dynamic_shape=1
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)