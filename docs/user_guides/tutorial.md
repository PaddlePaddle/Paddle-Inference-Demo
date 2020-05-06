## 使用流程

### 一： 模型准备
Paddle Inference目前支持的模型结构为PaddlePaddle深度学习框架产出的模型格式。因此，在您开始使用 Paddle Inference框架前您需要准备一个由PaddlePaddle框架保存的模型。 如果您手中的模型是由诸如Caffe2、Tensorflow等框架产出的，那么我们推荐您使用 X2Paddle 工具进行模型格式转换。


### 二： 环境准备

1） 如果您想用Paddle Inference Python API接口，请参照官方主页的引导，在您的环境中安装PaddlePaddle。

2） 如果您想使用Paddle Inference C++ API 接口，请参照接下来的[预测库的编译]()页面。


### 三：使用Paddle Inference执行预测

使用Paddle Inference进行推理部署的流程如下所示。  
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/10d5cee239374bd59e41283b3233f49dc306109da9d540b48285980810ab4e36" width="280" ></center>   

1) 配置推理选项。`AnalysisConfig`是飞桨提供的配置管理器API。在使用Paddle Inference进行推理部署过程中，需要使用`AnalysisConfig`详细地配置推理引擎参数，包括但不限于在何种设备（CPU/GPU）上部署(`config.EnableUseGPU`)、加载模型路径、开启/关闭计算图分析优化、使用MKLDNN/TensorRT进行部署的加速等。参数的具体设置需要根据实际需求来定。            

2) 创建`AnalysisPredictor`。`AnalysisPredictor`是Paddle Inference提供的推理引擎。你只需要简单的执行一行代码即可完成预测引擎的初始化，`std::unique_ptr<PaddlePredictor> predictor = CreatePaddlePredictor(config)`，config为1步骤中创建的`AnalysisConfig`。

3) 准备输入数据。执行 `auto input_names = predictor->GetInputNames()`，您会获取到模型所有输入tensor的名字，同时通过执行`auto tensor = predictor->GetInputTensor(input_names[i])`; 您可以获取第i个输入的tensor，通过`tensor->copy_from_cpu(data)` 方式，将data中的数据拷贝到tensor中。

4) 调用predictor->ZeroCopyRun()执行推理。           

5) 获取推理输出。执行 `auto out_names = predictor->GetOutputNames()`，您会获取到模型所有输出tensor的名字，同时通过执行`auto tensor = predictor->GetOutputTensor(out_names[i])`; 您可以获取第i个输出的tensor。通过 `tensor->copy_to_cpu(data)` 将tensor中的数据copy到data指针上。
。   
