# Paddle Inference Demos



Paddle Inference为飞桨核心框架推理引擎。Paddle Inference功能特性丰富，性能优异，针对服务器端应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。


为了能让广大用户快速的使用Paddle Inference进行部署应用，我们在此Repo中提供了C++、Python的使用样例。


**在这个repo中我们会假设您已经对Paddle Inference有了一定的了解。**

**如果您刚刚接触Paddle Inference不久，建议您[访问这里](https://paddle-inference.readthedocs.io/en/latest/#)对Paddle Inference做一个初步的认识。**


## 测试样例

1） 在python目录中，我们通过真实输入的方式罗列了一系列的测试样例，其中包括图像的分类，分割，检测，以及NLP的Ernie/Bert等Python使用样例，同时也包含Paddle-TRT， 多线程的使用样例。

2） 在c++目录中，我们通过单测方式展现了一系列的测试样例，其中包括图像的分类，分割，检测，以及NLP的Ernie/Bert等C++使用样例，同时也包含Paddle-TRT， 多线程的使用样例。

注意：如果您使用2.0以前的Paddle，请参考[release/1.8分支](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/release/1.8)

> **C++ 部署示例速查列表**

|    示例名称   |   功能概述  | 
| :---- | :---- | 
|ascend310 |[晟腾310 预测样例](../../c++/ascend310/)|
|IPU |[IPU 预测样例](../../c++/IPU/)|
|cpu/resnet50  |[单输入模型 oneDnn/ONNXRuntime 预测样例](../../c++/cpu/resnet50/)|
|cpu/yolov3|[多输入模型 oneDnn/ONNXRuntime 预测样例](../../c++/cpu/yolov3/)|
|gpu/resnet50|[单输入模型 原生GPU/TensorRT_fp32/TensorRT_fp16/TensorRT_int8/TensorRT_dynamic_shape 预测样例](../../c++/gpu/resnet50/)|
|gpu/yolov3|[多输入模型 原生GPU/TensorRT_fp32/TensorRT_fp16/TensorRT_int8/TensorRT_dynamic_shape 预测样例](../../c++/gpu/yolov3/)|
|gpu/tuned_dynamic_shape|[TensorRT动态shape自动推导 预测样例](../../c++/gpu/tuned_dynamic_shape/)|
|gpu/ernie_varlen|[ernie 变长预测样例](../../c++/gpu/ernie-varlen/)|
|gpu/gpu_fp16|[GPU 混合精度推理 预测样例](../../c++/gpu/gpu_fp16/)|
|gpu/multi_stream|[GPU 多流 预测样例](../../c++/gpu/multi_stream/)|
|advanced/custom_operator|[自定义算子 样例](../../c++/advanced/custom_operator/)|
|advanced/share_external_data|[share_external_data 预测样例](../../c++/advanced/share_external_data/)|
|advanced/multi_thread|[多线程预测样例](../../c++/advanced/multi_thread/)|
|advanced/x86_gru_int8|[slim_int8 预测样例](../../c++/advanced/custom-operator/ ../../c++/advanced/x86_gru_int8/)|
|mixed/LIC2020|[LIC2020比赛 预测样例](../../c++/mixed/LIC2020/)|


> **Python 部署示例速查列表**

|    示例名称   |   功能概述  | 
| :---- | :---- | 
|cpu/resnet50|[单输入模型 oneDnn/ONNXRuntime 预测样例](../../python/cpu/resnet50/)|
|cpu/yolov3|[多输入模型 oneDnn/ONNXRuntime 预测样例](../../python/cpu/yolov3/)|
|gpu/resnet50|[单输入模型 原生GPU/GPU混合精度推理/TensorRT_fp32/TensorRT_fp16/TensorRT_int8/TensorRT_dynamic_shape 预测样例](../../python/gpu/resnet50/)|
|gpu/yolov3|[多输入模型 原生GPU/GPU混合精度推理/TensorRT_fp32/TensorRT_fp16/TensorRT_int8/TensorRT_dynamic_shape 预测样例](../../python/gpu/yolov3/)|
|gpu/tuned_dynamic_shape|[TensorRT动态shape自动推导 预测样例](../../python/gpu/tuned_dynamic_shape/)|
|advanced/custom_operator|[自定义算子 样例](../../python/advanced/custom_operator/)|
|advanced/share_external_data|[share_external_data 预测样例](../../python/advanced/share_external_data/)|
|advanced/multi_thread|[多线程预测样例](../../python/advanced/multi_thread/)|
|mixed/ELMo|[ELMo 预测样例](../../python/mixed/ELMo/)|
|mixed/mask_detection|[口罩检测预测样例](../../python/mixed/mask_detection/)|
|mixed/x86_lstm_demo|[Lstm 预测样例](../../python/mixed/x86_lstm_demo/)|
   

> **Go 部署示例速查列表**

|    示例名称   |   功能概述  | 
| :---- | :---- | 
|resnet50|[Go 预测样例](../../go/resnet50/)|


