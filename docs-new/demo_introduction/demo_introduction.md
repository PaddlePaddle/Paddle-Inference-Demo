# Paddle Inference 部署示例

在 [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo) 中，提供了 C++、Python、Go 三种语言在不同平台下进行推理的示例。


> **C++ 部署示例速查列表**

|    示例名称   |   功能概述  | 
| :---- | :---- | 
|ascend310 |[晟腾310 预测样例](../../c++/ascend310/)|
|IPU |[IPU 预测样例](../../c++/ipu/)|
|cpu/resnet50  |[单输入模型 oneDnn/ONNXRuntime 预测样例](../../c++/cpu/resnet50/)|
|cpu/yolov3|[多输入模型 oneDnn/ONNXRuntime 预测样例](../../c++/cpu/yolov3/)|
|gpu/resnet50|[单输入模型 原生GPU/TensorRT_fp32/TensorRT_fp16/TensorRT_int8/TensorRT_dynamic_shape 预测样例](../../c++/gpu/resnet50/)|
|gpu/yolov3|[多输入模型 原生GPU/TensorRT_fp32/TensorRT_fp16/TensorRT_int8/TensorRT_dynamic_shape 预测样例](../../c++/gpu/yolov3/)|
|gpu/tuned_dynamic_shape|[TensorRT动态shape自动推导 预测样例](../../c++/gpu/tuned_dynamic_shape/)|
|gpu/ernie_varlen|[ernie 变长预测样例](../../c++/gpu/ernie-varlen/)|
|gpu/gpu_fp16|[GPU 混合精度推理 预测样例](../../c++/gpu/gpu_fp16/)|
|gpu/multi_stream|[GPU 多流 预测样例](../../c++/gpu/multi_stream/)|
|advanced/custom_operator|[自定义算子 样例](../../c++/advanced/custom-operator/)|
|advanced/share_external_data|[share_external_data 预测样例](../../c++/advanced/share_external_data/)|
|advanced/multi_thread|[多线程预测样例](../../c++/advanced/multi_thread/)|
|advanced/x86_gru_int8|[slim_int8 预测样例](../../c++/advanced/x86_gru_int8/)|
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

