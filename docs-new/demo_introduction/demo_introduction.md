# Paddle Inference 部署示例

在 [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo) 中，提供了 C++、Python、Go 三种语言在不同平台下进行推理的示例。


> **C++ 部署示例速查列表**

|    示例名称   |   功能概述  | 
| :---- | :---- | 
|ascend310 |[晟腾310 预测样例](../../c++/ascend310/)|
|ipu |ipu 预测样例|
|cpu/resnet50  |单输入模型 Onednn/OnnxRuntime 预测样例 |
|cpu/yolov3|多输入模型 Onednn/OnnxRuntime 预测样例|
|gpu/resnet50|单输入模型 原生gpu/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例|
|gpu/yolov3|多输入模型 原生gpu/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例|
|gpu/tuned_dynamic_shape|Trt_tuned_dynamic_shape 预测样例|
|gpu/ernie_varlen|ernie 变长预测样例|
|gpu/gpu_fp16|gpu 混合精度推理 预测样例|
|gpu/multi_stream|gpu 多流 预测样例|
|advanced/custom_operator|自定义算子 样例|
|advanced/share_external_data|share_external_data 预测样例|
|advanced/multi_thread|多线程预测样例|
|advanced/x86_gru_int8|slim_int8 预测样例|
|mixed/LIC2020|LIC2020比赛 预测样例|


> **Python 部署示例速查列表**

|    示例名称   |   功能概述  | 
| :---- | :---- | 
|cpu/resnet50  |单输入模型 Onednn/OnnxRuntime 预测样例 |
|cpu/yolov3|多输入模型 Onednn/OnnxRuntime 预测样例|
|gpu/resnet50|单输入模型 原生gpu/gpu混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例|
|gpu/yolov3|多输入模型 原生gpu/gpu混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例|
|gpu/tuned_dynamic_shape|Trt动态shape自动推导 预测样例|
|advanced/custom_operator|自定义算子 样例|
|advanced/share_external_data|share_external_data 预测样例|
|advanced/multi_thread|多线程预测样例|
|mixed/ELMo|ELMo 预测样例|
|mixed/mask_detection|口罩检测 预测样例|
|mixed/x86_lstm_demo|Lstm 预测样例|
   

> **Go 部署示例速查列表**

|    示例名称   |   功能概述  | 
| :---- | :---- | 
|resnet50|go 预测样例 |

