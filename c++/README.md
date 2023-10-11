# C++ 预测样例

**如果您看到这个目录，我们会假设您已经对 Paddle Inference 有了一定的了解。**

**如果您刚刚接触 Paddle Inference 不久，建议您[访问这里](https://paddle-inference.readthedocs.io/en/latest/#)对 Paddle Inference 做一个初步的认识。**

本目录提供 Paddle Inference 各个功能的使用样例。目录结构及功能如下所示，您可以根据自己的需求选择合适的样例。

```
├── ascend310                     晟腾310 预测样例
├── ipu                           ipu 预测样例
├── cpu                           
│   ├── resnet50                  单输入模型 oneDNN/OnnxRuntime 预测样例   
│   └── yolov3                    多输入模型 oneDNN/OnnxRuntime 预测样例
├── gpu
│   ├── resnet50                  单输入模型 原生gpu/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例
│   ├── yolov3                    多输入模型 原生gpu/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例
│   ├── tuned_dynamic_shape       Trt_tuned_dynamic_shape 预测样例
│   ├── ernie_varlen              ernie 变长预测样例
│   ├── gpu_fp16                  gpu 混合精度推理 预测样例
│   └── multi_stream              gpu 多流 预测样例
├── advanced 
│   ├── custom_operator           自定义算子 样例
│   ├── share_external_data       share_external_data 预测样例
│   ├── multi_thread              多线程 预测样例
│   ├── x86_gru_int8              slim_int8 预测样例
│   └── tensorrt_precision_debug  Paddle-TensorRT 精度调试工具 使用样例
├── mixed
│   └── LIC2020                   LIC2020比赛 预测样例
└── lib
```