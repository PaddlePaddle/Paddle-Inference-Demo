# Python 预测样例

**如果您看到这个目录，我们会假设您已经对 Paddle Inference 有了一定的了解。如果您刚刚接触 Paddle Inference 不久，建议您[访问这里](https://paddle-inference.readthedocs.io/en/latest/#)对 Paddle Inference 做一个初步的认识。**

本目录提供 Paddle Inference 各个功能的使用样例。目录结构及功能如下所示，您可以根据自己的需求选择合适的样例。

```
├── advanced
│   ├── share_external_data       share_external_data 预测样例
│   ├── multi_thread              多线程 预测样例
│   └── custom_operator           自定义算子 样例      
├── cpu
│   ├── resnet50                  单输入模型 oneDNN/OnnxRuntime 预测样例
│   └── yolov3                    多输入模型 oneDNN/OnnxRuntime 预测样例
├── gpu
│   ├── resnet50                  单输入模型 原生gpu/gpu混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例
│   ├── yolov3                    多输入模型 原生gpu/gpu混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例
│   └── tuned_dynamic_shape       Trt动态shape自动推导 预测样例
└── mixed
    ├── ELMo                      ELMo 预测样例
    ├── mask_detection            口罩检测 预测样例
    └── x86_lstm_demo             Lstm 预测样例
```
