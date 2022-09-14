# Paddle Inference GPU 多流推理

本文主要介绍 GPU 推理中，流相关的配置策略，主要分为默认流策略和外部流策略。注意该功能会在2.4发布，目前仅支持 C++ API，且仅在 develop 分支上可用。

- [1. 默认流策略](#1)
- [2. 外部流策略](#2)
  - [2.1 执行器流策略](#2.1)
  - [2.2 batch流策略](#2.2)
- [3. 问题反馈](#3)

<a name="1"></a>

## 1. 默认流策略

Paddle Inference 默认情况下为默认流策略，即推理过程中默认使用 Paddle 内部维护的进程级 stream, 同一进程中的多个 predictor 均向该 stream 提交任务。该种策略继承自旧版 Paddle Inference, 故而稳定性较好，但无法充分发挥显卡多 stream 的优势，此外也无法满足部分业务场景需要控制 stream 的需求：如模型需运行在业务提供的 stream 上等。

<a name="2"></a>

## 2. 外部流策略

外部流策略即推理引擎 predictor 可以接受外部用户传递的 stream, 这样即可保证模型运行到外部用户设置的 stream 上。


<a name="2.1"></a>

### 执行器流策略

执行器流即每个执行器绑定一个流。用户可通过配置以下接口，设置推理引擎 Predictor 使用外部 stream. 

```c++
config.SetExecStream(stream);
```

注意，如果您需要 clone 执行器，此时您需要为每个 clone 的执行器提供外部 stream, 即保证每个执行器在单独的 stream 上运行。 

```c++
auto predictor2 = main_predictor->clone(stream2);
```

C++ 示例代码见[multi_stream_demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/gpu/multi_stream)

<a name="2.2"></a>

### batch流策略

batch 流策略指的是每次喂入数据运行推理的时候传入 stream, 仅在某些特殊业务场景下使用。注意，batch 流策略的内部实现依赖于执行器流策略，因此您在初始化的过程中必须传入外部流来初始化运行时所需的资源，初始化完成后，即可根据业务场景每次运行传入不同的 stream.

使用 batch 流策略推理需按执行器流策略进行初始化，且使用特殊接口调用推理过程：

```c++
config.SetExecStream(stream1);

....

paddle_infer::experimental::InternalUtils::RunWithExternalStream(predictor, stream2);
```

C++ 示例代码见[batch_stream_demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/gpu/experimental/batch_stream)


<a name="3"></a>

## 3. 问题反馈

由于 Paddle 算子体系更替等原因，旧算子体系并没有对此功能进行完全的支持（新算子体系已支持），目前仅对常用的旧算子进行了改造，如果您在使用外部流策略的过程中遇到了随机挂或精度 diff 的问题，请尝试使用默认流策略，如果默认流策略结果正常，但外部流策略异常，请设置环境变量 `GLOG_v=10` 在 github issue 中贴上 log 反馈，感谢。