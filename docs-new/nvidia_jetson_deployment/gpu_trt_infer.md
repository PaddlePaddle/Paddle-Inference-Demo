# GPU TensorRT 加速推理(NV-GPU/Jetson)

## 一：概要

如果您的机器上已经安装 TensorRT 的话，那么你可以选择下载或编译带有 TensorRT 加速功能的 Paddle Inference 库。

NVIDIA TensorRT 是一个高性能机器学习推理SDK，专注于深度学习模型在NVIDIA硬件的快速高效的推理。PaddlePaddle 以子图方式集成了TensorRT，将可用TensorRT加速的算子组成子图供给 TensorRT，以获取 TensorRT 加速的同时，保留 PaddlePaddle 即训即推的能力。在这篇文章中，我们会介绍如何使用 Paddle-TRT 加速预测。


当模型被 Paddle 加载后，神经网络被表示为由变量和运算节点组成的计算图。在图分析阶段，Paddle 会对模型图进行分析同时发现图中可以使用 TensorRT 优化的子图，并使用 TensorRT 节点替换它们。在模型的推理期间，如果遇到 TensorRT 节点，Paddle 会调用 TensorRT 库对该节点进行优化，其他的节点调用 Paddle 的GPU 原生推理。TensorRT 除了有常见的OP融合以及显存/内存优化外，还针对性的对OP进行了优化加速实现，降低预测延迟，提升推理吞吐。

目前 Paddle-TRT 支持静态 shape 模式以及/动态 shape 模式。静态 shape 模式下支持图像分类，分割，检测模型，同时也支持Fp16， Int8 的预测加速。在动态shape模式下，除了对动态 shape 的图像模型（FCN， Faster rcnn）支持外，同时也对 NLP 的 Bert/Ernie 模型也进行了支持。

如果您需要安装 [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-6x-download)，请参考 [trt文档](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html).


目前 Paddle-TRT 支持静态 shape、动态 shape 两种运行方式。静态 shape 用于模型输入 shape 除 batch 维，其他维度信息不变的情况；动态 shape 可用于输入 size 任意变化的模型， 比如 NLP、OCR 等领域模型的支持，当然也包括静态 shape 支持的模型； 静态 shape 和动态 shape 都支持fp32、fp16、int8 等多种计算精度；Paddle-TRT 支持服务器端GPU，如T4、A10， 也支持边缘端硬件，如 Jetson NX、 Jetson Nano、 Jetson TX2 等。 在边缘硬件上，除支持常规的 GPU 外，还可用 DLA 进行推理；也支持 RTX2080，3090 等游戏显卡； 

用 TensorRT 首次推理时，TensorRT需要进行各 OP 融合、显存复用、以及 OP 的 kernel 选择等，导致首帧耗时过长，Paddle-TRT 开放了序列化接口，用于将 TensorRT分析的信息进行存储，在后续推理直接载入相关序列化信息，从而减少启动耗时；


Paddle Inference 提供了 Ubuntu/Windows/MacOS 平台的官方 Release 预测库下载，其均支持 TensorRT 加速推理，如果您使用的是以上平台，我们优先推荐您通过以下链接直接下载，或者您也可以参照文档进行[源码编译](../user_guides/source_compile.html)。

- [下载安装 Ubuntu 预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux)
  - 此链接中名称前缀包含 `nv_jetson` 的为用于NV Jetson平台的预测库。
- [下载安装 Windows 预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows)
- [下载安装 MacOS 预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#mac)

**Note:**

1. 从源码编译时，TensorRT 预测库目前仅支持使用 GPU 编译，且需要设置编译选项 TENSORRT_ROOT 为 TensorRT 所在的路径。
2. Windows 支持需要 TensorRT 版本5.0以上。
3. 使用 Paddle-TRT 的动态 shape 输入功能要求 TRT 的版本在6.0以上。

## 二：API使用介绍

在上一节中，我们了解到 Paddle Inference 预测流程包含了以下五步：

- 配置推理选项
- 创建 predictor
- 准备模型输入
- 模型推理
- 获取模型输出

使用 Paddle-TRT 也是遵照这样的流程。我们先用一个简单的例子来介绍这一流程（我们假设您已经对Paddle Inference有一定的了解，如果您刚接触Paddle Inference，请访问 `这里 <https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html>`_ 对Paddle Inference有个初步认识。）：


```python
    import numpy as np
    import paddle.inference as paddle_infer
    
    def create_predictor():
        config = paddle_infer.Config("./resnet50/model", "./resnet50/params")
        config.enable_memory_optim()
        config.enable_use_gpu(1000, 0)
        
        # 打开TensorRT。此接口的详细介绍请见下文
        config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                      max_batch_size = 1, 
                                      min_subgraph_size = 3, 
                                      precision_mode=paddle_infer.PrecisionType.Float32, 
                                      use_static = False, use_calib_mode = False)

        predictor = paddle_infer.create_predictor(config)
        return predictor

    def run(predictor, img):
        # 准备输入
        input_names = predictor.get_input_names()
        for i,  name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)   
            input_tensor.copy_from_cpu(img[i].copy())
        # 预测
        predictor.run()
        results = []
        # 获取输出
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results

    if __name__ == '__main__':
        pred = create_predictor()
        img = np.ones((1, 3, 224, 224)).astype(np.float32)
        result = run(pred, [img])
        print ("class index: ", np.argmax(result[0][0]))
```

通过例子我们可以看出，我们通过 `enable_tensorrt_engine` 接口来打开 TensorRT 选项的。

```python
    config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                  max_batch_size = 1, 
                                  min_subgraph_size = 3, 
                                  precision_mode=paddle_infer.PrecisionType.Float32, 
                                  use_static = False, use_calib_mode = False)
```
接下来让我们看下该接口中各个参数的作用:  

- **workspace_size**，类型：int，默认值为1 << 30 （1G）。指定TensorRT使用的工作空间大小，TensorRT会在该大小限制下筛选最优的kernel进行推理。
- **max_batch_size**，类型：int，默认值为1。需要提前设置最大的batch大小，运行时batch大小不得超过此限定值。
- **min_subgraph_size**，类型：int，默认值为3。Paddle-TRT是以子图的形式运行，为了避免性能损失，当子图内部节点个数大于 min_subgraph_size 的时候，才会使用Paddle-TRT运行。
- **precision_mode**，类型：**paddle_infer.PrecisionType**, 默认值为 **paddle_infer.PrecisionType.Float32**。指定使用TRT的精度，支持FP32（Float32），FP16（Half），Int8（Int8）。若需要使用Paddle-TRT int8离线量化校准，需设定precision为 **paddle_infer.PrecisionType.Int8** , 且设置 **use_calib_mode** 为True。
- **use_static**，类型：bool, 默认值为False。如果指定为True，在初次运行程序的时候会将TRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成。
- **use_calib_mode**，类型：bool, 默认值为False。若要运行Paddle-TRT int8离线量化校准，需要将此选项设置为True。


## 三：运行Dynamic shape

从1.8 版本开始， Paddle对 TensorRT 子图进行了Dynamic shape的支持。
使用接口如下：

```python
	config.enable_tensorrt_engine(
		workspace_size = 1<<30,
		max_batch_size=1, min_subgraph_size=5,
		precision_mode=paddle_infer.PrecisionType.Float32,
		use_static=False, use_calib_mode=False)
		  
	min_input_shape = {"image":[1,3, 10, 10]}
	max_input_shape = {"image":[1,3, 224, 224]}
	opt_input_shape = {"image":[1,3, 100, 100]}

	config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape, opt_input_shape)
```

从上述使用方式来看，在 config.enable_tensorrt_engine 接口的基础上，新加了一个config.set_trt_dynamic_shape_info 的接口。"image"对应模型文件中输入的名称。

该接口用来设置模型输入的最小，最大，以及最优的输入shape。 其中，最优的shape处于最小最大shape之间，在预测初始化期间，会根据opt shape对op选择最优的kernel。   

调用了 **config.set_trt_dynamic_shape_info** 接口，预测器会运行TRT子图的动态输入模式，运行期间可以接受最小，最大shape间的任意的shape的输入数据。


## 四：Paddle-TRT子图运行原理


Paddle Inference 采用子图的形式对 TensorRT 进行集成，当模型加载后，神经网络可以表示为由变量和运算节点组成的计算图。Paddle TensorRT实现的功能是对整个图进行扫描，发现图中可以使用 TensorRT 优化的子图，并使用 TensorRT 节点替换它们。在模型的推断期间，如果遇到 TensorRT 节点，Paddle Inference 会调用 TensorRT 库对该节点进行优化，其他的节点调用 Paddle 的原生实现。TensorRT 在推断期间能够进行 Op 的横向和纵向融合，过滤掉冗余的 Op，并对特定平台下的特定的Op选择合适的kernel等进行优化，能够加快模型的预测速度。  

下图使用一个简单的模型展示了这个过程：  

**原始网络**

<img src=https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_original.png >

**转换的网络**

<img src=https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_trt.png> 

我们可以在原始模型网络中看到，绿色节点表示可以被TensorRT支持的节点，红色节点表示网络中的变量，黄色表示Paddle只能被Paddle原生实现执行的节点。那些在原始网络中的绿色节点被提取出来汇集成子图，并由一个TensorRT节点代替，成为转换后网络中的 **block-25** 节点。在网络运行过程中，如果遇到该节点，Paddle将调用TensorRT库来对其执行。
