Using the Paddle-TensorRT Repository for Inference
================


NVIDIA TensorRT is an SDK for high-performance deep learning inference. It can lower the latency of the inference applications and improve their throughput. PaddlePaddle integrates TensorRT with subgraph design, so we can use the TensorRT module to enhance the performance of the Paddle model during the inference process. In this article, we will walk through how to use the subgraph module of Paddle-TRT to accelerate the inference. 

If you need to install `TensorRT <https://developer.nvidia.com/nvidia-tensorrt-6x-download>`_, please refer to the `TensorRT document <https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html>`_.

Overview
----------------

After the model is loaded, the neural network can be represented as a computing graph composed of variables and operation nodes. When the TRT subgraph mode is turned on, paddle will detect that the subgraph which can be optimized by TensorRT in the model during the graph analysis phase and replace it with TensorRT nodes. During the model inference, if a tensorrt node is encountered, paddle will call the TensorRT library to infer the node, and other nodes will call the native implementation of paddle.

At present,  TensorRT subgraph pattern supports two operation modes: static shape and dynamic shape. Static shape is mainly used when the input size of the model is unchanged except for the batch dimension; Dynamic shape can be used to  the models in which the dims of inputs  may change arbitraryly, such as NLP, OCR and other domain models. Of course, it also includes models supported by static shape; Both static shape and dynamic shape support fp32, fp16, int8 and other calculation precision; The TensorRT subgraph pattern supports server-side GPUs, such as T4 and A10, and edge side hardware, such as Jetson NX, Jetson Nano, Jetson TX2 and so on, in addition to supporting conventional GPU, DLA can also be used for reasoning; It also supports game graphics cards such as RTX2080、RTX3090;


When using TensorRT for the first time, TensorRT will consume a long time to perform OP fusion, device memory reuse, and optimal kernel selection . Paddle Inference opens the serialization interface to store the information analyzed by TensorRT, and directly loads the relevant serialization information in the subsequent inference to reduce the startup time;


**Note:**

1. During the compilation of the source code, the TensorRT inference repository only supports GPU compilation, and TENSORRT_ROOT is required to be set to the path of TensorRT. 
2. Only TensorRT versions above 7.0 are supported.

I. Environment Preparation
-------------

To use TensorRT, there are three ways to prepare the runtime environment of Paddle: 

1）Using pip to install a whl file under linux

Download a whl file with the consistent environment and TensorRT from `whl list <https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release>`_, and install it using pip. 

2）Using the docker

.. code:: shell

  # Pull the docker, where the Paddle 2.2 Python environment has been preinstalled and there is a precompiled library (c++) put in the main directory ～/.
  docker pull paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82

  sudo nvidia-docker run --name your_name -v $PWD:/paddle  --network=host -it paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82  /bin/bash

3）Manual Compilation  
Please refer to the `compilation document <../user_guides/source_compile.html>`_ 

**Note1：** During the cmake, please set TENSORRT_ROOT （the path of TRT lib）and WITH_PYTHON （set "whether to produce the python whl file" to ON).

**Note2:** There will be errors of TensorRT during the compilation.

Add virtual destructors to class IPluginFactory and class IGpuAllocator of NvInfer.h (trt5) or NvInferRuntime.h (trt6) file respectively by hand:

.. code:: c++

  virtual ~IPluginFactory() {};
  virtual ~IGpuAllocator() {};
  
Change **protected: ~IOptimizationProfile() noexcept = default;** in `NvInferRuntime.h` (trt6)

to

.. code:: c++

  virtual ~IOptimizationProfile() noexcept = default;
  

II. Introduction to the usage of APIs
-----------------

In the section of `the inference process <https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html>`_, we have got to know that there are five parts of Paddle Inference:

- Configuration of inference options
- Creation of the predictor
- Preparation for the model input
- Model inference
- Acquisition of the model output

Paddle-TRT also follows the same process. Let's use a simple example to introduce it (It is assumed that you have known about the Paddle Inference). If you are new to this, you can visit <https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html>`_ to get started.

.. code:: python

    import numpy as np
    import paddle.inference as paddle_infer
    
    def create_predictor():
        config = paddle_infer.Config("./resnet50/model", "./resnet50/params")
        config.enable_memory_optim()
        config.enable_use_gpu(1000, 0)
        
        # Open TensorRT. The details of this interface will be mentioned in the following part.
        config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                      max_batch_size = 1, 
                                      min_subgraph_size = 3, 
                                      precision_mode=paddle_infer.PrecisionType.Float32, 
                                      use_static = False, use_calib_mode = False)

        predictor = paddle_infer.create_predictor(config)
        return predictor

    def run(predictor, img):
        # Preparation for the input
        input_names = predictor.get_input_names()
        for i,  name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)   
            input_tensor.copy_from_cpu(img[i].copy())
        # Inference
        predictor.run()
        results = []
        # Acquisition of the output
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

From this example, it is clear that we open TensorRT options through the interface of `enable_tensorrt_engine`.

.. code:: python

    config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                  max_batch_size = 1, 
                                  min_subgraph_size = 3, 
                                  precision_mode=paddle_infer.PrecisionType.Float32, 
                                  use_static = False, use_calib_mode = False)

Then, let's have a look at the function of each parameter in the interface:

- **workspace_size**，type：int，and the default value is 1 << 30 （1G）. It designates the size of the working space of TensorRT, and TensorRT will sort out the optimum kernel for the execution of the inference computation under this limitation. 
- **max_batch_size**，type：int，and the default value is 1. The maximum batch is required to be set beforehand, and the batch size cannot exceed this max value in the execution. 
- **min_subgraph_size**，type：int，and the default value is 3. Paddle-TRT is operated in subgraphs. In order to avoid performance loss, Paddle-TRT will be operated only when the number of nodes within subgraphs is more than min_subgraph_size.
- **precision_mode**，type: **paddle_infer.PrecisionType**, and the default value is **paddle_infer.PrecisionType.Float32**. It designates the precision of TRT, and supports FP32（Float32）,FP16（Half）,and Int8（Int8）. If you need to use the post-training quantization (PTQ, or offline quantization) calibration of Paddle-TRT int8, set the precision to **paddle_infer.PrecisionType.Int8** and **use_calib_mode** to True.
- **use_static**，type：bool, and the default value is False. If it is designated as True, then the optimized TRT information will be serialized to the disk during the first run of the program, and will be directly loaded next time without regeneration.
- **use_calib_mode**，type：bool, and the default value is False. If you need to use the PTQ calibration of Paddle-TRT int8, set this to True. 

Int8 Quantization Inference
>>>>>>>>>>>>>>

To some extent, the parameters of the neural network are redundant. And in many tasks, we can turn the Float32 model into the Int8 model with the cost of an acceptable precision loss, in order to reduce the computation amount, computation time, memory used, and the model size. There are two steps to use Int8 for quantized inference: 1) produce the quantized model; 2) load the quantized model for Int8 inference. In the following part, we will elaborate on how to use Paddle-TRT for Int8 quantized inference.

**1. Produce the quantized model**

There are two methods are supported currently: 

a. Use the built-in functionality of TensorRT-- Int8 PTQ calibration. In calibration, a calibration table is made based on the trained FP32 model and a few calibrated data (e.g. about 500-1000 images), and during the inference, the FP32 model and the table can be used for the Int8 precision inference. Follow the guide to make the calibration table: 

  - When configurating TensorRT，set **precision_mode** to **paddle_infer.PrecisionType.Int8** and **use_calib_mode** to **True**.

    .. code:: python

      config.enable_tensorrt_engine(
        workspace_size=1<<30,
        max_batch_size=1, min_subgraph_size=5,
        precision_mode=paddle_infer.PrecisionType.Int8,
        use_static=False, use_calib_mode=True)

  - Prepare about 500 real input images, and run the model with the above configuration. (Paddle-TRT counts the range value of every tensor and records it in the table. After the running, the table will be written into `_opt_cache`. 

  If you want to know the code of making the calibration table using TensorRT's built-in functionality of Int8 PTQ calibration, please refer to `the demo here <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/README.md#%E7%94%9F%E6%88%90%E9%87%8F%E5%8C%96%E6%A0%A1%E5%87%86%E8%A1%A8>`_ .

b. Use the model compression tool library-- PaddleSlim to make the quantized model. PaddleSlim supports offline quantization and online quantization. And the offline quantization is similar to TensorRT PTQ calibration in principle; online quantization is also called quantization aware training (QAT), which depends on massive data (e.g. >=5000 images) to retrain the pretrained model and uses quantization simulation to update the weight in the training so that errors can be reduced. If you want to learn about how to make the quantized model using PaddleSlim, please refer to:
  
  - Post-training quantization `quick start <https://paddlepaddle.github.io/PaddleSlim/quick_start/quant_post_tutorial.html>`_
  - Post-training quantization `API description <https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-post>`_
  - Post-training quantization `Demo <https://github.com/PaddlePaddle/PaddleSlim/tree/release/1.1.0/demo/quant/quant_post>`_
  - Quant aware training `quick start <https://paddlepaddle.github.io/PaddleSlim/quick_start/quant_aware_tutorial.html>`_
  - Quant aware training `API description <https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-aware>`_
  - Quant aware training `Demo <https://github.com/PaddlePaddle/PaddleSlim/tree/release/1.1.0/demo/quant/quant_aware>`_

In PTQ, retraining is not required, but the precision may be affected. In QAT, the precision may be less affected, but retraining is required, and it is more complicated to perform QAT. Practically speaking, it is recommended to use the TRT functionality of PTQ calibration to make the quantized model. If the precision cannot meet the standard, then resort to PaddleSlim. 
  
**2. Load the quantized model for Int8 inference**       

  First, in the configuration of TensorRT, set **precision_mode** to **paddle_infer.PrecisionType.Int8** .

  If the quantized model is made by the TRT PTQ calibration, set **use_calib_mode** to **True** ：

  .. code:: python

    config.enable_tensorrt_engine(
      workspace_size=1<<30,
      max_batch_size=1, min_subgraph_size=5,
      precision_mode=paddle_infer.PrecisionType.Int8,
      use_static=False, use_calib_mode=True)

  For the complete demo, please refer to `here <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/README.md#%E5%8A%A0%E8%BD%BD%E6%A0%A1%E5%87%86%E8%A1%A8%E6%89%A7%E8%A1%8Cint8%E9%A2%84%E6%B5%8B>`_.
  
  If the quantized model is made by PaddleSlim quantization，set **use_calib_mode** to **False** ：

  .. code:: python

    config.enable_tensorrt_engine(
      workspace_size=1<<30,
      max_batch_size=1, min_subgraph_size=5,
      precision_mode=paddle_infer.PrecisionType.Int8,
      use_static=False, use_calib_mode=False)

  For the complete demo, please refer to `here <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/README.md#%E4%B8%89%E4%BD%BF%E7%94%A8trt-%E5%8A%A0%E8%BD%BDpaddleslim-int8%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B>`_ .

Run dynamic shape
>>>>>>>>>>>>>>

Since version 1.8, Paddle has begun to support the dynamic shape for the TRT subgraph.
APIs adopted here include：

.. code:: python

  config.enable_tensorrt_engine(
    workspace_size = 1<<30,
    max_batch_size=1, min_subgraph_size=5,
    precision_mode=paddle_infer.PrecisionType.Float32,
    use_static=False, use_calib_mode=False)
      
  min_input_shape = {"image":[1,3, 10, 10]}
  max_input_shape = {"image":[1,3, 224, 224]}
  opt_input_shape = {"image":[1,3, 100, 100]}

  config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape, opt_input_shape)


It can be seen that on the basis of config.enable_tensorrt_engine，there is another interface--config.set_trt_dynamic_shape_info added.  

The newly added interface is used to set the minimum, maximum, and optimum input shapes. The optimum shape lies between the minimum and the maximum. At the beginning of the inference, the optimum kernel of OPs will be chosen according to the optimum shape. 

The **config.set_trt_dynamic_shape_info** interface is adopted, and the predictor will run the dynamic input mode of the TRT subgraph. During the running, any input shape between the minimum and the maximum is OK. 


III. Test demo
-------------

More demos using the TRT subgraph for inference are provided on the github. 

- For Python demos, please refer to `the link <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/paddle_trt>`_ .
- For C++ demos, please refer to `the link <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt>`_ .

IV. The principle of the Paddle-TRT subgraph
---------------

   PaddlePaddle uses the subgraph to integrate TensorRT, and after loading the model, the neural network can be presented as a computing chart consisting of variables and computing nodes. Paddle TensorRT scans the whole image, detects subgraphs which can be optimized by TensorRT, and replaces them with its nodes. If encountering TensorRT nodes, Paddle will adopt the TensorRT repository to optimize them and use its original implementation for other nodes. During the inference, TensorRT can merge OPs both horizontally and vertically, filter out redundant OPs, and choose optimum kernels to optimize OPs in certain platforms so that the model inference can be accelerated. 

The following figure shows the process by taking a simple model as an example: 

**Original Network**

  .. image:: https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_original.png

**Converted Network**

  .. image:: https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_trt.png

 From the original network, we can know that the green nodes are those supported by TensorRT, that the red ones are variables in the network, and that the yellow ones are the nodes that only can be executed by Paddle's original implementation. Those green nodes are extracted from the original network and integrated into subgraphs. Then they are replaced with a TensorRT node and turn into the **block-25** node. When meeting this node, Paddle will call the TensorRT repository to execute it. 



