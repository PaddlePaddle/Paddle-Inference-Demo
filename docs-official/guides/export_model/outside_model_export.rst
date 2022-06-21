其他框架模型导出
=====================

Pytorch、ONNX、TensorFlow、Caffe模型，可以通过 X2Paddle 工具完成模型转换，转到 Paddle 模型后，即可使用 Paddle Inference 完成部署。

`X2Paddle <https://github.com/PaddlePaddle/X2Paddle>`_ 是飞桨生态下的模型转换工具，致力于帮助你快速迁移其他深度学习框架至飞桨框架。目前支持 **推理模型的框架转换** 与 **PyTorch训练代码迁移** ，除此之外还提供了详细的不同框架间API对比文档，降低你上手飞桨核心的学习成本。

1.安装模型转换工具X2Padlde
---------------

使用pip安装

.. code:: shell

	pip install x2paddle

使用源码安装

.. code:: shell

	git clone https://github.com/PaddlePaddle/X2Paddle.git
	cd X2Paddle
	python setup.py install

2.模型转换
------------

2.1 Pytorch模型转换
>>>>>>>>>>>>>>

.. code-block:: python

	from x2paddle.convert import pytorch2paddle
	pytorch2paddle(module=torch_module,
				save_dir="./pd_model",
				jit_type="trace",
				input_examples=[torch_input])
	# module (torch.nn.Module): PyTorch的Module。
	# save_dir (str): 转换后模型的保存路径。
	# jit_type (str): 转换方式。默认为"trace"。
	# input_examples (list[torch.tensor]): torch.nn.Module的输入示例，list的长度必须与输入的长度一致。默认为None。


**script** 模式以及更多细节可参考 `PyTorch模型转换文档 <https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/pytorch2paddle.md>`_ 。

2.2 ONNX模型转换
>>>>>>>>>>

.. code:: shell

	x2paddle --framework onnx --model onnx_model.onnx --save_dir pd_model

2.3 TensorFlow模型转换
>>>>>>>>>>

.. code:: shell

	x2paddle --framework tensorflow --model model.pb --save_dir pd_model

2.4 Caffe模型转换
>>>>>>>>>>>>>>

.. code:: shell

	x2paddle --framework caffe --prototxt model.proto --weight model.caffemodel --save_dir pd_model

转换参数说明
>>>>>>>>>>>>>>

=====================  =============================================================================
    参数                                     作用 
=====================  =============================================================================
--framework            源模型类型 (pytorch、tensorflow、caffe、onnx)                         
--prototxt             当framework为caffe时，该参数指定caffe模型的proto文件路径     
--weight               当framework为caffe时，该参数指定caffe模型的参数文件路径 
--save_dir             指定转换后的模型保存目录路径                                 
--model                当framework为tensorflow/onnx时，该参数指定tensorflow的pb模型文件或onnx模型路径
--caffe_proto          **[可选]** 由caffe.proto编译成caffe_pb2.py文件的存放路径，当存在自定义Layer时使用，默认为None 
--define_input_shape   **[可选]** For TensorFlow, 当指定该参数时，强制用户输入每个Placeholder的shape，见 `文档 <https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/FAQ.md>`_ 
--enable_code_optim    **[可选]** For PyTorch, 是否对生成代码进行优化，默认为True
=====================  =============================================================================

更多参数可参考 `X2Paddle官网 <https://github.com/PaddlePaddle/X2Paddle#%E8%BD%AC%E6%8D%A2%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E>`_

X2Paddle API
>>>>>>>>>>>>>>

目前X2Paddle提供API方式转换模型，可参考 `X2PaddleAPI <https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/x2paddle_api.md>`_

转换结果说明
--------------

在指定的 **save_dir** 以下目录以及文件

1. inference_model : 目录下有静态图模型结构以及参数
2. x2paddle_code.py : 自动生成的动态图组网代码
3. model.pdparams : 动态图模型参数

**问题反馈**

X2Paddle使用时存在问题时，欢迎您将问题或Bug报告以 `Github Issues <https://github.com/PaddlePaddle/X2Paddle/issues>`_ 的形式提交给我们，我们会实时跟进。
