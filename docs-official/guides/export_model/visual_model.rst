模型结构可视化
==============

通过 `快速开始 <https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html>`_ 一节，我们了解到，预测模型包含了两个文件，一部分为模型结构文件，通常以 **pdmodel** 文件存在；另一部分为参数文件，通常以 **pdiparams** 文件存在。

模型结构文件（*.pdmodel文件），顾名思义，存储了模型的拓扑结构，其中包括模型中各种OP的计算顺序以及OP的详细信息。很多时候，我们希望能够将这些模型的结构以及内部信息可视化，方便我们进行模型分析。Paddle生态开发了VisualDL来进行网络结构可视化工作。接下来将会通过两种方式来讲述如何对Paddle 预测模型进行可视化。

安装可视化工具VisualDL
>>>>>>>>>>>>>>

使用pip安装

.. code:: shell

	python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple

使用代码安装

.. code:: shell

	git clone https://github.com/PaddlePaddle/VisualDL.git
	cd VisualDL

	python setup.py bdist_wheel
	pip install --upgrade dist/visualdl-*.whl

可视化
>>>>>>>>>>>>>>

支持两种启动方式：

- 前端拖拽上传模型文件：

  - 无需添加任何参数，在命令行执行 visualdl 后启动界面上传文件即可：


.. image:: https://user-images.githubusercontent.com/48054808/88628504-a8b66980-d0e0-11ea-908b-196d02ed1fa2.png


- 后端透传模型文件：

  - 在命令行加入参数 --model 并指定 **模型文件** 路径（非文件夹路径），即可启动：

.. code:: shell

	visualdl --model model.pdmodel --port 8080


.. image:: https://user-images.githubusercontent.com/48054808/88621327-b664f280-d0d2-11ea-9e76-e3fcfeea4e57.png


更多具体细节可参考 `VisualDL使用指南 <https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md>`_