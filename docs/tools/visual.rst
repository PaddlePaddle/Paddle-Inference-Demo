模型可视化
==============

通过 `Quick Start <../introduction/quick_start.html>`_ 一节中，我们了解到，预测模型包含了两个文件，一部分为模型结构文件，通常以 **model** , **__model__** 文件存在；另一部分为参数文件，通常以params 文件或一堆分散的文件存在。

模型结构文件，顾名思义，存储了模型的拓扑结构，其中包括模型中各种OP的计算顺序以及OP的详细信息。很多时候，我们希望能够将这些模型的结构以及内部信息可视化，方便我们进行模型分析。接下来将会通过两种方式来讲述如何对Paddle 预测模型进行可视化。

一： 通过 Netron 可视化
------------------

1） 安装

Netron 是github开源的软件，我们可以进入它的 `主页 <https://github.com/lutzroeder/netron>`_ 进行下载安装。

2）可视化

`点击 <https://paddle-inference-dist.bj.bcebos.com/temp_data/sample_model/__model__>`_ 下载测试模型。

打开Netron软件， 我们将Paddle 预测模型结构文件命名成 `__model__` ， 然后将文件通过鼠标拖入到Netron软件中完成可视化。


.. image:: https://user-images.githubusercontent.com/5595332/81791426-4ca86400-9539-11ea-8776-e859d000f7f6.png

二： 通过代码方式生成dot文件
---------------------

1）pip 安装Paddle

2）生成dot文件

`点击 <https://paddle-inference-dist.bj.bcebos.com/temp_data/sample_model/__model__>`_ 下载测试模型。

.. code:: python

	#!/usr/bin/env python
	import paddle.fluid as fluid
	from paddle.fluid import core
	from paddle.fluid.framework import IrGraph

	def get_graph(program_path):
		with open(program_path, 'rb') as f:
			binary_str = f.read()
		program =   fluid.framework.Program.parse_from_string(binary_str)
		return IrGraph(core.Graph(program.desc), for_test=True)

	if __name__ == '__main__':
		program_path = './lecture_model/__model__' 
		offline_graph = get_graph(program_path)
		offline_graph.draw('.', 'test_model', [])


3）生成svg

**Note：需要环境中安装graphviz**

.. code:: python

	dot -Tsvg ./test_mode.dot -o test_model.svg
	

然后将test_model.svg以浏览器打开预览即可。

.. image::  https://user-images.githubusercontent.com/5595332/81796500-19b59e80-9540-11ea-8c70-31122e969683.png
