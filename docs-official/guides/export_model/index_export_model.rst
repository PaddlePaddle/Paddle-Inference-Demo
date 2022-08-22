
#######
导出模型
#######

Paddle Inference支持使用飞桨静态图模型进行推理，您可以通过以下两种方式获取静态图模型：

**（1）飞桨框架导出推理模型**  

飞桨框架在训练模型过程中，会在本地存储最终训练产出的模型结构和权重参数，这个步骤中存储的模型文件包含了模型的前向、反向以及优化器等信息（即常说的动态图模型，模型参数文件名为*.pdparams和*.pdopt）。
而在进行模型部署时，我们只需要模型的前向结构，以及前向的权重参数，并且会针对网络结构做部署优化（如算子融合等），以保证部署性能更优，因此在模型部署阶段，需要进行模型导出（即常说的静态图模型，模型参数文件名为*.pdmodel和*.pdiparams）。 您可以参考此篇文档导出用于推理的飞桨模型：

- `飞桨框架模型导出 <./paddle_model_export.html>`_


**（2）导入其他框架模型（X2Paddle）**  

通过X2Paddle工具，目前支持将Pytorch、ONNX、TensorFlow、Caffe的模型转换成飞桨静态图模型结构，具体使用方法请参考以下文档：

- `将Pytorch、TensorFlow、ONNX等框架转换成飞桨模型 <./outside_model_export.rst>`_


**（可选）模型结构可视化**

在得到用于Paddle Inference推理的 **飞桨静态图模型** 后，推荐您使用 VisualDL 或其他类似工具对您的模型进行查看，方便您后续的推理应用开发。 您可以参考以下文档可视化您的模型：

- `模型结构可视化 <./visual_model.rst>`_

..  toctree::
    :hidden:
    
    paddle_model_export.md
    outside_model_export.rst
    visual_model.rst