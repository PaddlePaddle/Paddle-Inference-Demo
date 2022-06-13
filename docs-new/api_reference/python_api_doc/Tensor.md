#  Tensor 类

Tensor 是 Paddle Inference 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。

**注意：** 应使用 `Predictor` 的 `get_input_handle` 和 `get_output_handle` 接口获取输入输出 `Tensor`。

类及方法定义如下：

```python
# Tensor 类定义
class paddle.inference.Tensor

# 设置 Tensor 的维度信息
# 参数：shape - 维度信息
# 返回：None
paddle.inference.Tensor.reshape(shape: numpy.ndarray|List[int])

# 从 CPU 获取数据，设置到 Tensor 内部
# 参数：data - CPU 数据 - 支持 float, int32, int64
# 返回：None
paddle.inference.Tensor.copy_from_cpu(data: numpy.ndarray)

# 从 Tensor 中获取数据到 CPU，该接口内含同步等待 GPU 运行结束，当 Predictor 
#    运行在 GPU 硬件时，在 CPU 线程下对该 API 调用进行计时是不准确的
# 参数：None
# 返回：numpy.ndarray - CPU 数据
paddle.inference.Tensor.copy_to_cpu()

# 获取 Tensor 的维度信息
# 参数：None
# 返回：List[int] - Tensor 的维度信息
paddle.inference.Tensor.shape()

# 设置 Tensor 的 LoD 信息
# 参数：x - Tensor 的 LoD 信息
# 返回：None
paddle.inference.Tensor.set_lod(x: numpy.ndarray|List[List[int]])

# 获取 Tensor 的 LoD 信息
# 参数：None
# 返回：List[List[int]] - Tensor 的 LoD 信息
paddle.inference.Tensor.lod()

# 获取 Tensor 的数据类型
# 参数：None
# 返回：DataType - Tensor 的数据类型
paddle.inference.Tensor.type()
```

代码示例：

```python
import numpy

# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)

# 准备输入数据
fake_input = numpy.random.randn(1, 3, 224, 224).astype("float32")

# 获取输入 Tensor
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])

# 设置 Tensor 的维度信息
input_tensor.reshape([1, 3, 224, 224])

# 从 CPU 获取数据，设置到 Tensor 内部
input_tensor.copy_from_cpu(fake_input)

# 执行预测
predictor.run()

# 获取输出 Tensor
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])

# 从 Tensor 中获取数据到 CPU
output_data = output_tensor.copy_to_cpu()

# 获取 Tensor 的维度信息
output_shape = output_tensor.shape()

# 获取 Tensor 的数据类型
output_type = output_tensor.type()
```