# Predictor 类

Paddle Inference的预测器，由 `create_predictor` 根据 `Config` 进行创建。用户可以根据Predictor提供的接口设置输入数据、执行模型预测、获取输出等。

类及方法定义如下：

```python
# Predictor 类定义
class paddle.inference.Predictor

# 获取所有输入 Tensor 的名称
# 参数：None
# 返回：List[str] - 所有输入 Tensor 的名称
paddle.inference.Predictor.get_input_names()

# 根据名称获取输入 Tensor 的句柄
# 参数：name - Tensor 的名称
# 返回：Tensor - 输入 Tensor
paddle.inference.Predictor.get_input_handle(name: str)

# 获取所有输出 Tensor 的名称
# 参数：None
# 返回：List[str] - 所有输出 Tensor 的名称
paddle.inference.Predictor.get_output_names()

# 根据名称获取输出 Tensor 的句柄
# 参数：name - Tensor 的名称
# 返回：Tensor - 输出 Tensor
paddle.inference.Predictor.get_output_handle(name: str)

# 执行模型预测，需要在设置输入数据后调用
# 参数：None
# 返回：None
paddle.inference.Predictor.run()

# 根据该 Predictor，克隆一个新的 Predictor，两个 Predictor 之间共享权重
# 参数：None
# 返回：Predictor - 新的 Predictor
paddle.inference.Predictor.clone()

# 根据该 Predictor，和外部 stream, 克隆一个新的 Predictor，两个 Predictor 之间共享权重，克隆 Predictor 绑定外部 stream
# 参数：paddle.device.cuda.Stream
# 返回：Predictor - 新的 Predictor
paddle.inference.Predictor.clone(paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1))

# 释放中间 Tensor
# 参数：None
# 返回：None
paddle.inference.Predictor.clear_intermediate_tensor()

# 释放内存池中的所有临时 Tensor
# 参数：None
# 返回：int - 释放的内存字节数
paddle.inference.Predictor.try_shrink_memory()
```

代码示例

```python
import numpy

# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)

# 获取输入 Tensor
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])

# 从 CPU 获取数据，设置到 Tensor 内部
fake_input = numpy.random.randn(1, 3, 224, 224).astype("float32")
input_tensor.copy_from_cpu(fake_input)

# 执行预测
predictor.run()

# 获取输出 Tensor
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])

# 释放中间Tensor
predictor.clear_intermediate_tensor()

# 释放内存池中的所有临时 Tensor
predictor.try_shrink_memory()
```
