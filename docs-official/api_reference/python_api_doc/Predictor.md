# Predictor 类

Paddle Inference的预测器，由 `create_predictor` 根据 `Config` 进行创建。用户可以根据 Predictor 提供的接口设置输入数据、执行模型预测、获取输出等。

类及方法定义如下：

```python
# Predictor 类定义
class paddle.inference.Predictor

# 获取所有输入 paddle.inference.Tensor 的名称
# 参数：None
# 返回：List[str] - 所有输入 paddle.inference.Tensor 的名称
paddle.inference.Predictor.get_input_names()

# 根据名称获取输入 paddle.inference.Tensor 的句柄
# 参数：name - paddle.inference.Tensor 的名称
# 返回：Tensor - 输入 paddle.inference.Tensor
paddle.inference.Predictor.get_input_handle(name: str)

# 获取所有输出 paddle.inference.Tensor 的名称
# 参数：None
# 返回：List[str] - 所有输出 Tensor 的名称
paddle.inference.Predictor.get_output_names()

# 根据名称获取输出 paddle.inference.Tensor 的句柄
# 参数：name - paddle.inference.Tensor 的名称
# 返回：paddle.inference.Tensor - 输出 paddle.inference.Tensor
paddle.inference.Predictor.get_output_handle(name: str)

# 执行模型预测，需要在设置输入数据后调用
# 参数：None
# 返回：None
# 备注：此接口对应于 paddle.inference.Tensor
paddle.inference.Predictor.run()

# 执行模型预测（推荐使用）
# 参数：List[paddle.Tensor] - 输入数据，对应模型输入的 paddle.Tensor 列表
# 返回：List[paddle.Tensor] - 输出数据，对应模型输出的 paddle.Tenosr 列表
# 备注：此接口对应于 paddle.Tensor
paddle.inference.Predictor.run(inputs: List[paddle.Tensor])

# 根据该 Predictor，克隆一个新的 Predictor，两个 Predictor 之间共享权重
# 参数：None
# 返回：Predictor - 新的 Predictor
paddle.inference.Predictor.clone()

# 释放中间 Tensor
# 参数：None
# 返回：None
paddle.inference.Predictor.clear_intermediate_tensor()

# 获取中间 op 的输出 paddle.inference.Tensor
# 参数：Exp_OutputHookFunc  -  具有三个接收参数的 hook 函数，第一个参数是 op type（name）
#                                                       第二个参数是输出 paddle.inference.Tensor's name
#                                                       第三个参数是输出 paddle.inference.Tensor
#                             (function) hook_function: (op_type : str, tensor_name : str, tensor : paddle.inference.Tensor) -> None
# 返回：None
paddle.inference.Predictor.register_output_hook(hookfunc : Exp_OutputHookFunc)

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
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)

# 定义 hook function
# 打印中间层的 op type, tensor's name, tensor's shape
def hookfunc(op_type, tensor_name, tensor):
    print(op_type)
    print(tensor_name)
    print(tensor.shape())
# 注册 hook function
# 通过该接口注册的 hook 函数，在每个 op run 完都会被执行一次
predictor.register_output_hook(hookfunc)

# 获取输入 paddle.inference.Tensor
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])

# 从 CPU 获取数据，设置到 paddle.inference.Tensor 内部
fake_input = numpy.random.randn(1, 3, 224, 224).astype("float32")
input_tensor.copy_from_cpu(fake_input)

# 执行预测
predictor.run()

# 获取输出 paddle.inference.Tensor
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])

# 释放中间 paddle.inference.Tensor
predictor.clear_intermediate_tensor()

# 释放内存池中的所有临时 Tensor
predictor.try_shrink_memory()
```
