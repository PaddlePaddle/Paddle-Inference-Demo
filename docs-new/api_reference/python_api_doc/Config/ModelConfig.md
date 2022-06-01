# 设置预测模型

## 从文件中加载预测模型

API定义如下：

```python
# 设置模型文件路径，当需要从磁盘加载模型时使用
# 参数：prog_file_path - 模型文件路径
#      params_file_path - 参数文件路径
# 返回：None
paddle.inference.Config.set_model(prog_file_path: str, params_file_path: str)

# 设置模型文件路径
# 参数：x - 模型文件路径
# 返回：None
paddle.inference.Config.set_prog_file(x: str)

# 设置参数文件路径
# 参数：x - 参数文件路径
# 返回：None
paddle.inference.Config.set_params_file(x: str)

# 获取模型文件路径
# 参数：None
# 返回：str - 模型文件路径
paddle.inference.Config.prog_file()

# 获取参数文件路径
# 参数：None
# 返回：str - 参数文件路径
paddle.inference.Config.params_file()
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config()

# 通过 API 设置模型文件夹路径
config.set_prog_file("./mobilenet_v2.pdmodel")
config.set_params_file("./mobilenet_v2.pdiparams")

# 通过 API 获取 config 中的模型文件和参数文件路径
print(config.prog_file())
print(config.params_file())

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)
```

## 从内存中加载预测模型

API定义如下：

```python
# 从内存加载模型
# 参数：prog_buffer - 内存中模型结构数据
#      prog_buffer_size - 内存中模型结构数据的大小
#      params_buffer - 内存中模型参数数据
#      params_buffer_size - 内存中模型参数数据的大小
# 返回：None
paddle.inference.Config.set_model_buffer(prog_buffer: str, prog_buffer_size: int, 
                                         params_buffer: str, params_buffer_size: int)

# 判断是否从内存中加载模型
# 参数：None
# 返回：bool - 是否从内存中加载模型
paddle.inference.Config.model_from_memory()
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config()

# 加载模型文件到内存
with open('./mobilenet_v2.pdmodel', 'rb') as prog_file:
    prog_data=prog_file.read()
    
with open('./mobilenet_v2.pdiparams', 'rb') as params_file:
    params_data=params_file.read()

# 从内存中加载模型
config.set_model_buffer(prog_data, len(prog_data), params_data, len(params_data))

# 通过 API 获取 config 中 model_from_memory 的值 - True
print(config.model_from_memory())

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)
```
