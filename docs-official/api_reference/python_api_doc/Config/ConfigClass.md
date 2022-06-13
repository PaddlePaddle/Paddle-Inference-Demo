# Config 类定义

`Config` 类为用于配置构建 `Predictor` 对象的配置信息，如模型路径、是否开启 gpu 等等。

构造函数定义如下：

```python
# Config 类定义，输入为 None
class paddle.inference.Config()

# Config 类定义，输入为其他 Config 对象
class paddle.inference.Config(config: Config)

# Config 类定义，输入分别为模型文件路径和参数文件路径
class paddle.inference.Config(prog_file: str, params_file: str)
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet.pdmodel", "./mobilenet.pdiparams")

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)
```
