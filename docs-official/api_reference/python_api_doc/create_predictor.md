# create_predictor 方法

API定义如下：

```python
# 根据 Config 构建预测执行器 Predictor
# 参数: config - 用于构建 Predictor 的配置信息
# 返回: Predictor - 预测执行器
paddle.inference.create_predictor(config: Config)
```

代码示例:

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)
```

# get_version 方法

API定义如下：

```python
# 获取 Paddle 版本信息
# 参数: NONE
# 返回: str - Paddle 版本信息
paddle.inference.get_version()
```

代码示例:

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 获取 Paddle 版本信息
paddle_infer.get_version()

# 获得输出如下:
# version: 2.0.0-rc0
# commit: 97227e6
# branch: HEAD
```