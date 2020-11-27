# 仅供内部使用

API定义如下：

```python
# 转化为 NativeConfig，不推荐使用
# 参数：None
# 返回：当前 Config 对应的 NativeConfig
paddle.inference.Config.to_native_config()

# 设置是否使用Feed, Fetch OP，仅内部使用
# 当使用 ZeroCopyTensor 时，需设置为 false
# 参数：x - 是否使用Feed, Fetch OP，默认为 true
# 返回：None
paddle.inference.Config.switch_use_feed_fetch_ops(x: bool = True)

# 判断是否使用Feed, Fetch OP
# 参数：None
# 返回：bool - 是否使用Feed, Fetch OP
paddle.inference.Config.use_feed_fetch_ops_enabled()

# 设置是否需要指定输入 Tensor 的 Name，仅对内部 ZeroCopyTensor 有效
# 参数：x - 是否指定输入 Tensor 的 Name，默认为 true
# 返回：None
paddle.inference.Config.switch_specify_input_names(x: bool = True)

# 判断是否需要指定输入 Tensor 的 Name，仅对内部 ZeroCopyTensor 有效
# 参数：None
# 返回：bool - 是否需要指定输入 Tensor 的 Name
paddle.inference.Config.specify_input_name()
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 转化为 NativeConfig
native_config = config.to_native_config()

# 禁用 Feed, Fetch OP
config.switch_use_feed_fetch_ops(False)
# 返回是否使用 Feed, Fetch OP - false
print("switch_use_feed_fetch_ops is: {}".format(config.use_feed_fetch_ops_enabled()))

# 设置需要指定输入 Tensor 的 Name
config.switch_specify_input_names(True)
# 返回是否需要指定输入 Tensor 的 Name - true
print("specify_input_name is: {}".format(config.specify_input_name()))
```