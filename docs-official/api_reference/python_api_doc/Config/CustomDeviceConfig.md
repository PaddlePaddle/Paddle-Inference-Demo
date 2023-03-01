
# 使用 CustomDevice 进行预测

API定义如下：

```python
# 启用CustomDevice
# 参数 ： device_type - 设备类型
# 返回 : None
paddle.inference.Config.enable_custom_device(device_type : str, device_id : int)

# 启用CustomDevice混合精度
# 参数 ： precision - 指定推理精度，默认是PrecisionType.Float32
# 返回 : None
paddle.inference.Config.enable_custom_device_mixed(precision : PrecisionType)

```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 启动CustomDevice Half精度
config.enable_custom_device("OpenCL", 0)
config.enable_custom_device_mixed(PrecisionType.Half)
```
