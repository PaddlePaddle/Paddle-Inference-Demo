
# 使用 Lite OpenCL 进行预测

API定义如下：

```python
# 启用Lite模式的OpenCL加速
# 返回 : None
paddle.inference.Config.enable_opencl()

# 查询Lite模式的OpenCL加速是否开启
# 返回 : OpenCL是否开启
paddle.inference.Config.use_opencl()
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 必须先启用Lite模式，再启用OpenCL模式
config.enable_lite_engine()
config.enable_opencl()
```
