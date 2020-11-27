
# 使用 XPU 进行预测

API定义如下：

```python
# 启用 XPU 进行预测
# 参数：l3_workspace_size - l3 cache 分配的显存大小
# 返回：None
paddle.inference.Config.enable_xpu(l3_workspace_size: int = 0xfffc00)
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 启用 XPU，并设置 l3 cache 大小为 100M
config.enable_xpu(100)
```
