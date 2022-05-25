
# 使用 ONNXRuntime 进行预测

API定义如下：

```python
# 启用 ONNXRuntime 进行预测
# 参数：None
# 返回：None
paddle.inference.Config.enable_onnxruntime()

# 禁用 ONNXRuntime 进行预测
# 参数：None
# 返回：None
paddle.inference.Config.disable_onnxruntime()

# 判断是否启用 ONNXRuntime 
# 参数：None
# 返回：bool - 是否启用 ONNXRuntime 
paddle.inference.Config.onnxruntime_enabled()

# 启用 ONNXRuntime 预测时开启优化
# 参数：None
# 返回：None
paddle.inference.Config.enable_ort_optimization()
```

ONNXRuntime设置代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./model.pdmodel", "./model.pdiparams")

# 启用 ONNXRuntime 进行预测
config.enable_onnxruntime()
# 通过 API 获取 ONNXRuntime 信息
print("Use ONNXRuntime is: {}".format(config.onnxruntime_enabled())) # True

# 开启 ONNXRuntime 优化
config.enable_ort_optimization()

# 设置 ONNXRuntime 算子计算线程数为 10
config.set_cpu_math_library_num_threads(10)

# 禁用 ONNXRuntime 进行预测
config.disable_onnxruntime()

# 通过 API 获取 ONNXRuntime 信息
print("Use ONNXRuntime is: {}".format(config.onnxruntime_enabled())) # False
```
