#  PredictorPool 类

`PredictorPool` 对 `Predictor` 进行了简单的封装，通过传入config和thread的数目来完成初始化，在每个线程中，根据自己的线程id直接从池中取出对应的 `Predictor` 来完成预测过程。

类及方法定义如下：

```python
# PredictorPool 类定义
# 参数：config - Config 类型
#      size - Predictor 对象数量
class paddle.inference.PredictorPool(config: Config, size: int)

# 根据线程 ID 取出该线程对应的 Predictor
# 参数：idx - 线程 ID
# 返回：Predictor - 线程 ID 对应的 Predictor
paddle.inference.PredictorPool.retrive(idx: int)
```

代码示例

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 Config
config = paddle_infer.Config("./mobilenet_v1")

# 创建 PredictorPool
pred_pool = paddle_infer.PredictorPool(config, 4)

# 获取 ID 为 2 的 Predictor 对象
predictor = pred_pool.retrive(2)
```