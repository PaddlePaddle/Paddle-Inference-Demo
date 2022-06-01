#  PredictorPool 类

`PredictorPool` 对 `Predictor` 进行了简单的封装，通过传入 config 和 thread 的数目来完成初始化，在每个线程中，根据自己的线程 id 直接从池中取出对应的 `Predictor` 来完成预测过程。

构造函数和 API 定义如下：

```c++
// PredictorPool 构造函数
// 参数：config - Config 对象
//      size - Predictor 对象数量
PredictorPool(const Config& config, size_t size = 1);

// 根据线程 ID 取出该线程对应的 Predictor
// 参数：idx - 线程 ID
// 返回：Predictor* - 线程 ID 对应的 Predictor 指针
Predictor* Retrive(size_t idx);
```

代码示例

```c++
// 构造 Config 对象
paddle_infer::Config config("./resnet.pdmodel", "./resnet.pdiparams");
// 启用 GPU 预测
config.EnableUseGpu(100, 0);

// 根据 Config 对象创建 PredictorPool
paddle_infer::PredictorPool pred_pool(config, 4);

// 获取 ID 为 2 的 Predictor 对象
auto pred = pred_pool.Retrive(2);
```