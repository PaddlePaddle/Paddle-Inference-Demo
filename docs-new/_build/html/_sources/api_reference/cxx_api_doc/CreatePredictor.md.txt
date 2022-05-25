# CreatePredictor 方法

API定义如下：

```c++
// 根据 Config 构建预测执行对象 Predictor
// 参数: config - 用于构建 Predictor 的配置信息
// 返回: std::shared_ptr<Predictor> - 预测对象的智能指针
std::shared_ptr<Predictor> CreatePredictor(const Config& config);
```

代码示例:

```c++
// 创建 Config
paddle_infer::Config config("../assets/models/mobilenet_v1");

// 根据 Config 创建 Predictor
auto predictor = paddle_infer::CreatePredictor(config);
```

# GetVersion 方法

API定义如下：

```c++
// 获取 Paddle 版本信息
// 参数: NONE
// 返回: std::string - Paddle 版本信息
std::string GetVersion();
```

代码示例:

```c++
// 获取 Paddle 版本信息
std::string paddle_version = paddle_infer::GetVersion();
```