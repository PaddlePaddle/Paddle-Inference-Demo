# Config 构造函数

`Config` 类为用于配置构建 `Predictor` 对象的配置信息，如模型路径、是否开启gpu等等。

构造函数定义如下：

```c++
// 创建 Config 对象，默认构造函数
Config();

// 创建 Config 对象，输入为其他 Config 对象
Config(const Config& other);

// 创建 Config 对象，输入分别为模型文件路径和参数文件路径
Config(const std::string& prog_file, const std::string& params_file);
```

代码示例：

```c++
// 字符串 prog_file 为 Combine 模型文件所在路径
std::string prog_file = "../assets/models/mobilenet_v1.pdmodel";
// 字符串 params_file 为 Combine 模型参数文件所在路径
std::string params_file = "../assets/models/mobilenet_v1.pdiparams";

// 根据模型文件和参数文件构造 Config 对象
paddle_infer::Config config(prog_file, params_file);

// 根据 Config 对象创建预测器对象
auto predictor = paddle_infer::CreatePredictor(config);
```

**注意事项：**
一个 `Config` 对象只能用于调用一次 `CreatePredictor` 生成一个 `Predictor`，需要通过 `CreatePredictor` 创建多个 `Predictor` 时请分别创建 `Config` 对象。
