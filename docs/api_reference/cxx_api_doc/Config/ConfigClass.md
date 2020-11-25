# Config 构造函数

`Config` 类为用于配置构建 `Predictor` 对象的配置信息，如模型路径、是否开启gpu等等。

构造函数定义如下：

```c++
// 创建 Config 对象，默认构造函数
Config();

// 创建 Config 对象，输入为其他 Config 对象
Config(const Config& other);

// 创建 Config 对象，输入为非 Combine 模型的文件夹路径
Config(const std::string& model_dir);

// 创建 Config 对象，输入分别为 Combine 模型的模型文件路径和参数文件路径
Config(const std::string& prog_file, const std::string& params_file);
```

代码示例 (1)：默认构造函数，通过API加载预测模型 - 非Combined模型

```c++
// 字符串 model_dir 为非 Combine 模型文件夹路径
std::string model_dir = "../assets/models/mobilenet_v1";

// 创建默认 Config 对象
paddle_infer::Config config();

// 通过 API 设置模型文件夹路径
config.SetModel(model_dir);

// 根据 Config 对象创建预测器对象
auto predictor = paddle_infer::CreatePredictor(config);
```

代码示例 (2)：通过构造函数加载预测模型 - 非Combined模型

```c++
// 字符串 model_dir 为非 Combine 模型文件夹路径
std::string model_dir = "../assets/models/mobilenet_v1";

// 根据非 Combine 模型的文件夹路径构造 Config 对象
paddle_infer::Config config(model_dir);

// 根据 Config 对象创建预测器对象
auto predictor = paddle_infer::CreatePredictor(config);
```

代码示例 (3)：通过构造函数加载预测模型 - Combined模型

```c++
// 字符串 prog_file 为 Combine 模型文件所在路径
std::string prog_file = "../assets/models/mobilenet_v1/__model__";
// 字符串 params_file 为 Combine 模型参数文件所在路径
std::string params_file = "../assets/models/mobilenet_v1/__params__";

// 根据 Combine 模型的模型文件和参数文件构造 Config 对象
paddle_infer::Config config(prog_file, params_file);

// 根据 Config 对象创建预测器对象
auto predictor = paddle_infer::CreatePredictor(config);
```
