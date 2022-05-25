# 设置预测模型

## 从文件中加载预测模型 - 非Combined模型 

API定义如下：

```c++
// 设置模型文件路径，当需要从磁盘加载非 Combined 模型时使用
// 参数：model_dir - 模型文件夹路径
// 返回：None
void SetModel(const std::string& model_dir);

// 获取非combine模型的文件夹路径
// 参数：None
// 返回：string - 模型文件夹路径
const std::string& model_dir();
```

代码示例：

```c++
// 字符串 model_dir 为非 Combined 模型文件夹路径
std::string model_dir = "../assets/models/mobilenet_v1";

// 创建默认 Config 对象
paddle_infer::Config config();

// 通过 API 设置模型文件夹路径
config.SetModel(model_dir);

// 通过 API 获取 config 中的模型路径
std::cout << "Model Path is: " << config.model_dir() << std::endl;

// 根据Config对象创建预测器对象
auto predictor = paddle_infer::CreatePredictor(config);
```

## 从文件中加载预测模型 -  Combined 模型

API定义如下：

```c++
// 设置模型文件路径，当需要从磁盘加载 Combined 模型时使用
// 参数：prog_file_path - 模型文件路径
//      params_file_path - 参数文件路径
// 返回：None
void SetModel(const std::string& prog_file_path,
              const std::string& params_file_path);

// 设置模型文件路径，当需要从磁盘加载 Combined 模型时使用。
// 参数：x - 模型文件路径
// 返回：None
void SetProgFile(const std::string& x);


// 设置参数文件路径，当需要从磁盘加载 Combined 模型时使用
// 参数：x - 参数文件路径
// 返回：None
void SetParamsFile(const std::string& x);

// 获取 Combined 模型的模型文件路径
// 参数：None
// 返回：string - 模型文件路径
const std::string& prog_file();

// 获取 Combined 模型的参数文件路径
// 参数：None
// 返回：string - 参数文件路径
const std::string& params_file();
```

代码示例：

```c++
// 字符串 prog_file 为 Combined 模型的模型文件所在路径
std::string prog_file = "../assets/models/mobilenet_v1/__model__";
// 字符串 params_file 为 Combined 模型的参数文件所在路径
std::string params_file = "../assets/models/mobilenet_v1/__params__";

// 创建默认 Config 对象
paddle_infer::Config config();
// 通过 API 设置模型文件夹路径，
config.SetModel(prog_file, params_file);
// 注意：SetModel API与以下2行代码等同
// config.SetProgFile(prog_file);
// config.SetParamsFile(params_file);

// 通过 API 获取 config 中的模型文件和参数文件路径
std::cout << "Model file path is: " << config.prog_file() << std::endl;
std::cout << "Model param path is: " << config.params_file() << std::endl;

// 根据 Config 对象创建预测器对象
auto predictor = paddle_infer::CreatePredictor(config);
```

## 从内存中加载预测模型

API定义如下：

```c++
// 从内存加载模型
// 参数：prog_buffer - 内存中模型结构数据
//      prog_buffer_size - 内存中模型结构数据的大小
//      params_buffer - 内存中模型参数数据
//      params_buffer_size - 内存中模型参数数据的大小
// 返回：None
void SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size,
                    const char* params_buffer, size_t params_buffer_size);

// 判断是否从内存中加载模型
// 参数：None
// 返回：bool - 是否从内存中加载模型
bool model_from_memory() const;
```

代码示例：

```c++
// 定义文件读取函数
std::string read_file(std::string filename) {
  std::ifstream file(filename);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

// 设置模型文件和参数文件所在路径
std::string prog_file = "../assets/models/mobilenet_v1/__model__";
std::string params_file = "../assets/models/mobilenet_v1/__params__";

// 加载模型文件到内存
std::string prog_str = read_file(prog_file);
std::string params_str = read_file(params_file);

// 创建默认 Config 对象
paddle_infer::Config config();
// 从内存中加载模型
config.SetModelBuffer(prog_str.c_str(), prog_str.size(),
                      params_str.c_str(), params_str.size());

// 通过 API 获取 config 中 model_from_memory 的值
std::cout << "Load model from memory is: " << config.model_from_memory() << std::endl;

// 根据 Confi 对象创建预测器对象
auto predictor = paddle_infer::CreatePredictor(config);
```
