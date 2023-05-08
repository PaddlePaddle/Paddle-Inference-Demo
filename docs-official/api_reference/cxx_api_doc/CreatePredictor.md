# CreatePredictor 方法

API定义如下：

```c++
// 根据 Config 构建预测执行对象 Predictor
// 参数: config - 用于构建 Predictor 的配置信息
// 返回: std::shared_ptr<Predictor> - 预测对象的智能指针
std::shared_ptr<Predictor> CreatePredictor(const Config& config);
```

代码示例：

```c++
// 创建 Config
paddle_infer::Config config("../assets/models/mobilenet_v1");

// 根据 Config 创建 Predictor
auto predictor = paddle_infer::CreatePredictor(config);
```

**注意事项：**
一个 `Config` 对象只能用于调用一次 `CreatePredictor` 生成一个 `Predictor`，需要通过 `CreatePredictor` 创建多个 `Predictor` 时请分别创建 `Config` 对象。

# GetVersion 方法

API定义如下：

```c++
// 获取 Paddle 版本信息
// 参数: NONE
// 返回: std::string - Paddle 版本信息
std::string GetVersion();
```

代码示例：

```c++
// 获取 Paddle 版本信息
std::string paddle_version = paddle_infer::GetVersion();
```

返回值实例：

```bash
version: 2.3.0
commit: b207edf916
branch: release/2.3
```

# ConvertToMixedPrecision 方法

`ConvertToMixedPrecision` 接口可对模型精度格式进行修改，如在选定的后端下，将 fp32 格式的模型转换为 fp16 格式，API 定义如下：

```c++
// 模型精度转换
// 参数：model_file - fp32 模型文件路径
//      params_file - fp32 权重文件路径
//      mixed_model_file - 混合精度模型文件保存路径
//      mixed_params_file - 混合精度权重文件保存路径
//      mixed_precision - 转换精度，如 PrecisionType.kHalf
//      backend - 后端，如 PlaceType.kGPU
//      keep_io_types - 保留输入输出精度信息，若为 True 则输入输出保留为 fp32 类型，否则转为 precision 类型
//      black_list - 黑名单列表，哪些 op 不需要进行精度类型转换
// 返回：NONE
void ConvertToMixedPrecision(
    const std::string& model_file,
    const std::string& params_file,
    const std::string& mixed_model_file,
    const std::string& mixed_params_file,
    PrecisionType mixed_precision,
    PlaceType backend,
    bool keep_io_types = true,
    std::unordered_set<std::string> black_list = {});
```

# SaveOptimizedModel 方法

`SaveOptimizedModel` 接口可保存经过选定 backend 硬件 analysis ir pass 优化的模型结构和模型参数，API 定义如下：

```c++
// 保存优化后的模型
// 参数：model_file - 原始模型文件路径
//      params_file - 原始权重文件路径
//      output_model_file - 优化后的模型文件保存路径
//      output_params_file - 优化后的权重文件保存路径
//      backend - 硬件后端，如 PlaceType.kXPU
//      black_list - 黑名单列表，哪些 pass 不需要执行
// 返回：NONE
void SaveOptimizedModel(
    const std::string& model_file,
    const std::string& params_file,
    const std::string& output_model_file,
    const std::string& output_params_file,
    PlaceType backend,
    std::unordered_set<std::string> black_list = {});
```
