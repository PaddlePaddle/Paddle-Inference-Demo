# 仅供内部使用

API定义如下：

```c++
// 转化为 NativeConfig，不推荐使用
// 参数：None
// 返回：当前 Config 对应的 NativeConfig
NativeConfig ToNativeConfig() const;

// 设置是否使用Feed, Fetch OP，仅内部使用
// 当使用 ZeroCopyTensor 时，需设置为 false
// 参数：x - 是否使用Feed, Fetch OP，默认为 true
// 返回：None
void SwitchUseFeedFetchOps(int x = true);

// 判断是否使用Feed, Fetch OP
// 参数：None
// 返回：bool - 是否使用Feed, Fetch OP
bool use_feed_fetch_ops_enabled() const;

// 设置是否需要指定输入 Tensor 的 Name，仅对内部 ZeroCopyTensor 有效
// 参数：x - 是否指定输入 Tensor 的 Name，默认为 true
// 返回：None
void SwitchSpecifyInputNames(bool x = true);

// 判断是否需要指定输入 Tensor 的 Name，仅对内部 ZeroCopyTensor 有效
// 参数：None
// 返回：bool - 是否需要指定输入 Tensor 的 Name
bool specify_input_name() const;

// 设置 Config 为无效状态，仅内部使用，保证每一个 Config 仅用来初始化一次 Predictor
// 参数：None
// 返回：None
void SetInValid();

// 判断当前 Config 是否有效
// 参数：None
// 返回：bool - 当前 Config 是否有效
bool is_valid() const;
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_infer_model + "/mobilenet");

// 转化为 NativeConfig
auto native_config = analysis_config->ToNativeConfig();

// 禁用 Feed, Fetch OP
config.SwitchUseFeedFetchOps(false);
// 返回是否使用 Feed, Fetch OP - false
std::cout << "UseFeedFetchOps is: " << config.use_feed_fetch_ops_enabled() << std::endl;

// 设置需要指定输入 Tensor 的 Name
config.SwitchSpecifyInputNames(true);
// 返回是否需要指定输入 Tensor 的 Name - true
std::cout << "Specify Input Name is: " << config.specify_input_name() << std::endl;

// 设置 Config 为无效状态
config.SetInValid();
// 判断当前 Config 是否有效 - false
std::cout << "Config validation is: " << config.is_valid() << std::endl;
```