
# PaddlePassBuilder 类

**注意：** PaddlePassBuilder 对象通过 `Config` 的 `pass_builder` 方法进行获取。其中存在2个成员对象 AnalysisPasses 和 Passes,AnalysisPasses 独立于 Passes 之外，仅 `AppendAnalysisPass` 和 `AnalysisPasses` 两个 API 能对其进行修改和读取，其余 API 的操作对象都仅限于Passes。

类及方法定义如下：

```c++
// 设置 IR 图分析阶段的 passes
// 参数：passes - IR 图分析阶段的 passes 的字符串列表
// 返回：None
void SetPasses(std::initializer_list<std::string> passes);

// 在 Passes 末尾添加 pass
// 参数：pass_type - 需要添加的 pass 字符串
// 返回：None
void AppendPass(const std::string &pass_type);

// 在 Passes 中的第 idx 位置插入 pass
// 参数：idx - 插入的 index 位置
//      pass_type - 需要插入的 pass 字符串
// 返回：None
void InsertPass(size_t idx, const std::string &pass_type);

// 删除第 idx 位置的 pass
// 参数：idx - 删除的 index 位置
// 返回：None
void DeletePass(size_t idx);

// 删除字符串匹配为 pass_type 的 pass
// 参数：pass_type - 需要删除的 pass 字符串
// 返回：None
void DeletePass(const std::string &pass_type);

// 清空所有 IR 优化中的 Passes
// 参数：None
// 返回：None
void ClearPasses();

// 启用Debug, 会在每一个 PASS 优化后生成当前计算图 DOT
// 即在每一个 fuse pass 之后添加一个 graph_viz_pass 进行 pass 可视化
// 参数：None
// 返回：None
void TurnOnDebug();

// 获取 IR 图分析阶段的 Passes 中的可读信息
// 参数：None
// 返回：std::string - 所有 Passes 的可读信息
std::string DebugString();

// 获取 IR 图分析阶段的所有 Passes
// 参数：None
// 返回：std::vector<std::string> - 所有 Passes 字符串列表
const std::vector<std::string> &AllPasses();

// 添加 Analysis Pass
// 参数：pass - 需要添加的 Analysis Pass 字符串表示
// 返回：None
void AppendAnalysisPass(const std::string &pass);

// 获取 IR 图分析阶段的所有 Analysis Passes
// 参数：None
// 返回：std::vector<std::string> - 所有 Analysis Passes 字符串列表
std::vector<std::string> AnalysisPasses() const;
```

自定义 IR Pass 代码示例：

```c++
// 构造 Config 对象
paddle_infer::Config config("./mobilenet.pdmodel", "./mobilenet.pdiparams");

// 开启 IR 优化
config.SwitchIrOptim();

// 得到 pass_builder 对象
auto pass_builder = config.pass_builder();

// 获取 pass_builder 中的所有 Passes
const std::vector<std::string> all_passes = pass_builder->AllPasses();

// all_passes 中返回结果如下:
// simplify_with_basic_ops_pass
// attention_lstm_fuse_pass
// ...
// runtime_context_cache_pass

// 清空所有 Passes
pass_builder->ClearPasses();
// 设置 Passes
pass_builder->SetPasses({"attention_lstm_fuse_pass", "fc_gru_fuse_pass"});
// 在末尾处添加 pass
pass_builder->AppendPass("fc_fuse_pass");
// 删除 Passes
pass_builder->DeletePass("fc_fuse_pass");
// 在 idx = 0 的位置添加 pass
pass_builder->InsertPass(0, "fc_fuse_pass");
// 删除 idx = 0 所在位置的 pass
pass_builder->DeletePass(0);
// 启用Debug, 会在每一个 PASS 优化后生成当前计算图 DOT
// 即在每一个 pass 之后添加一个 graph_viz_pass
pass_builder->TurnOnDebug();
// 获取 IR 图分析阶段的 Passes 中的可读信息
std::cout << pass_builder->DebugString() << std::endl;

// 运行以上代码得到的输出结果如下：
//  - attention_lstm_fuse_pass
//  - graph_viz_pass
//  - fc_gru_fuse_pass
//  - graph_viz_pass
```

对 Analysis Pass 进行操作和读取示例：

```c++
// 构造 Config 对象
paddle_infer::Config config("./mobilenet.pdmodel", "./mobilenet.pdiparams");

// 开启 IR 优化
config.SwitchIrOptim();

// 得到 pass_builder 对象
auto pass_builder = config.pass_builder();

// 添加 analysis pass
pass_builder->AppendAnalysisPass("ir_analysis_pass");

// 获取 pass_builder 中的所有 Analysis Passes
const std::vector<std::string> analysis_passes = pass_builder->AnalysisPasses();

// analysis_passes 中返回结果如下:
// ir_graph_build_pass
// ir_graph_clean_pass
// ...
// ir_graph_to_program_pass
```