# 设置模型优化方法

## IR 优化

**注意：** 关于自定义 IR 优化 Pass，请参考 [PaddlePassBuilder 类](../PaddlePassBuilder)

API定义如下：

```c++
// 启用 IR 优化
// 参数：x - 是否开启 IR 优化，默认打开
// 返回：None
void SwitchIrOptim(int x = true);

// 判断是否开启 IR 优化 
// 参数：None
// 返回：bool - 是否开启 IR 优化
bool ir_optim() const;

// 设置是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件
// 参数：x - 是否打印 IR，默认关闭
// 返回：None
void SwitchIrDebug(int x = true);

// 返回 pass_builder，用来自定义图分析阶段选择的 IR
// 参数：None
// 返回：PassStrategy - pass_builder对象
PassStrategy* pass_builder() const;
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_model_dir);

// 开启 IR 优化
config.SwitchIrOptim();
// 开启 IR 打印
config.SwitchIrDebug();

// 得到 pass_builder 对象
auto pass_builder = config.pass_builder();
// 在 IR 优化阶段，去除 fc_fuse_pass
pass_builder->DeletePass("fc_fuse_pass");

// 通过 API 获取 IR 优化是否开启 - true
std::cout << "IR Optim is: " << config.ir_optim() << std::endl;

// 根据Config对象创建预测器对象
auto predictor = paddle_infer::CreatePredictor(config);
```

运行结果示例：

```bash
# SwitchIrOptim 开启 IR 优化后，运行中会有如下 LOG 输出
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [attention_lstm_fuse_pass]
--- Running IR pass [seqconv_eltadd_relu_fuse_pass]
...
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]

# SwitchIrDebug 开启 IR 打印后，运行结束之后会在目录下生成如下 DOT 文件
-rw-r--r-- 1 root root  70K Nov 17 10:47 0_ir_simplify_with_basic_ops_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 10_ir_fc_gru_fuse_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 11_ir_graph_viz_pass.dot
...
-rw-r--r-- 1 root root  72K Nov 17 10:47 8_ir_mul_lstm_fuse_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 9_ir_graph_viz_pass.dot
```

## Lite 子图

```c++ 
// 启用 Lite 子图
// 参数：precision_mode - Lite 子图的运行精度，默认为 FP32
//      zero_copy      - 启用 zero_copy，lite 子图与 paddle inference 之间共享数据
//      Passes_filter  - 设置 lite 子图的 pass
//      ops_filter     - 设置不使用 lite 子图运行的 op
// 返回：None
void EnableLiteEngine(
      AnalysisConfig::Precision precision_mode = Precision::kFloat32,
      bool zero_copy = false,
      const std::vector<std::string>& passes_filter = {},
      const std::vector<std::string>& ops_filter = {});


// 判断是否启用 Lite 子图
// 参数：None
// 返回：bool - 是否启用 Lite 子图
bool lite_engine_enabled() const;
```

示例代码：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_model_dir);

config.EnableUseGpu(100, 0);
config.EnableLiteEngine(paddle_infer::PrecisionType::kFloat32);

// 通过 API 获取 Lite 子图启用信息 - true
std::cout << "Lite Engine is: " << config.lite_engine_enabled() << std::endl;
```