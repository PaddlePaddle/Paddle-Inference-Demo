# 设置模型优化方法

**注意：** 关于自定义 IR 优化 Pass，请参考 [PaddlePassBuilder 类](../PaddlePassBuilder)

API定义如下：

```c
// 启用 IR 优化
// 参数：config - AnalysisConfig 对象指针
//      x - 是否开启 IR 优化，默认打开
// 返回：None
void PD_SwitchIrOptim(PD_AnalysisConfig* config, bool x);

// 判断是否开启 IR 优化 
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 是否开启 IR 优化
bool PD_IrOptim(const PD_AnalysisConfig* config);

// 设置是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件
// 参数：config - AnalysisConfig 对象指针
//      x - 是否打印 IR，默认关闭
// 返回：None
void PD_SwitchIrDebug(PD_AnalysisConfig* config, bool x);

// 返回 pass_builder，用来自定义图分析阶段选择的 IR
// 参数：config - AnalysisConfig 对象指针
//      pass_name - 需要删除的 pass 名称
// 返回：None
void PD_DeletePass(PD_AnalysisConfig* config, char* pass_name);
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置预测模型路径，这里为非 Combined 模型
const char* model_dir  = "./mobilenet_v1";
PD_SetModel(config, model_dir, NULL);

// 开启 IR 优化
PD_SwitchIrOptim(config, true);
// 开启 IR 打印
PD_SwitchIrDebug(config, true);

// 通过 API 获取 IR 优化是否开启 - true
printf("IR Optim is: %s\n", PD_IrOptim(config) ? "True" : "False");

// 通过 config 去除 fc_fuse_pass
char * pass_name = "fc_fuse_pass";
PD_DeletePass(config, pass_name);

// 根据 Config 创建 Predictor
PD_Predictor* predictor = PD_NewPredictor(config);
```

运行结果示例：

```bash
# PD_SwitchIrOptim 开启 IR 优化后，运行中会有如下 LOG 输出
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [attention_lstm_fuse_pass]
--- Running IR pass [seqconv_eltadd_relu_fuse_pass]
...
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]

# PD_SwitchIrDebug 开启 IR 打印后，运行结束之后会在目录下生成如下 DOT 文件
-rw-r--r-- 1 root root  70K Nov 17 10:47 0_ir_simplify_with_basic_ops_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 10_ir_fc_gru_fuse_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 11_ir_graph_viz_pass.dot
...
-rw-r--r-- 1 root root  72K Nov 17 10:47 8_ir_mul_lstm_fuse_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 9_ir_graph_viz_pass.dot
```
