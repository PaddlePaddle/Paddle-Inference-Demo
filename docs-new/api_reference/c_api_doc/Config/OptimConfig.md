# 设置模型优化方法
## IR 优化

API定义如下：

```c
// 启用 IR 优化
// 参数：pd_config - Config 对象指针
//      x         - 是否开启 IR 优化，默认打开
// 返回：None
void PD_ConfigSwitchIrOptim(PD_Config* pd_config, PD_Bool x);

// 判断是否开启 IR 优化 
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否开启 IR 优化
PD_Bool PD_ConfigIrOptim(PD_Config* pd_config);

// 设置是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件
// 参数：pd_config - Config 对象指针
//      x         - 是否打印 IR，默认关闭
// 返回：None
void PD_ConfigSwitchIrDebug(PD_Config* pd_config, PD_Bool x);

// 删除图分析阶段指定的 PASS
// 参数：pd_config - Config 对象指针
//      pass_name - 要删除的 PASS 名称
// 返回：None
void PD_DeletePass(PD_AnalysisConfig* config, char* pass_name);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径
const char* model_path  = "./model/inference.pdmodel";  
const char* params_path = "./model/inference.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);

// 开启 IR 优化
PD_ConfigSwitchIrOptim(config, TRUE);
// 开启 IR 打印
PD_ConfigSwitchIrDebug(config, TRUE);
// 删除 PASS fc_fuse_pass
PD_DeletePass(config, "fc_fuse_pass");

// 通过 API 获取 IR 优化是否开启 - True
printf("IR Optim is: %s\n", PD_ConfigIrOptim(config) ? "True" : "False");

// 根据 Config 创建 Predictor, 并销毁该 Config 对象
PD_Predictor* predictor = PD_PredictorCreate(config);

// 利用该 Predictor 进行预测
.......

// 销毁 Predictor 对象
PD_PredictorDestroy(predictor);
```

运行结果示例：

```bash
# PD_ConfigSwitchIrOptim 开启 IR 优化后，运行中会有如下 LOG 输出
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [attention_lstm_fuse_pass]
--- Running IR pass [seqconv_eltadd_relu_fuse_pass]
...
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]

# PD_ConfigSwitchIrDebug 开启 IR 打印后，运行结束之后会在目录下生成如下 DOT 文件
-rw-r--r-- 1 root root  70K Nov 17 10:47 0_ir_simplify_with_basic_ops_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 10_ir_fc_gru_fuse_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 11_ir_graph_viz_pass.dot
...
-rw-r--r-- 1 root root  72K Nov 17 10:47 8_ir_mul_lstm_fuse_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 9_ir_graph_viz_pass.dot
```
## Lite子图
API定义如下：

```c
// 启用 Lite 子图
// 参数：pd_config         - Config 对象指针
//      precision         - Lite 子图的运行精度
//      zero_copy         - 启用 zero_copy，Lite 子图与 Paddle Inference 之间共享数据
//      passes_filter_num - 设置 Lite 子图的 PASS 数量
//      passes_filter     - 设置 Lite 子图的 PASS 名称
//      ops_filter_num    - 设置不使用 Lite 子图运行的 OP 数量
//      ops_filter        - 设置不使用 Lite 子图运行的 OP
// 返回：None
void PD_ConfigEnableLiteEngine(PD_Config* pd_config,
                               PD_PrecisionType precision,
                               PD_Bool zero_copy,
                               size_t passes_filter_num,
                               const char** passes_filter,
                               size_t ops_filter_num,
                               const char** ops_filter);

// 判断是否启用 Lite 子图
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 Lite 子图
PD_Bool PD_ConfigLiteEngineEnabled(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 启用 GPU 进行预测 - 初始化 GPU 显存 100MB, Deivce_ID 为 0
PD_ConfigEnableUseGpu(config, 100, 0);

// 启用 Lite 子图
PD_ConfigEnableLiteEngine(config, PD_PRECISION_FLOAT32, FALSE, 0, NULL, 0, NULL);

// 通过 API 获取 Lite 子图启用信息 - True
printf("Lite Engine is: %s\n", PD_ConfigLiteEngineEnabled(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```