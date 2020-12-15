# 设置模型优化方法

API定义如下：

```go
// 启用 IR 优化
// 参数：config - AnalysisConfig 对象指针
//      x - 是否开启 IR 优化，默认打开
// 返回：None
func (config *AnalysisConfig) SwitchIrOptim(x bool)

// 判断是否开启 IR 优化 
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 是否开启 IR 优化
func (config *AnalysisConfig) IrOptim() bool

// 设置是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件
// 参数：config - AnalysisConfig 对象指针
//      x - 是否打印 IR，默认关闭
// 返回：None
func (config *AnalysisConfig) SwitchIrDebug(x bool)

// 返回 pass_builder，用来自定义图分析阶段选择的 IR
// 参数：config - AnalysisConfig 对象指针
//      pass - 需要删除的 pass 名称
// 返回：None
func (config *AnalysisConfig) DeletePass(pass string)
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import "/pathto/Paddle/go/paddle"

func main() {
    // 创建 AnalysisConfig 对象
    config := paddle.NewAnalysisConfig()

    // 设置预测模型路径，这里为非 Combined 模型
    config.SetModel("data/model/__model__", "data/model/__params__")

    // 开启 IR 优化
    config.SwitchIrOptim(true);
    // 开启 IR 打印
    config.SwitchIrDebug(true);

    // 通过 API 获取 IR 优化是否开启 - true
    println("IR Optim is: ", config.IrOptim())

    // 通过 config 去除 fc_fuse_pass
    config.DeletePass("fc_fuse_pass")

    // 根据 Config 创建 Predictor
    predictor := paddle.NewPredictor(config)

    // 删除 Predictor
    paddle.DeletePredictor(predictor)
}
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
