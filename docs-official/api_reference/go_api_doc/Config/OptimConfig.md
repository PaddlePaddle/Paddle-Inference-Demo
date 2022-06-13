# 设置模型优化方法

API定义如下：

```go
// 启用 IR 优化
// 参数：x - 是否开启 IR 优化，默认打开
// 返回：None
func (config *Config) SwitchIrOptim(x bool)

// 判断是否开启 IR 优化 
// 参数：无
// 返回：bool - 是否开启 IR 优化
func (config *Config) IrOptim() bool

// 设置是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件
// 参数：x - 是否打印 IR，默认关闭
// 返回：None
func (config *Config) SwitchIrDebug(x bool)

// // 返回 pass_builder，用来自定义图分析阶段选择的 IR
// // 参数：pass - 需要删除的 pass 名称
// // 返回：None
// func (config *Config) DeletePass(pass string)
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 Config 对象
    config := pd.NewConfig()

    // 设置预测模型路径
    config.SetModel("./model/resnet.pdmodel", "./model/resnet.pdiparams")

    // 开启 IR 优化
    config.SwitchIrOptim(true);
    // 开启 IR 打印
    config.SwitchIrDebug(true);

    // 通过 API 获取 IR 优化是否开启 - true
    fmt.Println("IR Optim is: ", config.IrOptim())

    // 根据 Config 创建 Predictor
    predictor := paddle.NewPredictor(config)
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
