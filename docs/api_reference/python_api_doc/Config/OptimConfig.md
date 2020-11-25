# 设置模型优化方法

## IR 优化

API定义如下：

```python
# 启用 IR 优化
# 参数：x - 是否开启 IR 优化，默认打开
# 返回：None
paddle.inference.Config.switch_ir_optim(x: bool = True)

# 判断是否开启 IR 优化 
# 参数：None
# 返回：bool - 是否开启 IR 优化
paddle.inference.Config.ir_optim()

# 设置是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件
# 参数：x - 是否打印 IR，默认打开
# 返回：None
paddle.inference.Config.switch_ir_debug(x: int=True)

# 返回 pass_builder，用来自定义图分析阶段选择的 IR
# 参数：None
# 返回：PassStrategy - pass_builder对象
paddle.inference.Config.pass_builder()

# 删除字符串匹配为 pass 的 pass
# 参数：pass - 需要删除的 pass 字符串
# 返回：None
paddle.inference.Config.delete_pass(pass: str)
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 开启 IR 优化
config.switch_ir_optim()
# 开启 IR 打印
config.switch_ir_debug()

# 得到 pass_builder 对象
pass_builder = config.pass_builder()

# 或者直接通过 config 去除 fc_fuse_pass
config.delete_pass("fc_fuse_pass")

# 通过 API 获取 IR 优化是否开启 - true
print("IR Optim is: {}".format(config.ir_optim()))

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)


```

运行结果示例：

```bash
# switch_ir_optim 开启 IR 优化后，运行中会有如下 LOG 输出
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [attention_lstm_fuse_pass]
--- Running IR pass [seqconv_eltadd_relu_fuse_pass]
...
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]

# switch_ir_debug 开启 IR 打印后，运行结束之后会在目录下生成如下 DOT 文件
-rw-r--r-- 1 root root  70K Nov 17 10:47 0_ir_simplify_with_basic_ops_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 10_ir_fc_gru_fuse_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 11_ir_graph_viz_pass.dot
...
-rw-r--r-- 1 root root  72K Nov 17 10:47 8_ir_mul_lstm_fuse_pass.dot
-rw-r--r-- 1 root root  72K Nov 17 10:47 9_ir_graph_viz_pass.dot
```

## Lite 子图

```python 
# 启用 Lite 子图
# 参数：precision_mode - Lite 子图的运行精度，默认为 FP32
#      zero_copy      - 启用 zero_copy，lite 子图与 paddle inference 之间共享数据
#      Passes_filter  - 设置 lite 子图的 pass
#      ops_filter     - 设置不使用 lite 子图运行的 op
# 返回：None
paddle.inference.Config.enable_lite_engine(precision_mode: PrecisionType = paddle_infer.PrecisionType.Float32, 
                                           zero_copy: bool = False, 
                                           passes_filter: List[str]=[], 
                                           ops_filter: List[str]=[])


# 判断是否启用 Lite 子图
# 参数：None
# 返回：bool - 是否启用 Lite 子图
paddle.inference.Config.lite_engine_enabled()
```

示例代码：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 启用 GPU 进行预测
config.enable_use_gpu(100, 0)

# 启用 Lite 子图
config.enable_lite_engine(paddle_infer.PrecisionType.Float32)

# 通过 API 获取 Lite 子图启用信息 - true
print("Lite Engine is: {}".format(config.lite_engine_enabled()))
```