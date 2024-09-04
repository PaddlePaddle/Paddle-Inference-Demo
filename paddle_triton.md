
# Paddle Inference 支持Triton自定义算子使用方法

## 1. 相关背景

* Triton支持自定义算子，用户可以自己开发算子并注册到Triton中，在推理时调用。
* Triton 仅提供Python API，采用运行时JIT编译，即仅当用户代码的执行流执行到这个kernel的时候，这个kernel才真正的被编译。  

## 2. Paddle Inference支持Triton自定义算子使用方法

Paddle Inference 提供了部分 `Norm`类和`date copy`类的融合算子,集成在paddlemix库中，用户可以按照需求使用这些算子。  
Paddle Inference支持 `PaddleMix Triton`自定义算子使用方法如下：  
* 步骤1. 用户需要下载`PaddleMIX`库
```bash
git clone https://github.com/PaddlePaddle/PaddleMIX.git
```
* 步骤2. 用户需要安装Triton,并适配Paddle
```bash
python -m pip install triton
python -m pip install git+https://github.com/zhoutianzi666/UseTritonInPaddle.git
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"
```
* 步骤3. 在需要使用Triton算子的python文件中`import paddlemix` 
* 步骤4. 调用PaddleMix中对应的Triton算子API，实现高性能算子加速。

## 3. Paddle Inference 使用`PaddleMixTriton`自定义算子示例
使用Triton算子优化后代码:    
```py

import paddlemix
# 参数准备
emb = self.linear(self.silu(conditioning_embedding).cast(x.dtype))
scale, shift = paddle.chunk(emb, 2, axis=1)

# Triton API :adaptive_layer_norm
x = paddlemix.triton_ops.adaptive_layer_norm(
    x, scale_msa, shift_msa, self.norm.weight, self.norm.bias, epsilon=1e-06)
```

优化前代码:  
```py
# 参数准备
emb = self.linear(self.silu(conditioning_embedding).cast(x.dtype))
scale, shift = paddle.chunk(emb, 2, axis=1)
norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)

# 原低效的算子实现
self.norm = nn.LayerNorm(embedding_dim, epsilon=1e-6, **norm_elementwise_affine_kwargs)
x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
```


## 4. 注意事项
* 1.用户需要注意参数顺序Triton API中规定的参数，以及参数的填充顺序。
* 2.用户需要注意Triton API中参数的默认值与需要填充的参数的是否一致。例如权重和偏置的默认值是否为None。  
给出Triton API中参数的默认值和需要填充的参数的示例如下：

```py
adaptive_layer_norm(x, scale, shift, weight=None, bias=None, epsilon=1e-05)  

fused_adaLN_scale_residual(x, mha_out, gate_msa, scale_mlp, shift_mlp, weight=None, bias=None, epsilon=1e-05)  

split_concat(x, y)

triton_split(x, num_or_sections=[-1, -1], axis=1)
```