
# 混合精度推理diff原因排查方法


一般来说，混合精度推理中结果与 FP32 推理精度相比出现 diff，是由于run在低精度下的 `OP/Kernel` 计算出现 diff，原因有很多：
1. 低精度 `OP/Kernel` 的开发实现不合理（如本该 cast 成 FP32 计算的操作仍使用 FP16 计算）；
2. 有的 `OP/Kernel` 计算中依赖 OP 的特定属性值，属性值的范围可能很大（超过 Fp16 精度范围），这时往往表现在计算出 NAN（如scale op）；
3. 有些 `OP/Kernel` 的实现在特定输入场景下才会发生精度 diff 问题；
4. ...


这也是我们在享受混合精度训练/推理带来的高性能/低内存优势的同时必须要忍受的可能存在精度问题的代价。正所谓鱼与熊掌不可兼得。


当发现精度存在 diff 时，很直观的想法就是，我们希望能够看到模型运行中的所有中间 tensor 数据，和 FP32 模式数据一比较，就能定位存在精度问题的 OP。


Paddle Inference为此提供了一个方便易用的工具，即`predictor.register_output_hook(...)`来定位到精度问题的 OP，之后，再使用`config.exp_disable_mixed_precision_ops(...)`来禁止该 OP run在低精度下。

以 `Python` 为例，我们可能会出现下面的代码片段。
```python
# ...
config = paddle_infer.Config(model_path)
config.enable_use_gpu(512, 0, PrecisionType.Half)
# ...
predictor = paddle_infer.create_predictor(config)
# ...
predictor.run()
# ...
```
如下所示，加入下面几行代码。
```python
# ...
config = paddle_infer.Config(model_path)
config.enable_use_gpu(512, 0, PrecisionType.Half)
# ...
predictor = paddle_infer.create_predictor(config)

# 三个接收参数分别对应
# op_type：op 的名字
# tensor_name: op 的输出 var 的名字
# tensor: 对应的输出 paddle.Tensor 对象
def hook_func(op_type: str, tensor_name: str, tensor: paddle.Tensor):
    # ... 在这里你可以求和、求均值、求方差或者直接输出全部数据等等
    print(op_type, tensor_name, tensor)
    # ...

predictor.register_output_hook(hook_func)

# ...
predictor.run()
# ...
```
上面的代码 FP32 跑一遍，FP16 跑一遍，对比下结果就行。NAN 的问题更简单，直接从 FP16 的结果就能看出哪个 OP 结果为 NAN。实际中，你也可以先根据经验猜测是哪个 OP 的问题，然后在 hook 函数里只输出这个 OP 的数据。

比如说我们发现softmax op和fc op有精度问题，那么我们就可以禁止这两个 run 在低精度来验证。
```python
# ...
config.exp_disable_mixed_precision_ops(set(['softmax', 'fc']))
# ...
```
基本就是这样排查精度 diff 问题。
