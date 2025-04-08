# Paddle-TensorRT 接口类

本文档共介绍四个公开接口<br>
paddle.tensorrt.Input 用于为模型配置输入数据<br>
paddle.tensorrt.PrecisionMode 用于为模型配置精度模式<br>
paddle.tensorrt.TensorRTConfig 用于TensorRT优化配置的类<br>
paddle.tensorrt.convert 加载Paddle模型并产生经过TensorRT优化的模型<br>

API定义如下:

Input
-------------------------------

```python
paddle.tensorrt.Input(warmup_data,min_input_shape,max_input_shape,optim_input_shape,input_data_type,input_range,name)
```

用于为模型配置输入数据的类
**注意：**
1. 必须要选择warmup_data或者min_input_shape,max_input_shape,optim_input_shape。
2. warmup_data和min_input_shape,max_input_shape,optim_input_shape不能同时存在。
3. 选择了warmup_data,input_range和input_data_type不生效。

参数
- **warmup_data** (Tuple[np.ndarray,...] | None = None) - 实际输入数据的元组,默认值为 ``None`` 。
- **min_input_shape** (Tuple | None = None) - 输入的最小形状,默认值为 ``None`` 。
- **max_input_shape** (Tuple | None = None) - 输入的最大形状,默认值为 ``None`` 。
- **optim_input_shape** (Tuple | None = None) - 输入的最优形状,默认值为 ``None`` 。
- **input_data_type** (str | None = 'float32') - 输入的数据类型，默认是float32,默认值为 ``None`` 。
- **input_range** (Tuple | None = None) - 用于生成输入数据的值范围。对于浮点数，默认范围是 (0.0, 1.0)。对于整数，默认范围是 (1, 10)。此选项仅在提供了 min_input_shape、- optim_input_shape和max_input_shape时适用;不适用于warmup_data,默认值为 ``None`` 。
- **name** (str | None = None) - 模型输入的名称,默认值为 ``None`` 。

返回<br>
None

代码示例

```python
>>> from paddle.tensorrt.export import Input
>>> input_config = Input(
>>>     min_input_shape=(1,100),
>>>     optim_input_shape=(4,100),
>>>     max_input_shape=(8,100),
>>> )
>>> input_config.input_data_type='int64'
>>> input_config.input_range=(1,10)

>>> from paddle.tensorrt.export import Input
>>> import numpy as np
>>> input_config = Input(
>>>     warmup_data=(
>>>         np.random.rand(1,100).astype(np.float32),
>>>         np.random.rand(4,100).astype(np.float32),
>>>         np.random.rand(8,100).astype(np.float32),
>>>     )
>>> )
```

PrecisionMode
-------------------------------

```python
paddle.tensorrt.PrecisionMode(Enum)
```

此类定义了可用于配置TensorRT优化的不同精度模式。这些模式包括FP32、FP16、BF16和INT8。

参数

- **Enum** (Enum) - 枚举类型，包括FP32、FP16、BF16和INT8。

返回<br>
None

代码示例

```python
PrecisionMode.FP32
PrecisionMode.FP16
PrecisionMode.INT8
PrecisionMode.BFP16
```

TensorRTConfig
-------------------------------

```python
paddle.tensorrt.TensorRTConfig(inputs, min_subgraph_size, save_model_dir, disable_ops, precision_mode, ops_run_float, optimization_level, disable_passes, workspace_size, use_cuda_graph)
```

用于配置TensorRT优化的类

参数

- **inputs** (list) - 模型输入配置的列表,每个元素都是Input类的一个实例,用于指定输入数据的形状、类型等。
- **min_subgraph_size** (int, 可选) -最小可以被TensorRT优化的子图数量,默认为``3``。
- **save_model_dir** (str, 可选) - 指定优化后的模型保存路径，若不指定则不保存。
- **disable_ops** (str|list, 可选) - 一个字符串或列表，表示不应该转换为TensorRT的op名称,默认值为 ``None``。
- **precision_mode** (PrecisionMode, 可选) - 指定TensorRT优化的精度模式,可选PrecisionMode.FP32:32位浮点精度，PrecisionMode.FP16:16位浮点精度，PrecisionMode.INT8:8位浮点精度,PrecisionMode.BFP16:16位Brain浮点精度(仅在TensorRT版本大于9.0时支持)
- **ops_run_float** (str|list, 可选) - 指定某些op以fp32精度运行
- **optimization_level** (int, 可选) - 设置TensorRT优化级别,默认为``3``。仅在TensorRT版本大于8.6时支持，优化级别通常控制TensorRT在优化过程中应用的优化程度。
- **disable_passes** (str|list, 可选) - 一个字符串列表，表示不应用于原始程序的pass名称,默认为空列表[]。
- **workspace_size** (int, 可选) - 指定TensorRT优化过程中可以使用的最大GPU内存(以字节为单位)(默认为1<<30,即1GB)
- **use_cuda_graph** (bool, 可选) - 是否启用 CUDA Graph 优化。默认为 False，如果设置为 True，则所有算子均转换为 TensorRT 优化生效。

返回<br>
None

代码示例

```python
from paddle.tensorrt.export import(
    Input,
    TensorRTConfig,
    PrecisionMode,
)
Input_config=Input(
    min_input_shape=(1,100),
    optim_input_shape=(4,100),
    max_input_shape=(8,100),
)
input_config.input_data_type='int64'
input_config.input_range=(1,10)

trt_config=TensorRTConfig(inputs=[input_config])
```

paddle.tensorrt.convert(model_path,config)
-------------------------------

**注意：**
**Paddle-TensorRT建议使用json模型，同时也支持pdmodel，但是转换为新的中间表示(PIR)的过程中无法控制**

```python
paddle.tensorrt.convert(model_path,config)
```

加载Paddle模型并产生经过TensorRT优化的模型

参数
- **model_path** (str) - 模型路径
- **config** (TensorRTConfig) - TensorRTConfig实例

返回<br>
program:经过TensorRT优化的program

代码示例

```python
# 此示例采用用户指定的模型输入形状，Paddle会在内部生成相应的随机数据

>>> import numpy as np
>>> import paddle
>>> import paddle.inference as paddle_infer
>>> import paddle.nn.functional as F
>>> from paddle import nn
>>> from paddle.tensorrt.export import Input, TensorRTConfig

>>> class LinearNet(nn.Layer):
>>>     def __init__(self, input_dim):
>>>         super().__init__()
>>>         self.linear = nn.Linear(input_dim, input_dim)

>>>     def forward(self, x):
>>>         return F.relu(self.linear(x))

>>> input_dim = 3
>>> layer = LinearNet(input_dim)

>>> save_path = "/tmp/linear_net"
>>> paddle.jit.save(layer, save_path, [paddle.static.InputSpec(shape=[-1, input_dim])])

>>> input_config = Input(
>>>     min_input_shape=[1, input_dim],
>>>     optim_input_shape=[2, input_dim],
>>>     max_input_shape=[4, input_dim],
>>>     name='x',
>>> )

>>> trt_config = TensorRTConfig(inputs=[input_config])
>>> trt_config.save_model_dir = "/tmp/linear_net_trt"

>>> program_with_trt = paddle.tensorrt.convert(save_path, trt_config)

>>> config = paddle_infer.Config(
>>>     trt_config.save_model_dir + '.json',
>>>     trt_config.save_model_dir + '.pdiparams',
>>> )
>>> config.enable_use_gpu(100, 0)
>>> predictor = paddle_infer.create_predictor(config)

>>> input_data = np.random.randn(2, 3).astype(np.float32)
>>> model_input = paddle.to_tensor(input_data)

>>> output_converted = predictor.run([model_input])

# 示例2:
# 此示例使用用户指定的真实输入

>>> import numpy as np
>>> import paddle
>>> import paddle.inference as paddle_infer
>>> import paddle.nn.functional as F
>>> from paddle import nn
>>> from paddle.tensorrt.export import Input, TensorRTConfig

>>> class LinearNet(nn.Layer):
>>>     def __init__(self, input_dim):
>>>         super().__init__()
>>>         self.linear = nn.Linear(input_dim, input_dim)

>>>     def forward(self, x):
>>>         return F.relu(self.linear(x))

>>> input_dim = 3
>>> layer = LinearNet(input_dim)

>>> save_path = "/tmp/linear_net".
>>> paddle.jit.save(layer, save_path, [paddle.static.InputSpec(shape=[-1, input_dim])])

>>> input_config = Input(
>>>     warmup_data=(
>>>         np.random.rand(1,3).astype(np.float32),
>>>         np.random.rand(2,3).astype(np.float32),
>>>         np.random.rand(4,3).astype(np.float32),
>>>     ),
>>>     name='x',
>>> )

>>> trt_config = TensorRTConfig(inputs=[input_config])
>>> trt_config.save_model_dir = "/tmp/linear_net_trt"

>>> program_with_trt = paddle.tensorrt.convert(save_path, trt_config)

>>> config = paddle_infer.Config(
>>>     trt_config.save_model_dir + '.json',
>>>     trt_config.save_model_dir + '.pdiparams',
>>> )
>>> config.enable_use_gpu(100, 0)
>>> predictor = paddle_infer.create_predictor(config)

>>> input_data = np.random.randn(2, 3).astype(np.float32)
>>> model_input = paddle.to_tensor(input_data)

>>> output_converted = predictor.run([model_input])
```
