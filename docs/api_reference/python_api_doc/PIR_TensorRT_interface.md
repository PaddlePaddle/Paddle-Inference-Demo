# PIR-TensorRT 接口类

PIR-TensorRT是基于pir开发的将paddle op下沉到TensorRT op的机制，该机制将原始的pir算子经过Marker Pass和Partition Pass识别出可以进入TRT的子图，经过TensorRT Converter将标记的子图转换成TensorRTEngine Op，最后大部分的执行逻辑在TensorRT中执行。

API定义如下:

Input
-------------------------------

```python
paddle.tensorrt.Input(warmup_data,min_input_shape,max_input_shape,optim_input_shape,input_data_type,input_range,name)
```

用于为模型配置输入数据的类

参数
- **warmup_data** (Tuple[np.ndarray,...] | None = None) - 实际输入数据的元组
- **min_input_shape** (Tuple | None = None) - 输入的最小形状
- **max_input_shape** (Tuple | None = None) - 输入的最大形状
- **optim_input_shape** (Tuple | None = None) - 输入的最优形状
- **input_data_type** (str | None = 'float32') - 输入的数据类型，默认是float32
- **input_range** (Tuple | None = None) - 用于生成输入数据的值范围。对于浮点数，默认范围是 (0.0, 1.0)。对于整数，默认范围是 (1, 10)。此选项仅在提供了 min_input_shape、- optim_input_shape和max_input_shape时适用;不适用于warmup_data。
- **name** (str | None = None) - 模型输入的名称

返回 \
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

返回 \
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
paddle.tensorrt.TensorRTConfig(inputs,min_subgraph_size,save_model_dir,disable_ops,precision_mode,ops_run_float,optimization_level,disable_passes,workspace_size)
```

用于配置TensorRT优化的类

参数

- **inputs** (list) - 模型输入配置的列表,每个元素都是Input类的一个实例,用于指定输入数据的形状、类型等。
- **min_subgraph_size** (int, optional) -最小可以被TensorRT优化的子图数量(默认为3)。
- **save_model_dir** (str, optional) - 指定优化后的模型保存路径，若不指定则不保存。
- **disable_ops** (str|list, optional) - 一个字符串或列表，表示不应该转换为TensorRT的op名称(默认为None)。
- **precision_mode** (PrecisionMode, optional) - 指定TensorRT优化的精度模式,可选PrecisionMode.FP32:32位浮点精度，PrecisionMode.FP16:16位浮点精度，PrecisionMode.INT8:8位浮点精度,PrecisionMode.BFP16:16位Brain浮点精度(仅在TensorRT版本大于9.0时支持)
- **ops_run_float** (str|list, optional) - 指定某些op以fp32精度运行
- **optimization_level** (int, optional) - 设置TensorRT优化级别(默认为3)。仅在TensorRT版本大于8.6时支持，优化级别通常控制TensorRT在优化过程中应用的优化程度。
- **disable_passes** (str|list, optional) - 一个字符串列表，表示不应用于原始程序的pass名称(默认为空列表[])
- **workspace_size** (int, optional) - 指定TensorRT优化过程中可以使用的最大GPU内存(以字节为单位)(默认为1<<30,即1GB)

返回 \
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
__1. PIR-TensorRT建议使用json模型，同时也支持pdmodel，但是转换为PIR的过程中无法控制__

```python
paddle.tensorrt.convert(model_path,config)
```

加载Paddle模型并产生经过TensorRT优化的模型

参数
- **model_path** (str) - 模型路径
- **config** (TensorRTConfig) - TensorRTConfig实例

返回 \
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
