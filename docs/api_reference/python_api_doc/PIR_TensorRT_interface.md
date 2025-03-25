# PIR-TensorRT 接口类

PIR-TensorRT是基于pir开发的将paddle op下沉到TensorRT op的机制，该机制将原始的pir算子经过Marker Pass和Partition Pass识别出可以进入TRT的子图，经过TensorRT Converter将标记的子图转换成TensorRTEngine Op，最后大部分的执行逻辑在TensorRT中执行。

API定义如下:

## 1. Input输入类

```python
# 一个用于为模型配置输入数据的类
<<<<<<< HEAD
# 参数:warmup_data:tuple[np.ndarray,...] | None = None 实际输入数据的元组。
#     min_input_shape:tuple | None = None 输入的最小形状。
#     max_input_shape:tuple | None = None 输入的最大形状。
#     optim_input_shape:tuple | None = None 输入的最优形状。
#     input_data_type:str | None = 'float32' 输入的数据类型，默认是float32。
#     input_range:tuple | None = None 用于生成输入数据的值范围。对于浮点数，默认范围是 (0.0, 1.0)。对于整数，默认范围是 (1, 10)。此选项仅在提供了 min_input_shape、optim_input_shape 和 max_input_shape 时适用；不适用于 warmup_data。
#     name:str | None = None 模型输入的名称。
=======
# 参数:warmup_data:tuple[np.ndarray,...] | None = None 实际输入数据的元组（用于自动形状收集机制）。
#     min_input_shape:tuple | None = None 输入的最小形状。
#     max_input_shape:tuple | None = None 输入的最大形状。
#     optim_input_shape:tuple | None = None 输入的优化形状。
#     input_data_type:str | None = 'float32' 输入的数据类型。
#     input_range:tuple | None = None 输入的范围。
#     name:str | None = None 输入的名称。
>>>>>>> 048737927e9c3e0297d250d1d3f62bfa8cdbfa3b

class Input:
    def __init__(
        self,
        warmup_data: tuple[np.ndarray, ...] | None = None,
        min_input_shape: tuple | None = None,
        max_input_shape: tuple | None = None,
        optim_input_shape: tuple | None = None,
        input_data_type: str | None = 'float32',
        input_range: tuple | None = None,
        name: str | None = None,
    ) -> None:
        """
        A class used to configure input data for models. This class serves two purposes:

        1. Random Data Generation: When no input data is supplied, it automatically generates random input data based on the specified minimum, optimal, and maximum shapes. In this mode,you can configure the data type (e.g., 'float32', 'int64', etc.) and the range of values (e.g.,(0.0, 1.0) for floats or (1, 10) for integers).

        2. User-Provided Input: Alternatively, you can supply your own input data via the `warmup_data` argument. In this case, the provided data will be used directly, and the`input_data_type` and `input_range` settings will be ignored.

        Args:
            warmup_data (tuple):
                The tuple of actual input data (for the automatic shape collection mechanism).
            min_input_shape (tuple):
                The shape of the minimum input tensor.
            max_input_shape (tuple):
                The shape of the maximum input tensor.
            optim_input_shape (tuple):
                The shape of the optimal input tensor.
            input_data_type (str, optional):
                The data type for the input tensors, such as 'float32' or 'int64' or 'float32' or 'int32'  (default is float32).
                This option only applies when min_input_shape, optim_input_shape, and max_input_shape are provided; it does not apply to warmup_data.
            input_range (tuple, optional):
                The range of values used to generate input data. For floats, the default range is (0.0, 1.0). For integers, the default range is (1, 10).
                This option only applies when min_input_shape, optim_input_shape, and max_input_shape are provided; it does not apply to warmup_data.
            name:(str,optional):
                The name of the input to the model.
        Returns:
            None
        """
```


代码示例:


```python
from paddle.tensorrt.export import Input
input_config=Input(
    min_input_shape=(1,100),
    optim_input_shape=(4,100),
    max_input_shape=(8,100),
)

from paddle.tensorrt.export import Input
import numpy as np
input_config=Input(
    warmup_data=(
        np.random.rand(1,100).astype(np.float32),
        np.random.rand(4,100).astype(np.float32),
        np.random.rand(8,100).astype(np.float32),
    )
)
```

## 2. PrecisionMode类
```python
# 此类定义了可用于配置TensorRT优化的不同精度模式。这些模式包括FP32、FP16、BF16和INT8。
class PrecisionMode(Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"

    """
    This class defines different precision modes that can be used to configure
    TensorRT optimization. The modes include FP32, FP16, BF16, and INT8.
    Specifies the precision mode for TensorRT optimization. The options are:
    - PrecisionMode.FP32: 32-bit floating point precision (default).
    - PrecisionMode.FP16: 16-bit floating point precision.
    - PrecisionMode.INT8: 8-bit integer precision.
    - PrecisionMode.BFP16: 16-bit Brain Floating Point precision. Only supported in TensorRT versions greater than 9.0.
    """
```

## 3. TensorRTConfig类

```python
# 一个用于配置 TensorRT 优化的类。
# 参数:inputs:(list) 模型输入配置的列表,每个元素都是Input类的一个实例,用于指定输入数据的形状、类型等。
#     min_subgraph_size(int,optional):最小可以被TensorRT优化的子图数量(默认为3)。
#     save_model_dir(str,optional):指定优化后的模型保存路径，若不指定则不保存。
#     disable_ops(str|list,optional):一个字符串或列表，表示不应该转换为TensorRT的op名称(默认为None)。
#     precision_mode(PrecisionMode,optional):指定TensorRT优化的精度模式,可选PrecisionMode.FP32:32位浮点精度，PrecisionMode.FP16:16位浮点精度，PrecisionMode.INT8:8位浮点精度,PrecisionMode.BFP16:16位Brain浮点精度(仅在TensorRT版本大于9.0时支持)
#     ops_run_float(str|list,optional):指定某些op以fp32精度运行
#     optimization_level(int,optional):设置TensorRT优化级别(默认为3)。仅在TensorRT版本大于8.6时支持，优化级别通常控制TensorRT在优化过程中应用的优化程度。
#     disable_passes(str|list,optional):一个字符串列表，表示不应用于原始程序的pass名称(默认为空列表[])
#     workspace_size(int,optional):指定TensorRT优化过程中可以使用的最大GPU内存(以字节为单位)(默认为1<<30,即1GB)

class TensorRTConfig:
    def __init__(
        self,
        inputs: list,
        min_subgraph_size: int | None = 3,
        save_model_dir: str | None = None,
        disable_ops: str | list | None = None,
        precision_mode: PrecisionMode = PrecisionMode.FP32,
        ops_run_float: str | list | None = None,
        optimization_level: int | None = 3,
        disable_passes: list = [],
        workspace_size: int | None = 1 << 30,
    ) -> None:
        """
        A class for configuring TensorRT optimizations.

        Args:
            inputs (list):
                A list of Input configurations
            min_subgraph_size (int, optional):
                The minimum number of operations in a subgraph for TensorRT to optimize (default is 3).
            save_model_dir (str, optional):
                The directory where the optimized model will be saved (default is not to save).
            disable_ops : (str|list, optional):
                A string representing the names of operations that should not be entering by TensorRT (default is None).
            precision_mode (PrecisionMode, optional):
                Specifies the precision mode for TensorRT optimization. The options are:
                - PrecisionMode.FP32: 32-bit floating point precision (default).
                - PrecisionMode.FP16: 16-bit floating point precision.
                - PrecisionMode.INT8: 8-bit integer precision.
                - PrecisionMode.BFP16: 16-bit Brain Floating Point precision. Only supported in TensorRT versions greater than 9.0.
            ops_run_float (str|list, optional):
                A set of operation names that should be executed using FP32 precision regardless of the `tensorrt_precision_mode` setting.
                The directory where the optimized model will be saved (default is None).
            optimization_level (int, optional):
                Set TensorRT optimization level (default is 3). Only supported in TensorRT versions greater than 8.6.
            disable_passes : (str|list, optional):
                A list of string representing the names of pass that should not be used for origin program (default is []).
            workspace_size (int, optional):
                Specifies the maximum GPU memory (in bytes) that TensorRT can use for the optimization process (default is 1 << 30).
        Returns:
            None

```

代码示例:


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

## 4.paddle.tensorrt.convert(model_path,config)

### 注意:
1.PIR-TensorRT建议使用json模型，同时也支持pdmodel，但是转换为PIR的过程中无法控制

API定义如下:
```python
# 加载Paddle模型并产生经过TensorRT优化的模型
# 参数:model_path:Paddle的模型路径，既可以是模型前缀，如model_dir/inference,也可以是model_dir/inference.json
# config:TensorRTConfig
# 返回:经过TensorRT优化的模型
paddle.tensorrt.convert(model_path,config)
```

代码示例1:
```python
# 此示例采用用户指定的模型输入形状，Paddle会在内部生成相应的随机数据
import numpy as np
import paddle
import paddle.inference as paddle_infer
import paddle.nn.functional as F
from paddle import nn
from paddle.tensorrt.export import Input,TensorRTConfig

class LinearNet(nn.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return F.relu(self.linear(x))

input_dim = 3
layer = LinearNet(input_dim)

save_path="/tmp/linear_net"
paddle.jit.save(layer, save_path, [paddle.static.InputSpec(shape=[-1, input_dim])])

input_config = Input(
    min_input_shape=[1, input_dim],
    optim_input_shape=[2, input_dim],
    max_input_shape=[4, input_dim],
    name='x',
)

trt_config = TensorRTConfig(inputs=[input_config])
trt_config.save_model_dir = "/tmp/linear_net_trt"

program_with_trt = paddle.tensorrt.convert(save_path, trt_config)

config = paddle_infer.Config(
    trt_config.save_model_dir + '.json',
    trt_config.save_model_dir + '.pdiparams',
)
config.enable_use_gpu(100, 0)
predictor = paddle_infer.create_predictor(config)

input_data = np.random.randn(2, 3).astype(np.float32)
model_input = paddle.to_tensor(input_data)
output_converted = predictor.run([model_input])

```

代码示例2:
```python
# 此示例使用用户指定的真实输入
import numpy as np
import paddle
import paddle.inference as paddle_infer
import paddle.nn.functional as F
from paddle import nn
from paddle.tensorrt.export import Input,TensorRTConfig

class LinearNet(nn.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return F.relu(self.linear(x))

input_dim=3
layer = LinearNet(input_dim)

save_path = "/tmp/linear_net"
paddle.jit.save(layer, save_path, [paddle.static.InputSpec(shape=[-1, input_dim])])

input_config = Input(
    warmup_data=(
        np.random.rand(1,3).astype(np.float32),
        np.random.rand(2,3).astype(np.float32),
        np.random.rand(4,3).astype(np.float32),
    ),
    name='x',
)

trt_config = TensorRTConfig(inputs=[input_config])
trt_config.save_model_dir = "/tmp/linear_net_trt"

program_with_trt = paddle.tensorrt.convert(save_path, trt_config)

config = paddle_infer.Config(
    trt_config.save_model_dir + '.json',
    trt_config.save_model_dir + '.pdiparams',
)
config.enable_use_gpu(100, 0)
predictor = paddle_infer.create_predictor(config)

input_data = np.random.randn(2, 3).astype(np.float32)
model_input = paddle.to_tensor(input_data)
output_converted = predictor.run([model_input])
```
