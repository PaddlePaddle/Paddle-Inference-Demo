# 枚举类型

## DataType

`DataType`定义了`Tensor`的数据类型，由传入`Tensor`的numpy数组类型确定。

```python
# DataType 枚举定义
class paddle.inference.DataType:

# 获取各个 DataType 对应的字节数
# 参数：dtype - DataType 枚举
# 输出：dtype 对应的字节数
paddle.inference.get_num_bytes_of_data_type(dtype: DataType)
```

DataType 中包括以下成员:

* `INT64`: 64位整型
* `INT32`: 32位整型
* `FLOAT32`: 32位浮点型

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 FLOAT32 类型 DataType
data_type = paddle_infer.DataType.FLOAT32

# 输出 data_type 的字节数 - 4
paddle_infer.get_num_bytes_of_data_type(data_type)
```

## PrecisionType

PrecisionType设置模型的运行精度，默认值为 `kFloat32(float32)`。枚举变量定义如下：

```python
# PrecisionType 枚举定义
class paddle.inference.PrecisionType
```

PrecisionType 中包括以下成员:

* `Float32`: FP32 模式运行
* `Half`: FP16 模式运行
* `Int8`: INT8 模式运行

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 启用 GPU, 初始化100M显存，使用gpu id为0
config.enable_use_gpu(100, 0)

# 开启 TensorRT 预测，精度为 FP32，开启 INT8 离线量化校准
config.enable_tensorrt_engine(precision_mode=paddle_infer.PrecisionType.Float32,
                              use_calib_mode=True)
```