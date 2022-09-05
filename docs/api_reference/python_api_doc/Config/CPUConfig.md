# 使用 CPU 进行预测

**注意：**
1. 在 CPU 型号允许的情况下，进行预测库下载或编译试尽量使用带 AVX 和 MKL 的版本
2. 可以尝试使用 Intel 的 MKLDNN 进行 CPU 预测加速，默认 CPU 不启用 MKLDNN
3. 在 CPU 可用核心数足够时，可以通过设置 `set_cpu_math_library_num_threads` 将线程数调高一些，默认线程数为 1

## CPU 设置

API定义如下：

```python
# 设置 CPU Blas 库计算线程数
# 参数：cpu_math_library_num_threads - blas库计算线程数
# 返回：None
paddle.inference.Config.set_cpu_math_library_num_threads(cpu_math_library_num_threads: int)

# 获取 CPU Blas 库计算线程数
# 参数：None
# 返回：int - cpu blas库计算线程数
paddle.inference.Config.cpu_math_library_num_threads()
```

代码示例：

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config()

# 设置 CPU Blas 库线程数为 10
config.set_cpu_math_library_num_threads(10)

# 通过 API 获取 CPU 信息 - 10
print(config.cpu_math_library_num_threads())
```

## MKLDNN 设置

**注意：** 
1. 启用 MKLDNN 的前提为已经使用 CPU 进行预测，否则启用 MKLDNN 无法生效
2. 启用 MKLDNN BF16 要求 CPU 型号可以支持 AVX512，否则无法启用 MKLDNN BF16
3. `set_mkldnn_cache_capacity` 请参考 <a class="reference external" href="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md">MKLDNN cache设计文档</a>

API定义如下：

```python
# 启用 MKLDNN 进行预测加速
# 参数：None
# 返回：None
paddle.inference.Config.enable_mkldnn()

# 判断是否启用 MKLDNN 
# 参数：None
# 返回：bool - 是否启用 MKLDNN
paddle.inference.Config.mkldnn_enabled()

# 设置 MKLDNN 针对不同输入 shape 的 cache 容量大小
# 参数：int - cache 容量大小
# 返回：None
paddle.inference.Config.set_mkldnn_cache_capacity(capacity: int=0)

# 指定使用 MKLDNN 加速的 OP 集合
# 参数：使用 MKLDNN 加速的 OP 集合
# 返回：None
paddle.inference.Config.set_mkldnn_op(op_list: Set[str])

# 启用 MKLDNN BFLOAT16
# 参数：None
# 返回：None
paddle.inference.Config.enable_mkldnn_bfloat16()

# 指定使用 MKLDNN BFLOAT16 加速的 OP 集合
# 参数：使用 MKLDNN BFLOAT16 加速的 OP 集合
# 返回：None
paddle.inference.Config.set_bfloat16_op(op_list: Set[str])

# 设置新版本量化模型的 calibration file 路径
# 参数：新版量化模型的 calibration file 路径
# 返回：None
paddle.inference.Config.set_calibration_file_path(calibration_file_path: str)

# 启用 MKLDNN INT8
# 参数：使用 MKLDNN INT8 加速的 OP 集合
# 返回：None
paddle.inference.Config.enable_mkldnn_int8(op_list: Set[str])
```

代码示例 (1)：使用 MKLDNN 进行预测

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 启用 MKLDNN 进行预测
config.enable_mkldnn()

# 通过 API 获取 MKLDNN 启用结果 - true
print(config.mkldnn_enabled())

# 设置 MKLDNN 的 cache 容量大小
config.set_mkldnn_cache_capacity(1)

# 设置启用 MKLDNN 进行加速的 OP 列表
config.set_mkldnn_op({"softmax", "elementwise_add", "relu"})
```

代码示例 (2)：使用 MKLDNN BFLOAT16 进行预测

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 启用 MKLDNN 进行预测
config.enable_mkldnn()

# 启用 MKLDNN BFLOAT16 进行预测
config.enable_mkldnn_bfloat16()

# 设置启用 MKLDNN BFLOAT16 的 OP 列表
config.set_bfloat16_op({"conv2d"})
```

代码示例 (3)：使用 MKLDNN INT8 进行预测

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1")

# 启用 MKLDNN 进行预测
config.enable_mkldnn()

# 设置新版本量化模型的量化标定文件路径
config.set_calibration_file_path("./mobilenet_v1/calibration_table.txt")

# 启用 MKLDNN INT8 进行预测
config.enable_mkldnn_int8()
```
