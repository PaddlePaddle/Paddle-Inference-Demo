# create_predictor 方法

API定义如下：

```python
# 根据 Config 构建预测执行器 Predictor
# 参数: config - 用于构建 Predictor 的配置信息
# 返回: Predictor - 预测执行器
paddle.inference.create_predictor(config: Config)
```

代码示例:

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)
```

# get_version 方法

API定义如下：

```python
# 获取 Paddle 版本信息
# 参数: NONE
# 返回: str - Paddle 版本信息
paddle.inference.get_version()
```

代码示例:

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 获取 Paddle 版本信息
paddle_infer.get_version()

# 获得输出如下:
# version: 2.0.0-rc0
# commit: 97227e6
# branch: HEAD
```

# convert_to_mixed_precision 方法

`convert_to_mixed_precision` 接口可对模型精度格式进行修改，如在选定的后端下，将 fp32 格式的模型转换为 fp16 格式，API 定义如下：

```python
# 模型精度转换
# 参数：model_file - fp32 模型文件路径
#      params_file - fp32 权重文件路径
#      mixed_model_file - 混合精度模型文件保存路径
#      mixed_params_file - 混合精度权重文件保存路径
#      mixed_precision - 转换精度，如 PrecisionType.kHalf
#      backend - 后端，如 PlaceType.kGPU
#      keep_io_types - 保留输入输出精度信息，若为 True 则输入输出保留为 fp32 类型，否则转为 precision 类型
#      black_list - 黑名单列表，哪些 op 不需要进行精度类型转换，即不跑低精度
#      white_list - 白名单列表，哪些 op 需要进行精度类型转换，即需要跑低精度
# 返回：NONE
def convert_to_mixed_precision(
    model_file: str,
    params_file: str,
    mixed_model_file: str,
    mixed_params_file: str,
    mixed_precision: PrecisionType,
    backend: PlaceType,
    keep_io_types: bool = True,
    black_list: Set[str] = set(),
    **kwargs,
):
    '''
    Convert a fp32 model to mixed precision model.
    Args:
        model_file: fp32 model file, e.g. inference.pdmodel.
        params_file: fp32 params file, e.g. inference.pdiparams.
        mixed_model_file: The storage path of the converted mixed-precision model.
        mixed_params_file: The storage path of the converted mixed-precision params.
        mixed_precision: The precision, e.g. PrecisionType.Half.
        backend: The backend, e.g. PlaceType.GPU.
        keep_io_types: Whether the model input and output dtype remains unchanged.
        black_list: Operators that do not convert precision.
        kwargs: Supported keys including 'white_list'.
            - white_list: Operators that do convert precision.
    '''
```

代码示例：

```python
from paddle.inference import PrecisionType, PlaceType
from paddle.inference import convert_to_mixed_precision
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src_model', type=str, help='src_model')
    parser.add_argument('src_params', type=str, help='src_params')
    parser.add_argument('dst_model', type=str, help='dst_model')
    parser.add_argument('dst_params', type=str, help='dst_params')
    args = parser.parse_args()

    convert_to_mixed_precision(
        model_file=args.src_model,
        params_file=args.src_params,
        mixed_model_file=args.dst_model,
        mixed_params_file=args.dst_params, 
        mixed_precision=PrecisionType.Half,
        backend=PlaceType.GPU,    
        keep_io_types=True,          
        black_list=set(),     
        white_list=set()     
    )
```
