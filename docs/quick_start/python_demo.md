# 预测示例 (Python)

本章节包含2部分内容：(1) [运行 Python 示例程序](#id1)；(2) [Python 预测程序开发说明](#id6)。

## 运行 Python 示例程序

### 1. 安装 Python 预测库

请参照 [官方主页-快速安装](https://www.paddlepaddle.org.cn/install/quick) 页面进行自行安装或编译，当前支持 pip/conda 安装，docker镜像 以及源码编译等多种方式来准备 Paddle Inference 开发环境。

### 2. 准备预测部署模型

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 预测格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可将 `inference.pdmodel` 加载到模型可视化工具 Netron 中打开。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得模型目录即文件如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

### 3. 准备预测部署程序

将以下代码保存为 `python_demo.py` 文件：

```python
import argparse
import numpy as np

# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

def main():
    args = parse_args()

    # 创建 config
    config = paddle_infer.Config(args.model_file, args.params_file)

    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)

    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 设置输入
    fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
    input_handle.reshape([args.batch_size, 3, 318, 318])
    input_handle.copy_from_cpu(fake_input)

    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
    print("Output data size is {}".format(output_data.size))
    print("Output data shape is {}".format(output_data.shape))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

if __name__ == "__main__":
    main()
```

### 4. 执行预测程序

```bash
# 参数输入为本章节第2步中下载的 ResNet50 模型
python python_demo.py --model_file ./resnet50/inference.pdmodel --params_file ./resnet50/inference.pdiparams --batch_size 2
```

成功执行之后，得到的预测输出结果如下：

```bash
# 程序输出结果如下
Output data size is 2000
Output data shape is (2, 1000)
```

## Python 预测程序开发说明

使用 Paddle Inference 开发 Python 预测程序仅需以下五个步骤：


(1) 引用 paddle inference 预测库

```python
import paddle.inference as paddle_infer
```

(2) 创建配置对象，并根据需求配置，详细可参考 [Python API 文档 - Config](../api_reference/python_api_doc/Config_index)

```python
# 创建 config，并设置预测模型路径
config = paddle_infer.Config(args.model_file, args.params_file)
```

(3) 根据Config创建预测对象，详细可参考 [Python API 文档 - Predictor](../api_reference/python_api_doc/Predictor)

```python
predictor = paddle_infer.create_predictor(config)
```

(4) 设置模型输入 Tensor，详细可参考 [Python API 文档 - Tensor](../api_reference/python_api_doc/Tensor)

```python
# 获取输入的名称
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

# 设置输入
fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
input_handle.reshape([args.batch_size, 3, 318, 318])
input_handle.copy_from_cpu(fake_input)
```

(5) 执行预测，详细可参考 [Python API 文档 - Predictor](../api_reference/python_api_doc/Predictor)

```python
predictor.run()
```

(5) 获得预测结果，详细可参考 [Python API 文档 - Tensor](../api_reference/python_api_doc/Tensor)

```python
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
```
