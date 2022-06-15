# 快速上手Python推理

本章节包含2部分内容, 
- [运行 Python 示例程序](#id1)
- [Python 推理程序开发说明](#id2)

## 运行 Python 示例程序

在此环节中，共包含以下4个步骤，
- 环境准备
- 模型准备
- 推理代码
- 执行程序

### 1. 环境准备

Paddle Inference 提供了 Ubuntu/Windows/MacOS 平台的官方 Release 推理库wheel包，用户需根据开发环境和硬件自行下载安装，具体可参阅[Python推理环境安装](../install/python_install.md)。


### 2. 模型准备

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 推理格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可参考[模型结构可视化文档](../export_model/visual_model.html)。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得模型目录即文件如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

### 3. 推理代码

将以下代码保存为 `python_demo.py` 文件：

```python
import argparse
import numpy as np

# 引用 paddle inference 推理库
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

### 4. 执行程序

```bash
# 参数输入为本章节第2步中下载的 ResNet50 模型
python python_demo.py --model_file ./resnet50/inference.pdmodel --params_file ./resnet50/inference.pdiparams --batch_size 2
```

成功执行之后，得到的推理输出结果如下：

```bash
# 程序输出结果如下
Output data size is 2000
Output data shape is (2, 1000)
```

## Python 推理程序开发说明

使用 Paddle Inference 开发 Python 推理程序仅需以下五个步骤：


(1) 引用 paddle inference 推理库

```python
import paddle.inference as paddle_infer
```

(2) 创建配置对象，并根据需求配置，详细可参考 [Python API 文档 - Config](../api_reference/python_api_doc/Config_index)

```python
# 创建 config，并设置推理模型路径
config = paddle_infer.Config(args.model_file, args.params_file)
```

(3) 根据Config创建推理对象，详细可参考 [Python API 文档 - Predictor](../api_reference/python_api_doc/Predictor)

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

(5) 执行推理，详细可参考 [Python API 文档 - Predictor](../api_reference/python_api_doc/Predictor)

```python
predictor.run()
```

(5) 获得推理结果，详细可参考 [Python API 文档 - Tensor](../api_reference/python_api_doc/Tensor)

```python
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
```

至此 Paddle Inference 推理已跑通，如果想更进一步学习 Paddle Inference，可以根据硬件情况选择学习 GPU 推理、CPU 推理、进阶使用等章节。
