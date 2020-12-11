# X86 Linux上Python预测部署示例

## 1.1 流程解析

1) 准备环境

请参考[飞桨官网](https://www.paddlepaddle.org.cn/)安装2.0及以上版本的Paddle。

Python安装opencv：`pip install opencv-python`。

2）准备预测模型

使用Paddle训练结束后，得到预测模型，可以用于预测部署。

本示例准备了mobilenet_v1预测模型，可以从[链接](https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz)下载，或者wget下载。

```shell
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleInference/mobilenetv1_fp32.tar.gz
tar zxf mobilenetv1_fp32.tar.gz
```

3）Python导入

```
from paddle.inference import Config
from paddle.inference import create_predictor
```

4) 设置Config

根据预测部署的实际情况，设置Config，用于后续创建Predictor。

Config默认是使用CPU预测，可以设置开启MKLDNN加速、设置CPU的线程数、开启IR优化、开启内存优化。

```python
# args 是解析的输入参数
if args.model_dir == "":
    config = Config(args.model_file, args.params_file)
else:
    config = Config(args.model_dir)
config.enable_mkldnn()
config.set_cpu_math_library_num_threads(args.threads)
config.switch_ir_optim()
config.enable_memory_optim()
```

5) 创建Predictor

```python
predictor = create_predictor(config)
```

6) 设置输入

从Predictor中获取输入的names和handle，然后设置输入数据。

```python
img = cv2.imread(args.img_path)
img = preprocess(img)
input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])
input_tensor.reshape(img.shape)
input_tensor.copy_from_cpu(img.copy())
```

7) 执行Predictor

```python
predictor->Run();
```

8) 获取输出

```python
output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])
output_data = output_tensor.copy_to_cpu()
```

## 1.2 编译运行示例

文件`img_preprocess.py`是对图像进行预处理。
文件`model_test.py`是示例程序。

参考前面步骤准备环境、下载预测模型。

下载预测图片。

```shell
wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
```

执行预测命令。

```
python model_test.py --model_dir mobilenetv1_fp32 --img_path ILSVRC2012_val_00000247.jpeg
``

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。
