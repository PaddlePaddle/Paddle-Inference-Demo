# 训练推理示例说明

本文档将向您介绍如何使用 Paddle 2.0 新接口训练和推理一个模型。

## 一、使用 Paddle 2.0 新接口训练一个简单模型

我们参考[ LeNet 的 MNIST 数据集图像分类 ](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/cv_case/image_classification/image_classification.html#lenetmnist)，使用 Paddle 2.0 接口训练一个简单的模型，分别存储成预训练和预测部署模型。我们将着重介绍如何生成模型文件。

- 依赖包导入

```
import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor
```

- 查看 Paddle 版本

```
print(paddle.__version__)
```

- 数据集准备

```
train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())
```

- 构建 LeNet 网络

```
class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
```

- 模型训练

```
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
model = LeNet()
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
def train(model, optim):
    model.train()
    epochs = 2
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()
train(model, optim)
```

- 存储训练模型（训练格式）：您可以参考[ 参数存储 ](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html#id8)，了解如何在动态图下存储训练格式的模型。只需调用`paddle.save`接口即可。

```
paddle.save(model.state_dict(), 'lenet.pdparams')
paddle.save(optim.state_dict(), "lenet.pdopt")
```

## 二、预训练模型如何转换为预测部署模型

- 加载预训练模型：您可以参考[参数载入](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html#id9)了解如何在动态图下加载训练格式的模型，此方法可帮助您完成恢复训练，即模型状态回到训练中断的时刻，恢复训练之后的梯度更新走向是和恢复训练前的梯度走向完全相同的。只需调用`paddle.load`接口加载训练格式的模型，再调用`set_state_dict`接口恢复模型训练中断时刻的状态。

```
model_state_dict = paddle.load('lenet.pdparams')
opt_state_dict = paddle.load('lenet.pdopt')
model.set_state_dict(model_state_dict)
optim.set_state_dict(opt_state_dict)
```

- 存储为预测部署模型：实际部署时，您需要使用预测格式的模型，预测格式模型相对训练格式模型而言，在拓扑上进行了裁剪，去除了预测不需要的算子。您可以参考[InputSpec](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/input_spec_cn.html)来完成动转静功能。只需InputSpec标记模型的输入，调用`paddle.jit.to_static`和`paddle.jit.save`即可得到预测格式的模型。

```
net = to_static(model, input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
paddle.jit.save(net, 'inference_model/lenet')
```

### 参考代码

```
import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor


class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1,
                                      out_channels=6,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6,
                                      out_channels=16,
                                      kernel_size=5,
                                      stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16 * 5 * 5,
                                        out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # x = x.reshape((-1, 1, 28, 28))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


def train(model, optim):
    model.train()
    epochs = 2
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # calc loss
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
                    epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()


if __name__ == '__main__':
    # paddle version
    print(paddle.__version__)

    # prepare datasets
    train_dataset = MNIST(mode='train', transform=ToTensor())
    test_dataset = MNIST(mode='test', transform=ToTensor())

    # load dataset
    train_loader = paddle.io.DataLoader(train_dataset,
                                        batch_size=64,
                                        shuffle=True)

    # build network
    model = LeNet()
    # prepare optimizer
    optim = paddle.optimizer.Adam(learning_rate=0.001,
                                  parameters=model.parameters())

    # train network
    train(model, optim)

    # save training format model
    paddle.save(model.state_dict(), 'lenet.pdparams')
    paddle.save(optim.state_dict(), "lenet.pdopt")

    # load training format model
    model_state_dict = paddle.load('lenet.pdparams')
    opt_state_dict = paddle.load('lenet.pdopt')
    model.set_state_dict(model_state_dict)
    optim.set_state_dict(opt_state_dict)

    # save inferencing format model
    net = to_static(model,
                    input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
    paddle.jit.save(net, 'inference_model/lenet')
```

## 三、使用 Paddle 2.0 Python 接口预测部署

我们使用存储好的预测部署模型，借助 Python 2.0 接口执行预测部署。

### 加载预测模型并进行预测配置

首先，我们加载预测模型，并配置预测时的一些选项，根据配置创建预测引擎：

```python
config = Config("inference_model/lenet/lenet.pdmodel", "inference_model/lenet/lenet.pdiparams") # 通过模型和参数文件路径加载
config.disable_gpu() # 使用cpu预测
predictor = create_predictor(config) # 根据预测配置创建预测引擎predictor
```
更多配置选项可以参考[官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/python_infer_cn.html#config)。

### 设置输入

我们先通过获取输入Tensor的名称，再根据名称获取到输入Tensor的句柄。

```python
# 获取输入变量名称
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])
```

下面我们准备输入数据，并将其拷贝至待预测的设备上。这里我们使用了随机数据，您在实际使用中可以将其换为需要预测的真实图片。

```python
### 设置输入
fake_input = np.random.randn(1, 1, 28, 28).astype("float32")
input_handle.reshape([1, 1, 28, 28])
input_handle.copy_from_cpu(fake_input)
```

### 运行预测

```python
predictor.run()
```

### 获取输出

```python
# 获取输出变量名称
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu()
```
获取输出句柄的方式与输入类似，我们最后获取到的输出是numpy.ndarray类型，方便使用numpy对其进行后续的处理。

### 完整可运行代码
```python
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor

config = Config("inference_model/lenet/lenet.pdmodel", "inference_model/lenet/lenet.pdiparams")
config.disable_gpu()

# 创建PaddlePredictor
predictor = create_predictor(config)

# 获取输入的名称
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

# 设置输入
fake_input = np.random.randn(1, 1, 28, 28).astype("float32")
input_handle.reshape([1, 1, 28, 28])
input_handle.copy_from_cpu(fake_input)

# 运行predictor
predictor.run()

# 获取输出
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu() # numpy.ndarray类型

print(output_data)
```

## 四、使用 Paddle 2.0 C++ 接口预测部署

我们将存储好的模型使用 Paddle 2.0 C++ 接口执行预测部署。

### 准备预测库

首先，我们需要Paddle Inference预测库用于模型推理部署。这里下载2.0.0版本的用于x86 cpu的预测库：

```shell
wget https://paddle-inference-lib.bj.bcebos.com/2.0.0-cpu-avx-mkl/paddle_inference.tgz
```

### 下载预测样例包

下载预测样例代码包：

```shell
wget https://paddle-inference-dist.bj.bcebos.com/lenet_demo.tgz
```

其中，`lenet_infer_test.cc`为预测脚本，`run.sh`为执行脚本。

### 配置依赖库路径

我们需要在执行脚本`run.sh`中，配置预测库的路径：

```shell
LIB_DIR=/path/to/paddle_inference
```

### 加载预测模型并进行预测配置

首先，我们加载预测模型，并配置预测时的一些选项，根据配置创建预测引擎：

```c++
std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.DisableGpu();
  return CreatePredictor(config);
}
```

这里设置使用CPU来进行预测，更多配置选项可以参考[官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/native_infer.html#a-name-config-config-a)。

### 设置输入

我们先通过获取输入Tensor的名称，再根据名称获取到输入Tensor的句柄：

```c++
  std::vector<float> input(1 * 1 * 28 * 28, 0);
  # 获取输入变量名称
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input.data());
```

这里我们使用了全零数据，您在实际使用中可以将其换为需要预测的真实图片。

### 运行预测

```c++
predictor->Run();
```

### 获取输出

```c++
# 获取输出变量名称
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
```
out_data即为所需输出，可以对其进行后续的分析和处理。

### 执行预测

配置好之后，在预测样例文件目录下，使用下列命令编译、执行预测样例。

```shell
sh run.sh
./build/lenet_infer_test --model_file=inference_model/lenet/lenet.pdmodel --params_file=inference_model/lenet/lenet.pdiparams
```

即可打印出预测结果：
![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-dc2e9237bf385ad86dea8efae001da13d81e2b2d)
