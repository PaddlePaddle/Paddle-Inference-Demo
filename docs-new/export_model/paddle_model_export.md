# 训练推理示例说明

本文档将向您介绍如何使用 Paddle 训练和推理一个模型。

## 使用 Paddle 训练一个简单模型

我们参考[ LeNet 的 MNIST 数据集图像分类 ](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/cv_case/image_classification/image_classification.html#lenetmnist)，使用 Paddle 训练一个简单的模型，分别存储成预训练和预测部署模型。我们将着重介绍如何生成模型文件。

- 依赖包导入

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

- 存储训练模型（训练格式）：您可以参考[ 参数保存 ](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/model_save_load_cn.html#canshubaocun)，了解如何在动态图下存储训练格式的模型。只需调用`paddle.save`接口即可。

```
paddle.save(model.state_dict(), 'lenet.pdparams')
paddle.save(optim.state_dict(), "lenet.pdopt")
```

## 预训练模型如何转换为预测部署模型

- 加载预训练模型：您可以参考[参数载入](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/model_save_load_cn.html#canshuzairu)了解如何在动态图下加载训练格式的模型，此方法可帮助您完成恢复训练，即模型状态回到训练中断的时刻，恢复训练之后的梯度更新走向是和恢复训练前的梯度走向完全相同的。只需调用`paddle.load`接口加载训练格式的模型，再调用`set_state_dict`接口恢复模型训练中断时刻的状态。

```
model_state_dict = paddle.load('lenet.pdparams')
opt_state_dict = paddle.load('lenet.pdopt')
model.set_state_dict(model_state_dict)
optim.set_state_dict(opt_state_dict)
```

- 存储为预测部署模型：实际部署时，您需要使用预测格式的模型，预测格式模型相对训练格式模型而言，在拓扑上进行了裁剪，去除了预测不需要的算子。您可以参考[InputSpec](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/InputSpec_cn.html#inputspec)来完成动转静功能。只需InputSpec标记模型的输入，调用`paddle.jit.to_static`和`paddle.jit.save`即可得到预测格式的模型。

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

## 使用 PaddleSlim 进行模型压缩

若使用PaddleSlim产出量化模型，请参考文档：

### 静态图量化

- [离线量化-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/quant_post_static_tutorial.html)
- [量化训练-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/quant_aware_tutorial.html)
- [量化API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/quant/quantization_api.html)

### 动态图量化

- [离线量化-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/dygraph/dygraph_quant_post_tutorial.html)
- [量化训练-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/dygraph/dygraph_quant_aware_training_tutorial.html)
- [量化API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/dygraph/quanter/qat.html)
