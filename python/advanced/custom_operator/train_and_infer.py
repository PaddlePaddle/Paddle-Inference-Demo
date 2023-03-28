# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, Normalize
from paddle.utils.cpp_extension import load
from paddle.inference import Config
from paddle.inference import create_predictor

EPOCH_NUM = 4
BATCH_SIZE = 64

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops", sources=["custom_relu_op.cc", "custom_relu_op.cu"])

class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(
            in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = custom_ops.custom_relu(x)
        x = self.max_pool1(x)
        x = custom_ops.custom_relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = custom_ops.custom_relu(x)
        x = self.linear2(x)
        x = custom_ops.custom_relu(x)
        x = self.linear3(x)
        return x


# set device
paddle.set_device("gpu")

# model
net = LeNet()
loss_fn = nn.CrossEntropyLoss()
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# data loader
transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image, label) in enumerate(train_loader()):
        out = net(image)
        loss = loss_fn(out, label)
        loss.backward()

        if batch_id % 300 == 0:
            print("Epoch {} batch {}: loss = {}".format(epoch_id, batch_id,
                                                        np.mean(loss.numpy())))

        opt.step()
        opt.clear_grad()

# save inference model
path_prefix = "custom_relu_test_dynamic/net"
paddle.jit.save(
    net,
    path_prefix,
    input_spec=[
        paddle.static.InputSpec(
            shape=[None, 1, 28, 28], dtype='float32')
    ])

# run the predictor
np_data = np.random.random((1, 1, 28, 28)).astype("float32")
np_label = np.random.random((1, 1)).astype("int64")

config = Config(path_prefix + ".pdmodel", path_prefix + ".pdiparams")

# set device
config.enable_use_gpu(100)

predictor = create_predictor(config)

input_tensor = predictor.get_input_handle(predictor.get_input_names()[0])
input_tensor.reshape(np_data.shape)
input_tensor.copy_from_cpu(np_data)
predictor.run()
output_tensor = predictor.get_output_handle(predictor.get_output_names()[0])
predict_infer = output_tensor.copy_to_cpu()

# print the results
print(predict_infer)
