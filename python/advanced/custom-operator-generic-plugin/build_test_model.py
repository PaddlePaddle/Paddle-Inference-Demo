import paddle
import paddle.nn.functional as F
import numpy as np
from gap import gap

class GapTestNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.test_attr1 = [1, 2, 3]
        self.test_attr2 = 1
        self.linear = paddle.nn.Linear(96, 1)
        self.conv1 = paddle.nn.Conv2D(3, 6, kernel_size=3)
        self.conv2 = paddle.nn.Conv2D(6, 3, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = gap(x, self.test_attr1, self.test_attr2)
        x = paddle.flatten(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    x = (np.ones([32, 3, 7, 7])).astype('float32')

    model = GapTestNet()
    
    # print(model.sublayers())
    x = paddle.to_tensor(x)
    y = model(x)
    # print(y.shape)
    print(y.numpy().tolist())

    path = "custom_gap_infer_model/custom_gap"
    paddle.jit.save(model, path,
         input_spec=[paddle.static.InputSpec(shape=[32, 3, 7, 7], dtype='float32')])
