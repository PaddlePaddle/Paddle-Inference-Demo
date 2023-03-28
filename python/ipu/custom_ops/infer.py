# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.utils.cpp_extension import load
from paddle.inference import Config
from paddle.inference import create_predictor

EPOCH_NUM = 4
BATCH_SIZE = 64

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops", sources=["custom_relu_op.cc", "custom_relu_op_ipu.cc"],
    extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'])

# run the predictor
np.random.seed(2022)
np_data = np.random.random((1, 1, 28, 28)).astype("float32")
np_label = np.random.random((1, 1)).astype("int64")

config = Config("custom_relu_infer_model/custom_relu.pdmodel", "custom_relu_infer_model/custom_relu.pdiparams")

# enable ipu
config.enable_ipu()
config.set_ipu_custom_info([["custom_relu", "Relu", "custom.ops", "1"]])

# set device
predictor = create_predictor(config)

input_tensor = predictor.get_input_handle(predictor.get_input_names()[0])
input_tensor.reshape(np_data.shape)
input_tensor.copy_from_cpu(np_data)
predictor.run()
output_tensor = predictor.get_output_handle(predictor.get_output_names()[0])
predict_infer = output_tensor.copy_to_cpu()

# print the results
print(predict_infer)
