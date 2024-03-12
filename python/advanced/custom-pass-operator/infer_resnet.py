# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse

import cv2
import numpy as np
from img_preprocess import preprocess

from paddle.inference import Config, PrecisionType, create_predictor

import custom_relu_op_pass


def init_predictor(args):
    if args.model_dir != "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    gpu_precision = PrecisionType.Float32  # default
    if args.run_mode == "gpu_fp16":
        gpu_precision = PrecisionType.Half

    config.enable_use_gpu(1000, 0, gpu_precision)

    # 下面是关键配置
    config.enable_new_executor()
    config.enable_new_ir()
    # 如果你有多个Pass可以按照预期顺序放进输入参数里
    config.enable_custom_passes(["relu_replace_pass"])

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i])

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Model filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="",
        help="Parameter filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Model dir, If you load a non-combined model, specify the directory of the model.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="gpu_fp32",
        help="Run_mode which can be: gpu_fp32 and gpu_fp16.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred = init_predictor(args)
    img = cv2.imread("./ILSVRC2012_val_00000247.jpeg")
    img = preprocess(img)
    # img = np.ones((1, 3, 224, 224)).astype(np.float32)
    result = run(pred, [img])
    print("class index: ", np.argmax(result[0][0]))
