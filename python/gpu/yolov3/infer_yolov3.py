import numpy as np
import argparse
import cv2
from PIL import Image

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

from utils import preprocess, draw_bbox


def init_predictor(args):
    if args.model_dir != "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0)
    if args.run_mode == "trt_fp32":
        config.enable_tensorrt_engine(workspace_size=1 << 30,
                                max_batch_size=1,
                                min_subgraph_size=5,
                                precision_mode=PrecisionType.Float32,
                                use_static=False,
                                use_calib_mode=False)
    elif args.run_mode == "trt_fp16":
        config.enable_tensorrt_engine(workspace_size=1 << 30,
                                max_batch_size=1,
                                min_subgraph_size=5,
                                precision_mode=PrecisionType.Half,
                                use_static=False,
                                use_calib_mode=False)
    elif args.run_mode == "trt_int8":
        config.enable_tensorrt_engine(workspace_size=1 << 30,
                                max_batch_size=1,
                                min_subgraph_size=5,
                                precision_mode=PrecisionType.Int8,
                                use_static=False,
                                use_calib_mode=True)
    if args.use_dynamic_shape:
        names = ["image"]
        min_input_shape = [1, 1, 1]
        max_input_shape = [100, 128, 1]
        opt_input_shape = [10, 60, 1]

        config.set_trt_dynamic_shape_info(
            {names[0]: min_input_shape,}, 
            {names[0]: max_input_shape,}, 
            {names[0]: opt_input_shape,}
        )

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

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
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="",
        help=
        "Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="",
        help=
        "Run_mode which can be: trt_fp32, trt_fp16, trt_int8 and gpu_fp16."
    )
    parser.add_argument("--use_dynamic_shape",
                        type=int,
                        default=0,
                        help="Whether use trt dynamic shape.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_name = 'kite.jpg'
    save_img_name = 'res.jpg'
    im_size = 608
    pred = init_predictor(args)
    img = cv2.imread(img_name)
    data = preprocess(img, im_size)
    scale_factor = np.array([im_size * 1. / img.shape[0], im_size * 1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
    im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)
    result = run(pred, [im_shape, data, scale_factor])
    img = Image.open(img_name).convert('RGB')
    draw_bbox(img, result[0], save_name=save_img_name)
