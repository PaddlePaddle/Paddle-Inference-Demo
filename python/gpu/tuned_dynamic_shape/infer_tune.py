import numpy as np
import argparse

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

shape_file = "shape_range_info.pbtxt"


def init_predictor(args):
    if args.model_dir is not "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0)
    if args.tune:
        config.collect_shape_range_info(shape_file)
    if args.use_trt:
        # using dynamic shpae mode, the max_batch_size will be ignored.
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        if args.tuned_dynamic_shape:
            if args.auto_tune:
                config.enable_tuned_tensorrt_dynamic_shape()
            else:
                config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)

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
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="",
        help="Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Model dir, If you load a non-combined model, specify the directory of the model."
    )
    parser.add_argument(
        "--use_gpu", type=int, default=0, help="Whether use gpu.")
    parser.add_argument(
        "--use_trt", type=int, default=0, help="Whether use trt.")
    parser.add_argument(
        "--tune",
        type=int,
        default=0,
        help="Whether use tune to get shape range.")
    parser.add_argument(
        "--auto_tune",
        type=int,
        default=0,
        help="Whether use auto tune to get shape range.")
    parser.add_argument(
        "--tuned_dynamic_shape",
        type=int,
        default=0,
        help="Whether use tuned dynamic shape.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pred = init_predictor(args)
    for batch in [1, 2, 4]:
        input = np.ones((batch, 3, 224, 224)).astype(np.float32)
        result = run(pred, [input])
        print("class index: ", np.argmax(result[0][0]))
