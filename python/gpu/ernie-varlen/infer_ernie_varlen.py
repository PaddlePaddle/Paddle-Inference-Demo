import numpy as np
import argparse

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
from paddle.inference import InternalUtils


def init_predictor(args):
    if args.model_dir is not "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()

    gpu_precision = PrecisionType.Float32
    if args.run_mode == "gpu_fp16":
        gpu_precision = PrecisionType.Half

    config.enable_use_gpu(1000, 0, gpu_precision)

    if args.run_mode == "trt_fp16":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
    else:
        raise ValueError(
                        "Can not use trt_fp32. Varlen only support trt_fp16, trt_int8. "
        )

    min_batch = 1
    max_batch = 10
    min_single_seq_len = 1
    max_single_seq_len = 384
    opt_single_seq_len = 384
    min_batch_seq_len = 1
    max_batch_seq_len = 3840
    opt_batch_seq_len = 3840

    input_name0 = "read_file_0.tmp_0"
    input_name1 = "read_file_0.tmp_1"
    input_name2 = "read_file_0.tmp_2"
    input_name3 = "read_file_0.tmp_4"

    min_shape = [min_batch_seq_len]
    max_shape = [max_batch_seq_len]
    opt_shape = [opt_batch_seq_len]
      
    config.set_trt_dynamic_shape_info(
            {input_name0: min_shape,input_name1: min_shape, input_name2: [1], input_name3: [min_batch, min_single_seq_len, 1]},
            {input_name0: max_shape,input_name1: max_shape, input_name2: [max_batch + 1], input_name3: [max_batch, max_single_seq_len, 1]},
            {input_name0: opt_shape,input_name1: opt_shape, input_name2: [max_batch + 1], input_name3: [max_batch, opt_single_seq_len, 1]},
    )

    config.enable_tensorrt_varseqlen()
    InternalUtils.set_transformer_posid(config,input_name2)
    InternalUtils.set_transformer_maskid(config,input_name3)
    
    predictor = create_predictor(config)
    return predictor


def run(predictor):

    run_batch = 10
    seq_len = 384
    run_seq_len = run_batch*seq_len
    max_seq_len = seq_len
    i0 = np.ones(run_seq_len, dtype=np.int64)
    i1 = np.zeros(run_seq_len, dtype=np.int64)
    i2 = np.array([0,384,768,1152,1536,1920,2304,2688,3072,3456,3840], dtype=np.int64)
    i3 = np.ones([run_batch, max_seq_len ,1], dtype=float)

    input_names = predictor.get_input_names()

    input_tensor0 = predictor.get_input_handle(input_names[0])
    input_tensor0.copy_from_cpu(i0)

    input_tensor1 = predictor.get_input_handle(input_names[1])
    input_tensor1.copy_from_cpu(i1)

    input_tensor2 = predictor.get_input_handle(input_names[2])
    input_tensor2.copy_from_cpu(i2)

    input_tensor3 = predictor.get_input_handle(input_names[3])
    input_tensor3.copy_from_cpu(i3)

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
        default="trt_fp16",
        help="Run_mode which can be: trt_fp32, trt_fp16, trt_int8 and gpu_fp16.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred = init_predictor(args)
    result = run(pred)
    print("result: ", result)
