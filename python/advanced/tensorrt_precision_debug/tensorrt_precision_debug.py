import json
import numpy as np
import pandas as pd
import argparse

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

check_diff_tensor_names = []
check_diff_tensor2op = {}
check_diff_baseline_tensor = {}
check_diff_mismatch_tensors_info = {}
check_diff_reorder_info = []
check_diff_mark_tensor_names = []

def get_mark_names_from_serialized_json():
    with open("cache/engine_info_*.json", "r") as f:
        data = json.load(f)
    for layer in data:
        for output in layer['Outputs']:
            output_name = output["Name"]
            if "subgraph" in output_name:
                output_name = output_name[0:output_name.index("_subgraph")]
                check_diff_mark_tensor_names.append(output_name)

def assign_mark_names():
    flag = False
    start = ["tensor_name.tmp_0", "tensor_name.tmp_1"]
    end = ["tensor_name.tmp_5"]
    with open("save_baseline.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        baseline_tensor_names = line.split(":")[-1].split(" ")[-1].strip()
        if baseline_tensor_names in start:
            flag = True
        if flag:
            # print(baseline_tensor_names)
            check_diff_mark_tensor_names.append(baseline_tensor_names)
        if baseline_tensor_names in end:
            flag = False

def save_baseline_hook(op_type: str, tensor_name: str, tensor: paddle.Tensor):
    print(">>>> save_baseline_hook: {} {}".format(op_type, tensor_name))
    # with open("save_baseline.txt", "a") as f:
    #     f.write(">>>> save_baseline_hook: {} {}".format(op_type, tensor_name) + "\n")
    check_diff_tensor_names.append(tensor_name)
    check_diff_tensor2op[tensor_name] = op_type
    check_diff_baseline_tensor[tensor_name] = np.array(tensor)

def assert_tensor_close_hook(op_type: str, tensor_name: str, tensor: paddle.Tensor):
    print(">>>> assert_tensor_close_hook: {} {}".format(op_type, tensor_name))
    if tensor_name in check_diff_baseline_tensor:
        match_status = []
        match_status.append(op_type)
        # match_status.append(check_diff_tensor2op[tensor_name])
        match_status.append(tensor_name)
        actual = np.array(tensor).astype(float)
        desire = check_diff_baseline_tensor[tensor_name].astype(float)
        if (actual.shape != desire.shape):
            match_status.append("expect " + str(desire.shape) +  ", but got " + str(actual.shape))
            if actual.shape[0] > desire.shape[0]:
                actual = actual[:desire.shape[0]]
            elif actual.shape[0] < desire.shape[0]:
                desire = desire[:actual.shape[0]]
        else:
            match_status.append("match, " + str(desire.shape))
                
        if (actual.shape == desire.shape):
            if np.size(actual) != 0:
                atol_array = np.abs(actual - desire)
                rtol_array = atol_array / (np.abs(desire) + 1e-7)
                max_atol = np.max(atol_array)
                max_rtol = np.max(rtol_array)
                min_base = np.min(desire)
                min_cur = np.min(actual)
                max_base = np.max(desire)
                max_cur = np.max(actual)
                compare_ret = np.isclose(actual, desire, rtol=1e-04, atol=1e-04, equal_nan=False)
                item_num = np.size(compare_ret)
                
                match_status.append(str(item_num - np.sum(compare_ret)) + "/" + str(item_num))
                match_status.append("a:" + str(round(max_atol, 6)))
                match_status.append("r:" + str(round(max_rtol, 6)))
                match_status.append(str(round(min_cur, 6)) + "(" + str(round(min_base, 6)) + ")")
                match_status.append(str(round(max_cur, 6)) + "(" + str(round(max_base, 6)) + ")")
        if (len(match_status) != 8):
            for i in range(0, 8 - len(match_status)):
                match_status.append(None)
        check_diff_mismatch_tensors_info[tensor_name] = match_status
    else:
        print("Tensor {} not found in paddle inference with ir optim off.".format(tensor_name))

def reorder():
    check_diff_reorder_info.append(["Operator Type", "Tensor Name", "Shape", 
          "Mismatched Elements", "Max Atol", "Max Rtol", "Min Val(base)", "Max Val(base)"])
    for name in check_diff_tensor_names:
        if name in check_diff_mismatch_tensors_info:
            check_diff_reorder_info.append(check_diff_mismatch_tensors_info[name])

def init_baseline_predictor(args):
    if args.model_dir is not "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0, PrecisionType.Float32)
    config.switch_ir_optim(False)
    
    print(config.summary())
    predictor = paddle.inference.create_predictor(config)
    return predictor

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

    if args.run_mode == "trt_fp32":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
    elif args.run_mode == "trt_fp16":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
    elif args.run_mode == "trt_int8":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Int8,
            use_static=False,
            use_calib_mode=True,
        )
    if args.use_dynamic_shape:
        names = ["inputs"]
        min_input_shape = [[1, 3, 112, 112]]
        max_input_shape = [[1, 3, 448, 448]]
        opt_input_shape = [[1, 3, 224, 224]]

        config.set_trt_dynamic_shape_info(
            {names[0]: min_input_shape[0]},
            {names[0]: max_input_shape[0]},
            {names[0]: opt_input_shape[0]},
        )

    config.enable_tensorrt_inspector(True)
    config.mark_trt_engine_outputs(check_diff_mark_tensor_names)
    
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
        default="",
        help="Run_mode which can be: trt_fp32, trt_fp16, trt_int8 and gpu_fp16.",
    )
    parser.add_argument(
        "--use_dynamic_shape",
        type=int,
        default=0,
        help="Whether use trt dynamic shape.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img = np.ones((1, 3, 224, 224)).astype(np.float32)

    pred_base = init_baseline_predictor(args)
    result_base = run(pred_base, [img])
    
    get_mark_names_from_serialized_json()
    # assign_mark_names()
    
    pred = init_predictor(args)
    result = run(pred, [img])

    reorder()
    df = pd.DataFrame(check_diff_reorder_info)
    df.to_excel('output.xlsx', sheet_name='Sheet1', header=None)
    