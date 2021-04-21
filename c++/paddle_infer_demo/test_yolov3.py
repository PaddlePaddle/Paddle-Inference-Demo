import numpy as np
import argparse
import time
import os


from paddle.inference import Config
from paddle.inference import create_predictor

def init_predictor(args):
    config = Config()
    if args.model_dir == "":
        config.set_model(args.model_file, args.params_file)
    else:
        config.set_model(args.model_dir)
    #config.disable_glog_info()
    config.enable_use_gpu(1000, 3)
    predictor = create_predictor(config)
    return predictor

def run(args, predictor, data):
    # copy data to input tensor
    
    input_names = predictor.get_input_names()
    for i,  name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(data[i].shape)   
        data[i] = data[i].copy()
        input_tensor.copy_from_cpu(data[i])
        
 
    # warm up
    for i in range(10):
        predictor.run()

    # do the inference
    repeat = 100
    start = time.clock()
    for i in range(repeat):
        for i,  name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(data[i].shape)
            input_tensor.copy_from_cpu(data[i])
        predictor.run()
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
    end = time.clock()

    precision = "int8" if args.use_int8 else "float32"    
    latency = (end - start) * 1000 / repeat
    print("latency:", latency, "ms")

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
    parser.add_argument("--model_file", type=str, default="", help="Model filename, Specify this when your model is a combined model.")
    parser.add_argument("--params_file", type=str, default="", help="Parameter filename, Specify this when your model is a combined model.")
    parser.add_argument("--model_dir", type=str, default="", help="Model dir, If you load a non-combined model, specify the directory of the model.")
    parser.add_argument("--int8", dest='use_int8', action='store_true', help="Use int8.")
    parser.add_argument("--float32", dest='use_int8', action='store_false', help="Use float32.")
    parser.set_defaults(use_int8=False)
    parser.add_argument("--min", type=int, default=3, help="min_subgraph_size for tensorrt") 
    return parser.parse_args()

def fake_input(shape):
    fake_img = np.ones(shape).astype(np.float32)
    return fake_img    

if __name__ == '__main__':
    args = parse_args()
    pred = init_predictor(args)
    input_shape = (1, 3, 608, 608) 
    fake_img = fake_input(input_shape)
    im_size = np.array([[608, 608]]).astype('int32')
    result = run(args, pred, [fake_img, im_size]) 
