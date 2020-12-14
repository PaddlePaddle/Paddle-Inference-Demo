import numpy as np
import argparse
import cv2

from paddle.inference import Config
from paddle.inference import create_predictor
from img_preprocess import preprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
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
    parser.add_argument("--img_path", type=str, default="", help="Input image path.")
    parser.add_argument("--threads",
                        type=int,
                        default=1,
                        help="Whether use gpu.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    assert (args.model_dir != "") or \
            (args.model_file != "" and args.params_file != ""), \
            "Set model path error."
    assert args.img_path != "", "Set img_path error."
    
    # Init config
    if args.model_dir == "":
        config = Config(args.model_file, args.params_file)
    else:
        config = Config(args.model_dir)
    config.enable_use_gpu(500, 0)
    config.switch_ir_optim()
    config.enable_memory_optim()
    config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=AnalysisConfig.Precision.Float32,max_batch_size=1, min_subgraph_size=5, use_static=False, use_calib_mode=False)
        
    # Create predictor
    predictor = create_predictor(config)

    # Set input
    img = cv2.imread(args.img_path)
    img = preprocess(img)
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.reshape(img.shape)
    input_tensor.copy_from_cpu(img.copy())

    # Run
    predictor.run()

    # Set output
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])
    output_data = output_tensor.copy_to_cpu()
    
    print("Predict class index: ", np.argmax(output_data))
