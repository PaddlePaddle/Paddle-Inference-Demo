import numpy as np
import argparse
from threading import Thread

from paddle.inference import Config
from paddle.inference import create_predictor, PredictorPool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default=None,
        help=
        "Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
    parser.add_argument("--use_gpu",
                        type=int,
                        default=0,
                        help="Whether use gpu.")
    parser.add_argument("--thread_num", type=int, default=1, help="thread num")
    return parser.parse_args()


def init_predictors(args):
    if args.model_dir is not None:
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        # If not specific mkldnn, you can set the blas thread.
        # The thread num should not be greater than the number of cores in the CPU.
        config.set_cpu_math_library_num_threads(4)
        config.enable_mkldnn()

    predictors = PredictorPool(config, args.thread_num)
    return predictors


class WrapperThread(Thread):
    def __init__(self, func, args):
        super(WrapperThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


def run_model(predictor, blob):
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_handle = predictor.get_input_handle(name)
        input_handle.reshape(blob[i].shape)
        input_handle.copy_from_cpu(blob[i])

    # do the inference
    predictor.run()

    results = {}
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_handle = predictor.get_output_handle(name)
        output_data = output_handle.copy_to_cpu()
        results[name] = output_data

    return results


if __name__ == '__main__':
    args = parse_args()
    thread_num = args.thread_num
    predictors = init_predictors(args)
    blob = [np.ones((1, 3, 224, 224)).astype(np.float32)]

    threads = []
    for i in range(thread_num):
        t = WrapperThread(run_model, args=(predictors.retrive(i), blob))
        threads.append(t)
        t.start()

    for i in range(thread_num):
        t.join()

    for i in range(thread_num):
        for v in threads[i].get_result().values():
            print('thread:', i, ', out shape: ', v.shape)
