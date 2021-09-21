from paddle.fluid.core import AnalysisConfig, create_paddle_predictor
from paddle import fluid
import time
import numpy as np
import argparse
import os

from data_reader import get_data, get_data_with_ptq_warmup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, default='', help='A path to a model.')
    parser.add_argument('--data_path', type=str, default='', help='Data file.')
    parser.add_argument(
        '--warmup_iter',
        type=int,
        default=100,
        help='Number of the first iterations to skip in performance statistics.')
    parser.add_argument(
        '--use_analysis',
        type=bool,
        default=True,
        help='If True AnalysisConfig will be used, if False NativeConfig will be used.')
    parser.add_argument(
        '--num_threads', type=int, default=1, help='Number of threads.')
    parser.add_argument(
        '--mkldnn_cache_capacity',
        type=int,
        default=100,
        help='The default value in Python API is 15, which can slow down int8 models.'
    )
    parser.add_argument(
        '--ops_to_quantize',
        type=str,
        default='',
        help='A comma separated list of operators to quantize. Only quantizable operators are taken into account. If the option is not used, an attempt to quantize all quantizable operators will be made.'
    )
    parser.add_argument(
        '--use_ptq',
        type=bool,
        default=False,
        help='Set True to use post-training quantization. Dafault False'
    )
    return parser.parse_args()


def set_config(model_path):
    config = None
    if os.path.exists(os.path.join(model_path, '__model__')):
        config = AnalysisConfig(model_path)
    else:
        config = AnalysisConfig(model_path + '/model', model_path+'/params')
    if test_args.use_analysis:
        config.switch_ir_optim(True)
        config.enable_mkldnn()
        config.set_mkldnn_cache_capacity(test_args.mkldnn_cache_capacity)
        config.set_cpu_math_library_num_threads(test_args.num_threads)
    else:
        config.to_native_config()

    return config


def set_config_ptq(model_path,
                   warmup_data):
    config = None
    if os.path.exists(os.path.join(model_path, '__model__')):
        config = AnalysisConfig(model_path)
    else:
        config = AnalysisConfig(model_path + '/model', model_path+'/params')
    config.switch_ir_optim(True)
    # This pass must be added before fc_fuse_pass to work properly
    config.pass_builder().insert_pass(5, "fc_lstm_fuse_pass")

    config.enable_mkldnn()
    config.set_mkldnn_cache_capacity(test_args.mkldnn_cache_capacity)
    config.set_cpu_math_library_num_threads(test_args.num_threads)

    config.enable_quantizer()
    config.quantizer_config().set_quant_data(warmup_data)
    config.quantizer_config().set_quant_batch_size(1)
    ops_to_quantize = set()
    if len(test_args.ops_to_quantize) > 0:
        ops_to_quantize = set(test_args.ops_to_quantize.split(','))
    config.quantizer_config().set_enabled_op_types(ops_to_quantize)

    return config


def run_program(model_path, data_path):
    place = fluid.CPUPlace()
    inputs = []
    labels = []
    config = None
    if test_args.use_ptq:
        warmup_data, inputs, labels = get_data_with_ptq_warmup(
            data_path, place)
        config = set_config_ptq(model_path, warmup_data)
    else:
        inputs, labels = get_data(data_path, place)
        config = set_config(model_path)

    predictor = create_paddle_predictor(config)
    all_hz_num = 0
    ok_hz_num = 0
    all_ctc_num = 0
    ok_ctc_num = 0
    dataset_size = len(inputs)
    start = time.time()
    for i in range(dataset_size):
        if i == test_args.warmup_iter:
            start = time.time()
        hz_out, ctc_out = predictor.run([inputs[i]])
        np_hz_out = np.array(hz_out.data.float_data()).reshape(-1)
        np_ctc_out = np.array(ctc_out.data.int64_data()).reshape(-1)
        out_hz_label = np.argmax(np_hz_out)
        this_label = labels[i]
        this_label_data = np.array(
            this_label.data.int32_data()).reshape(-1)
        if this_label.shape[0] == 1:
            all_hz_num += 1
            best = this_label_data[0]
            if out_hz_label == best:
                ok_hz_num += 1
            if this_label_data[0] <= 6350:
                all_ctc_num += 1
                if np_ctc_out.shape[0] == 1 and np_ctc_out.all(
                ) == this_label_data.all():
                    ok_ctc_num += 1
        else:
            all_ctc_num += 1
            if np_ctc_out.shape[0] == this_label.shape[
                    0] and np_ctc_out.all() == this_label_data.all():
                ok_ctc_num += 1
        if all_ctc_num > 1000 or all_hz_num > 1000:
            break
    end = time.time()
    fps = (dataset_size - test_args.warmup_iter) / (end - start)
    hx_acc = ok_hz_num / all_hz_num
    ctc_acc = ok_ctc_num / all_ctc_num
    return hx_acc, ctc_acc, fps


def test_lstm_model():
    model_path = test_args.model_path
    assert model_path, 'The model path cannot be empty. Please, use the --model_path option.'
    data_path = test_args.data_path
    assert data_path, 'The dataset path cannot be empty. Please, use the --data_path option.'

    (hx_acc, ctc_acc, fps) = run_program(model_path, data_path)
    print("FPS {0}, HX_ACC {1}, CTC_ACC {2}".format(
        fps, hx_acc, ctc_acc))


if __name__ == "__main__":
    global test_args
    test_args = parse_args()
    test_lstm_model()
