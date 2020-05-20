from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import six
import logging
from io import open
import numpy as np
import json

import codecs
import paddle.fluid as fluid

import reader.task_reader as task_reader
from utils.args import print_arguments, check_cuda, prepare_logger
from finetune_args import parser

from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

args = parser.parse_args()
log = logging.getLogger()


def infer(args):
    spo_label_map = json.load(open(args.spo_label_map_config))
    config = AnalysisConfig(args.init_checkpoint + "/model",
                            args.init_checkpoint + "/params")
    # config.enable_use_gpu(100, 0)
    # if use zero copy, need set this option false
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(False)
    predictor = create_paddle_predictor(config)
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor0 = predictor.get_input_tensor(input_names[0])
    input_tensor1 = predictor.get_input_tensor(input_names[1])
    input_tensor2 = predictor.get_input_tensor(input_names[2])
    input_tensor3 = predictor.get_input_tensor(input_names[3])
    input_tensor4 = predictor.get_input_tensor(input_names[4])
    input_tensor5 = predictor.get_input_tensor(input_names[5])
    input_tensor6 = predictor.get_input_tensor(input_names[6])

    # prepare input.
    reader = task_reader.RelationExtractionMultiCLSReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        spo_label_map_config=args.spo_label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed,
        task_id=args.task_id,
        num_labels=args.num_labels)

    test_sets = args.test_set.split(',')
    save_dirs = args.test_save.split(',')
    assert len(test_sets) == len(
        save_dirs
    ), 'number of test_sets & test_save not match, got %d vs %d' % (
        len(test_sets), len(save_dirs))

    batch_size = args.batch_size if args.predict_batch_size is None else args.predict_batch_size

    res = []
    for test_f, save_f in zip(test_sets, save_dirs):
        examples = reader._read_json(test_f)
        data_generator = reader.data_generator(
            test_f, batch_size=batch_size, epoch=1, dev_count=1, shuffle=False)
        for x_list in data_generator():
            input_tensor0.copy_from_cpu(x_list[0].astype('int64'))
            input_tensor1.copy_from_cpu(x_list[1].astype('int64'))
            input_tensor2.copy_from_cpu(x_list[2].astype('int64'))
            input_tensor3.copy_from_cpu(x_list[4].astype('float32'))
            input_tensor4.copy_from_cpu(x_list[6].astype('int64'))
            input_tensor5.copy_from_cpu(x_list[8].astype('int64'))
            input_tensor6.copy_from_cpu(x_list[9].astype('int64'))

            predictor.zero_copy_run()

            logits = predictor.get_output_tensor(output_names[0])
            logits_tensor = logits.copy_to_cpu()
            tok_to_orig_start_index_list = predictor.get_output_tensor(
                output_names[1])
            tok_to_orig_end_index_list = predictor.get_output_tensor(
                output_names[2])
            logits_lod = logits.lod()
            tok_to_orig_start_index_list_lod = tok_to_orig_start_index_list.lod(
            )
            tok_to_orig_end_index_list_lod = tok_to_orig_end_index_list.lod()
            tok_to_orig_start_index_list = tok_to_orig_start_index_list.copy_to_cpu(
            ).flatten()
            tok_to_orig_end_index_list = tok_to_orig_end_index_list.copy_to_cpu(
            ).flatten()

            example_index_list = x_list[7]
            example_index_list = np.array(example_index_list).astype(
                int) - 100000
            # perform evaluation
            for i in range(len(logits_lod[0]) - 1):
                # prepare prediction results for each example
                example_index = example_index_list[i]
                example = examples[example_index]
                tok_to_orig_start_index = tok_to_orig_start_index_list[
                    tok_to_orig_start_index_list_lod[0][i]:
                    tok_to_orig_start_index_list_lod[0][i + 1] - 2]
                tok_to_orig_end_index = tok_to_orig_end_index_list[
                    tok_to_orig_end_index_list_lod[0][i]:
                    tok_to_orig_end_index_list_lod[0][i + 1] - 2]
                inference_tmp = logits_tensor[logits_lod[0][i]:logits_lod[0][
                    i + 1]]

                # some simple post process
                inference_tmp = post_process(inference_tmp)

                # logits -> classification results
                inference_tmp[inference_tmp >= 0.5] = 1
                inference_tmp[inference_tmp < 0.5] = 0
                predict_result = []
                for token in inference_tmp:
                    predict_result.append(np.argwhere(token == 1).tolist())
                # format prediction into spo, calculate metric
                formated_result = format_output(
                    example, predict_result, spo_label_map,
                    tok_to_orig_start_index, tok_to_orig_end_index)

                res.append(formated_result)

        save_dir = os.path.dirname(save_f)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    with codecs.open(save_f, 'w', 'utf-8') as f:
        for result in res:
            json_str = json.dumps(result, ensure_ascii=False)
            print(json_str)
            f.write(json_str)
            f.write('\n')


def post_process(inference):
    # this post process only brings limited improvements (less than 0.5 f1) in order to keep simplicity
    # to obtain better results, CRF is recommended
    reference = []
    for token in inference:
        token_ = token.copy()
        token_[token_ >= 0.5] = 1
        token_[token_ < 0.5] = 0
        reference.append(np.argwhere(token_ == 1))

    #  token was classified into conflict situation (both 'I' and 'B' tag)
    for i, token in enumerate(reference[:-1]):
        if [0] in token and len(token) >= 2:
            if [1] in reference[i + 1]:
                inference[i][0] = 0
            else:
                inference[i][2:] = 0

    #  token wasn't assigned any cls ('B', 'I', 'O' tag all zero)
    for i, token in enumerate(reference[:-1]):
        if len(token) == 0:
            if [1] in reference[i - 1] and [1] in reference[i + 1]:
                inference[i][1] = 1
            elif [1] in reference[i + 1]:
                inference[i][np.argmax(inference[i, 1:]) + 1] = 1

    #  handle with empty spo: to be implemented

    return inference


def format_output(example, predict_result, spo_label_map,
                  tok_to_orig_start_index, tok_to_orig_end_index):
    # format prediction into example-style output
    complex_relation_label = [8, 10, 26, 32, 46]
    complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]
    instance = {}
    predict_result = predict_result[1:len(predict_result) -
                                    1]  # remove [CLS] and [SEP]
    text_raw = example['text']

    flatten_predict = []
    for layer_1 in predict_result:
        for layer_2 in layer_1:
            flatten_predict.append(layer_2[0])

    subject_id_list = []
    for cls_label in list(set(flatten_predict)):
        if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predict:
            subject_id_list.append(cls_label)
    subject_id_list = list(set(subject_id_list))

    def find_entity(id_, predict_result):
        entity_list = []
        for i in range(len(predict_result)):
            if [id_] in predict_result[i]:
                j = 0
                while i + j + 1 < len(predict_result):
                    if [1] in predict_result[i + j + 1]:
                        j += 1
                    else:
                        break
                entity = ''.join(text_raw[tok_to_orig_start_index[i]:
                                          tok_to_orig_end_index[i + j] + 1])
                entity_list.append(entity)

        return list(set(entity_list))

    spo_list = []
    for id_ in subject_id_list:
        if id_ in complex_relation_affi_label:
            continue
        if id_ not in complex_relation_label:
            subjects = find_entity(id_, predict_result)
            objects = find_entity(id_ + 55, predict_result)
            for subject_ in subjects:
                for object_ in objects:
                    spo_list.append({
                        "predicate":
                        spo_label_map['predicate'][id_],
                        "object_type": {
                            '@value': spo_label_map['object_type'][id_]
                        },
                        'subject_type':
                        spo_label_map['subject_type'][id_],
                        "object": {
                            '@value': object_
                        },
                        "subject":
                        subject_
                    })
        else:
            #  traverse all complex relation and look through their corresponding affiliated objects
            subjects = find_entity(id_, predict_result)
            objects = find_entity(id_ + 55, predict_result)
            for subject_ in subjects:
                for object_ in objects:
                    object_dict = {'@value': object_}
                    object_type_dict = {
                        '@value':
                        spo_label_map['object_type'][id_].split('_')[0]
                    }

                    if id_ in [8, 10, 32, 46] and id_ + 1 in subject_id_list:
                        id_affi = id_ + 1
                        object_dict[spo_label_map['object_type']
                                    [id_affi].split('_')[1]] = find_entity(
                                        id_affi + 55, predict_result)[0]
                        object_type_dict[
                            spo_label_map['object_type'][id_affi].split('_')
                            [1]] = spo_label_map['object_type'][id_affi].split(
                                '_')[0]
                    elif id_ == 26:
                        for id_affi in [27, 28, 29]:
                            if id_affi in subject_id_list:
                                object_dict[spo_label_map['object_type'][id_affi].split('_')[1]] = \
                                find_entity(id_affi + 55, predict_result)[0]
                                object_type_dict[spo_label_map['object_type'][id_affi].split('_')[1]] = \
                                spo_label_map['object_type'][id_affi].split('_')[0]

                    spo_list.append({
                        "predicate":
                        spo_label_map['predicate'][id_],
                        "object_type":
                        object_type_dict,
                        "subject_type":
                        spo_label_map['subject_type'][id_],
                        "object":
                        object_dict,
                        "subject":
                        subject_
                    })

    instance['text'] = example['text']
    instance['spo_list'] = spo_list
    return instance


if __name__ == '__main__':
    prepare_logger(log)
    print_arguments(args)
    check_cuda(args.use_cuda)
    infer(args)
