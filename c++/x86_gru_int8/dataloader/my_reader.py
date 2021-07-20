import argparse
import numpy as np

import paddle.fluid as fluid

import utils
import creator
import reader

parser = argparse.ArgumentParser(__doc__)
# 1. model parameters
model_g = utils.ArgumentGroup(parser, "model", "model configuration")
model_g.add_arg("word_emb_dim", int, 128,
                "The dimension in which a word is embedded.")
model_g.add_arg("grnn_hidden_dim", int, 128,
                "The number of hidden nodes in the GRNN layer.")
model_g.add_arg("bigru_num", int, 2,
                "The number of bi_gru layers in the network.")
model_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")

# 2. data parameters
data_g = utils.ArgumentGroup(parser, "data", "data paths")
data_g.add_arg("word_dict_path", str, "./conf/word.dic",
               "The path of the word dictionary.")
data_g.add_arg("label_dict_path", str, "./conf/tag.dic",
               "The path of the label dictionary.")
data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic",
               "The path of the word replacement Dictionary.")
data_g.add_arg("test_data", str, "./data/test.tsv",
               "The folder where the training data is located.")
data_g.add_arg("save_bin_path", str, "./data/test_eval_1000.bin",
               "The converted binary file is located.")

data_g.add_arg(
    "batch_size", int, 200,
    "The number of sequences contained in a mini-batch, "
    "or the maximum number of tokens (include paddings) contained in a mini-batch."
)


def do_eval(args):
    words = fluid.data(
        name='words', shape=[None, 1], dtype='int64', lod_level=1)
    targets = fluid.data(
        name='targets', shape=[None, 1], dtype='int64', lod_level=1)
    dataset = reader.Dataset(args)
    pyreader = creator.create_pyreader(
        args,
        file_name=args.test_data,
        # feed_list = test_ret['feed_list'],
        feed_list=[words, targets],
        place=fluid.CPUPlace(),
        model='lac',
        reader=dataset,
        mode='test')
    lods = []
    words = []
    targets = []
    sum_words = 0
    sum_sentences = 0

    for data in pyreader():
        print(len(data[0]['words'].lod()[0]))
        print(data[0]['words'])
        new_lod = data[0]['words'].lod()[0][1]
        new_words = np.array(data[0]['words'])
        new_targets = np.array(data[0]['targets'])
        assert new_lod == len(new_words)
        assert new_lod == len(new_targets)
        lods.append(new_lod)
        words.extend(new_words.flatten())
        targets.extend(new_targets.flatten())
        sum_sentences = sum_sentences + 1
        sum_words = sum_words + new_lod
    file1 = open(args.save_bin_path, "w+b")
    file1.write(np.array(int(sum_sentences)).astype('int64').tobytes())
    file1.write(np.array(int(sum_words)).astype('int64').tobytes())
    file1.write(np.array(lods).astype('uint64').tobytes())
    file1.write(np.array(words).astype('int64').tobytes())
    file1.write(np.array(targets).astype('int64').tobytes())
    file1.close()
    print("SUCCESS!! Binary file saved at ",args.save_bin_path, )


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    args = parser.parse_args()
    do_eval(args)
