import numpy as np
import struct
from paddle import fluid


def get_data(data_path, place):
    inputs = []
    labels = []
    with open(data_path, 'rb') as in_f:
        while True:
            plen = in_f.read(4)
            if plen is None or len(plen) != 4:
                break
            alllen = struct.unpack('i', plen)[0]
            label_len = alllen & 0xFFFF
            seq_len = (alllen >> 16) & 0xFFFF
            label = in_f.read(4 * label_len)
            label = np.frombuffer(
                label, dtype=np.int32).reshape([len(label) // 4])
            feat = in_f.read(4 * seq_len * 8)
            feat = np.frombuffer(
                feat, dtype=np.float32).reshape([len(feat) // 4 // 8, 8])
            lod_feat = [feat.shape[0]]
            minputs = fluid.create_lod_tensor(feat, [lod_feat], place)
            infer_data = fluid.core.PaddleTensor()
            infer_data.lod = minputs.lod()
            infer_data.data = fluid.core.PaddleBuf(np.array(minputs))
            infer_data.shape = minputs.shape()
            infer_data.dtype = fluid.core.PaddleDType.FLOAT32
            infer_label = fluid.core.PaddleTensor()
            infer_label.data = fluid.core.PaddleBuf(np.array(label))
            infer_label.shape = label.shape
            infer_label.dtype = fluid.core.PaddleDType.INT32
            inputs.append(infer_data)
            labels.append(infer_label)
    return inputs, labels


def get_data_with_ptq_warmup(data_path, place, warmup_batch_size=1):
    all_inputs, all_labels = get_data(data_path, place)
    warmup_inputs = all_inputs[:warmup_batch_size]
    inputs = all_inputs[warmup_batch_size:]
    labels = all_labels[warmup_batch_size:]
    return warmup_inputs, inputs, labels
