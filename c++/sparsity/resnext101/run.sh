#!/bin/bash
GLOG_v=4 ./build/trt_fp16_test --model_file models/ResNeXt101_32x4d_density/model --params_file models/ResNeXt101_32x4d_density/params >jzz.log 2>&1
