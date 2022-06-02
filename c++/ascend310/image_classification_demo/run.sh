#!/bin/bash

TARGET_OS=linux
TARGET_ABI=arm64

NNADAPTER_CONTEXT_PROPERTIES="null"

export SUBGRAPH_ONLINE_MODE=true

NNADAPTER_DEVICE_NAMES="huawei_ascend_npu" 

if [ "$NNADAPTER_DEVICE_NAMES" == "huawei_ascend_npu" ]; then
    HUAWEI_ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
    if [ "$TARGET_OS" == "linux" ]; then
      if [[ "$TARGET_ABI" != "arm64" && "$TARGET_ABI" != "amd64" ]]; then
        echo "Unknown OS $TARGET_OS, only supports 'arm64' or 'amd64' for Huawei Ascend NPU."
        exit -1
      fi
    else
      echo "Unknown OS $TARGET_OS, only supports 'linux' for Huawei Ascend NPU."
      exit -1
    fi
    NNADAPTER_CONTEXT_PROPERTIES="HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/stub:$HUAWEI_ASCEND_TOOLKIT_HOME/fwkacllib/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/opp/op_proto/built-in
    export PYTHONPATH=$PYTHONPATH:$HUAWEI_ASCEND_TOOLKIT_HOME/fwkacllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/pyACL/python/site-packages/acl
    export PATH=$PATH:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/ccec_compiler/bin:${HUAWEI_ASCEND_TOOLKIT_HOME}/acllib/bin:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/bin
    export ASCEND_AICPU_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME
    export ASCEND_OPP_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME/opp
    export TOOLCHAIN_HOME=$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    export ASCEND_GLOBAL_LOG_LEVEL=1
fi

work_path=$(dirname $(readlink -f $0))
LIB_DIR=${work_path}/../../lib/paddle_inference
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LIB_DIR}/third_party/install/lite/cxx/lib

./build/demo \
    --model_dir assets/models/mobilenet_v1_fp32_224 \
    --label_path assets/labels/synset_words.txt \
    --image_path assets/images/tabby_cat.raw \
    --nnadapter_device_names $NNADAPTER_DEVICE_NAMES \
    --nnadapter_context_properties $NNADAPTER_CONTEXT_PROPERTIES
