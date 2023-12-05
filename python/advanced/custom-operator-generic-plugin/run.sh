#!/bin/bash
set +x
set -e

# 1. 编译并安装自定义算子
python setup.py install

# 2. 组网并保存自定义算子测试模型
python build_test_model.py

# 3. 运行自定义算子测试
python custom_gap_op_test.py