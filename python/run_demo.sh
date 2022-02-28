#!/bin/bash
set +x
set -e
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color

# tmp support demos
test_demos=(cuda_linux_demo ELMo resnet50 x86_linux_demo yolov3)

for demo in ${test_demos[@]};
do
    pushd $demo
    printf "${RED} run ${demo} ${NC}\n"
    bash run.sh
    popd
done
