#!/bin/bash
set +x
set -e
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

work_path=$(dirname $(readlink -f $0))

# check paddle_inference exists
if [ ! -d "${work_path}/lib/paddle_inference" ]; then
  echo "Please download paddle_inference lib and move it in Paddle-Inference-Demo/lib"
  exit 1
fi

# find all cxx demos
declare -a test_demos
for dir in $(ls ${work_path})
do
    # test all demos
    if [ "${dir}" != 'lib' -a -d "${dir}" ]; then
        test_demos+=("${dir}")
    fi
done

# tmp support demos
test_demos=(yolov3 LIC2020 resnet50)

for demo in ${test_demos[@]};
do
    pushd $demo
    printf "${RED} run ${demo} ${NC}\n"
    bash run.sh
    popd
done
