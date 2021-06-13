# rm -rf build && mkdir build && cd build
# cmake .. -DPADDLE_LIB=$PADDLE_ROOT
# make -j 12
cd build
data_src=/home/li/repo/Paddle-Inference-Demo/c++/x86_gru_int8/lac_demo_file/lac_model_conf_data
model_path=/mnt/disk500/models/GRU/GRU_infer_model
#model_path=$data_src/lac_opt.nb
input_path=$data_src/test_1w.txt
label_path=$data_src/label_1w.txt
conf_path=$data_src/conf
test_num=10000

cgdb --args ./model_test --model_path=$model_path\
    --conf_path=$conf_path\
    --input_path=$input_path\
    --label_path=$label_path\
    --test_num=$test_num
