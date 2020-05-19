set -eux

export TASK_DATA_PATH=./data/
export DEV_FILE=dev.json
export INFER_MODEL=./lic_model/
export TEST_SAVE=./data/
export PYTHONPATH=./ernie:${PYTHONPATH:-}
python -u ./ernie/predict.py \
                   --use_cuda true \
                   --do_test true \
                   --batch_size 64 \
                   --init_checkpoint ${INFER_MODEL} \
                   --num_labels 112 \
                   --label_map_config ${TASK_DATA_PATH}relation2label.json \
                   --spo_label_map_config ${TASK_DATA_PATH}label2relation.json \
                   --test_set ${TASK_DATA_PATH}${DEV_FILE} \
                   --test_save ${TEST_SAVE}infer_dev.json \
                   --vocab_path ${TASK_DATA_PATH}vocab.txt \
                   --use_fp16 false \
                   --max_seq_len 512 \
                   --skip_steps 10 \
                   --random_seed 1
