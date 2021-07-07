python2 my_reader.py \
    --batch_size 1 \
    --word_emb_dim 128 \
    --grnn_hidden_dim 128 \
    --bigru_num 2 \
    --test_data ./data/test_8000_bk.tsv \
    --word_dict_path ./conf/word.dic \
    --label_dict_path ./conf/tag.dic \
    --word_rep_dict_path ./conf/q2b.dic
