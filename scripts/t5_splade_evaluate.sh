#!/bin/bash

data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["./data/msmarco/TREC_DL_2019/queries_2019/","./data/msmarco/TREC_DL_2020/queries_2020/","./data/msmarco/dev_queries/"]'
eval_qrel_path='["./data/msmarco/dev_qrel.json","./data/msmarco/TREC_DL_2019/qrel.json","./data/msmarco/TREC_DL_2019/qrel_binary.json","./data/msmarco/TREC_DL_2020/qrel.json","./data/msmarco/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-splade

model_dir="./data/$experiment_dir/t5-splade-0-12l"
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
#out_dir=$model_dir/out

export CUDA_VISIBLE_DEVICES=0
python -m t5_pretrainer.evaluate \
    --task=sparse_index \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --index_retrieve_batch_size=128 \
    --collection_path=$collection_path \
    --max_length=128

#python -m t5_pretrainer.evaluate \
#    --task=sparse_retrieve_and_evaluate \
#    --pretrained_path=$pretrained_path \
#    --index_dir=$index_dir \
#    --out_dir=$out_dir \
#    --q_collection_paths=$q_collection_paths \
#    --max_length=128 \
#    --topk=1000 \
#    --eval_qrel_path=$eval_qrel_path 