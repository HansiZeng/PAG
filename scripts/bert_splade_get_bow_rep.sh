#!/bin/bash

data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco
collection_path=$data_root_dir/full_collection/
experiment_dir=experiments-splade

export CUDA_VISIBLE_DEVICES=1b
model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/bert-splade-01-0"
pretrained_path=$model_dir/checkpoint

out_dir=$model_dir/top_bow

python -m t5_pretrainer.evaluate \
    --task=spalde_get_bow_rep \
    --pretrained_path=$pretrained_path \
    --index_retrieve_batch_size=128 \
    --collection_path=$collection_path \
    --out_dir=$out_dir