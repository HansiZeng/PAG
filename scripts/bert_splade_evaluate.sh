#!/bin/bash

data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco
collection_path=$data_root_dir/full_collection/
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-splade

export CUDA_VISIBLE_DEVICES=0
model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/bert-splade-1-0"
pretrained_path=$model_dir/checkpoint
index_dir=$model_dir/index
out_dir=$model_dir/out

#python -m t5_pretrainer.evaluate \
#    --task=sparse_index \
#    --pretrained_path=$pretrained_path \
#    --index_dir=$index_dir \
#    --index_retrieve_batch_size=128 \
#    --collection_path=$collection_path

python -m t5_pretrainer.evaluate \
    --task=sparse_retrieve_and_evaluate \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir \
    --q_collection_paths=$q_collection_paths \
    --max_length=128 \
    --topk=1000 \
    --eval_qrel_path=$eval_qrel_path 