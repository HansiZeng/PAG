#!/bin/bash

task="all_pipeline"
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco
collection_path=$data_root_dir/full_collection/
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-term-encoder


# data_dir
docid_to_smtid_path="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0/top_bow/docid_to_tokenids.json"

if [ $task == all_pipeline ]; then 
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/t5-term-encoder-1"
    pretrained_path=$model_dir/checkpoint
    out_dir=$model_dir/out

    python -m t5_pretrainer.evaluate \
        --task=term_encoder_retrieve \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir \
        --q_collection_paths=$q_collection_paths \
        --max_length=128 \
        --topk=1000 \
        --eval_qrel_path=$eval_qrel_path \
        --docid_to_smtid_path=$docid_to_smtid_path
elif [ $task == retrieve_train_queries ]; then
    echo "run retrieve_train_queries task"
    # the model_dir should be changed every time
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/t5-term-encoder-0"
    pretrained_path=$model_dir/checkpoint
    out_dir=$model_dir/out

    python -m t5_pretrainer.evaluate \
        --task=term_encoder_retrieve \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir  \
        --q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/labeled_queries/"]' \
        --max_length=128 \
        --topk=100 \
        --eval_qrel_path=$eval_qrel_path \
        --docid_to_smtid_path=$docid_to_smtid_path
else 
    echo "Error: Unknown task."
    exit 1
fi