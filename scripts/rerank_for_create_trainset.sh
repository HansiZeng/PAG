#!/bin/bash

data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco
collection_path=$data_root_dir/full_collection/

root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/
run_path=$root_dir/t5-dense-1/out/MSMARCO_TRAIN/run.json
out_dir=$root_dir/t5-dense-1/out/MSMARCO_TRAIN/
q_collection_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/labeled_queries/


python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.rerank \
    --task=rerank_for_create_trainset \
    --run_json_path=$run_path \
    --out_dir=$out_dir \
    --collection_path=$collection_path \
    --q_collection_path=$q_collection_path \
    --json_type=json \
    --batch_size=256

python -m t5_pretrainer.rerank \
    --task=rerank_for_create_trainset_2 \
    --out_dir=$out_dir
