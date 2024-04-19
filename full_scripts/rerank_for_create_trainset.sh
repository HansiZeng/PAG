#!/bin/bash

data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
#run_path="./data/msmarco-full/bm25_run/top100.marco.train.all.json"
#out_dir="./data/msmarco-full/bm25_run/"

root_dir="./data/experiments-full-lexical-ripor"

for experiment_dir in "t5-full-dense-0-5e-4-12l"
do
    run_path=$root_dir/$experiment_dir/out/MSMARCO_TRAIN/run.json
    out_dir=$root_dir/$experiment_dir/out/MSMARCO_TRAIN/
    q_collection_path=./data/msmarco-full/all_train_queries/train_queries


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
done