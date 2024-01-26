#!/bin/bash

collection_path=./data/msmarco-full/full_collection/
experiment_dir=experiments-splade

export CUDA_VISIBLE_DEVICES=0
model_dir="./data/$experiment_dir/t5-splade-0-12l"
pretrained_path=$model_dir/checkpoint

out_dir=$model_dir/top_bow

python -m t5_pretrainer.evaluate \
    --task=spalde_get_bow_rep \
    --pretrained_path=$pretrained_path \
    --index_retrieve_batch_size=128 \
    --collection_path=$collection_path \
    --out_dir=$out_dir \
    --bow_topk=64