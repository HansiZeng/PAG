#!/bin/bash

task="retrieve_train_queries"
data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["./data/msmarco/TREC_DL_2019/queries_2019/","./data/msmarco/TREC_DL_2020/queries_2020/","./data/msmarco/dev_queries/"]'
eval_qrel_path='["./data/msmarco/dev_qrel.json","./data/msmarco/TREC_DL_2019/qrel.json","./data/msmarco/TREC_DL_2019/qrel_binary.json","./data/msmarco/TREC_DL_2020/qrel.json","./data/msmarco/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-lexical-ripor


# data_dir
docid_to_smtid_path=./data/experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json

if [ $task == all_pipeline ]; then 
    for bow_topk in 16 32 128
    do  
        # model path
        model_dir="./data/$experiment_dir/t5-term-encoder-1-bow-$bow_topk-12l"
        pretrained_path=$model_dir/checkpoint

        # docid_to_smtid
        splade_dir="./data/experiments-splade/t5-splade-0-12l/"
        docid_to_smtid_path=$splade_dir/top_bow_"$bow_topk"/docid_to_tokenids.json
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
    done 

elif [ $task == retrieve_train_queries ]; then
    echo "run retrieve_train_queries task"
    # the model_dir should be changed every time
    # model path
    model_dir="./data/$experiment_dir/t5-term-encoder-0-bow-12l"
    pretrained_path=$model_dir/checkpoint

    # docid_to_smtid
    splade_dir="./data/experiments-splade/t5-splade-0-12l/"
    docid_to_smtid_path=$splade_dir/top_bow/docid_to_tokenids.json

    out_dir=$model_dir/out

    python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
        --task=term_encoder_parallel_retrieve \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir  \
        --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
        --max_length=128 \
        --topk=100 \
        --eval_qrel_path=$eval_qrel_path \
        --docid_to_smtid_path=$docid_to_smtid_path

    python -m t5_pretrainer.evaluate \
        --task=term_encoder_parallel_retrieve_2 \
        --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
        --out_dir=$out_dir

    # let's re-rank the run path
    run_path=$model_dir/out/MSMARCO_TRAIN/run.json
    out_dir=$model_dir/out/MSMARCO_TRAIN/
    q_collection_path=./data/msmarco-full/all_train_queries/train_queries

    python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.rerank \
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
else 
    echo "Error: Unknown task."
    exit 1
fi