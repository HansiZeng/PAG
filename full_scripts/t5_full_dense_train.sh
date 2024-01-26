#!/bin/bash

data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
queries_path=./data/msmarco-full/all_train_queries/train_queries
experiment_dir=experiments-full-lexical-ripor

finetune_step=bm25_neg

lr=5e-4
output_dir="./data/$experiment_dir/"
if [ finetune_step=bm25_neg ]; then 
        pretrained_path=t5-base
        teacher_score_path=./data/msmarco-full/bm25_run/qrel_added_qid_docids_teacher_scores.train.json
        run_name=t5-full-dense-0-"$lr"-12l
elif [ finetune_step=self_neg ]; then 
        model_dir=./data/experiments-full-lexical-ripor/t5-full-dense-0-5e-4-12l/
        pretrained_path=$model_dir/checkpoint
        teacher_score_path=$model_dir/out/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json
        run_name=t5-full-dense-1-"$lr"-12l
else 
        echo "Error: Unknown task."
        exit 1
fi

python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.main \
        --epochs=50 \
        --run_name=$run_name \
        --learning_rate=$lr \
        --loss_type=margin_mse \
        --model_name_or_path=$pretrained_path \
        --model_type=t5_dense \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names='["rank"]' \
        --wandb_project_name=full_lexical_ripor \
        --use_fp16 \
        --collection_path=$collection_path \
        --max_length=128 \
        --per_device_train_batch_size=64 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --num_decoder_layers=12