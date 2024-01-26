#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-splade

# train 
collection_path=./data/msmarco-full/full_collection/
queries_path=./data/msmarco/train_queries/labeled_queries
teacher_score_path=./data/msmarco/hard_negatives_scores/qrel_added_teacher_scores.json

output_dir="./data/$experiment_dir/"
pretrained_path="t5-base"
run_name="t5-splade-0-12l"

# eval 
#eval_collection_path=./data/msmarco/val_retrieval/collection
#eval_queries_path=./data/msmarco/val_retrieval/queries
#full_rank_eval_qrel_path=./data/msmarco/val_retrieval/qrel.json
#full_rank_eval_topk=200
#full_rank_index_dir=$output_dir/eval_index
#full_rank_out_dir=$output_dir/eval_out/

echo $teacher_score_path

python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --max_steps=200_000 \
        --run_name=$run_name \
        --learning_rate=5e-4 \
        --loss_type=margin_mse \
        --model_name_or_path=t5-base \
        --model_type=t5_splade \
        --output_dir=$output_dir \
        --task_names='["rank","query_reg","doc_reg"]' \
        --wandb_project_name=term_generative_retriever \
        --use_fp16 \
        --max_length=128 \
        --per_device_train_batch_size=32 \
        --collection_path=$collection_path \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --teacher_score_path=$teacher_score_path \
        --save_steps=25_000 \
        --num_decoder_layers=12 
        #--eval_collection_path=$eval_collection_path \
        #--eval_queries_path=$eval_queries_path \
        #--full_rank_eval_qrel_path=$full_rank_eval_qrel_path \
        #--full_rank_eval_topk=$full_rank_eval_topk \
        #--full_rank_index_dir=$full_rank_index_dir \
        #--full_rank_out_dir=$full_rank_out_dir \