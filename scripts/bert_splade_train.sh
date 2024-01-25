#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-splade

# train 
collection_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/full_collection/
queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/labeled_queries
teacher_score_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/hard_negatives_scores/qrel_added_teacher_scores.json

output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/"
pretrained_path="bert-base-uncased"
run_name="bert-splade-1-0"

# eval 
eval_collection_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/val_retrieval/collection
eval_queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/val_retrieval/queries
full_rank_eval_qrel_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/val_retrieval/qrel.json
full_rank_eval_topk=200
full_rank_index_dir=$output_dir/eval_index
full_rank_out_dir=$output_dir/eval_out/

echo $teacher_score_path

max_steps=250_000
python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --run_name=$run_name \
        --learning_rate=1e-4 \
        --loss_type=margin_mse \
        --model_name_or_path=bert-base-uncased \
        --model_type=bert_splade \
        --output_dir=$output_dir \
        --task_names='["rank","query_reg","doc_reg"]' \
        --wandb_project_name=term_generative_retriever \
        --use_fp16 \
        --max_length=256 \
        --per_device_train_batch_size=32 \
        --collection_path=$collection_path \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --teacher_score_path=$teacher_score_path \
        --max_step=$max_steps \
        --eval_collection_path=$eval_collection_path \
        --eval_queries_path=$eval_queries_path \
        --full_rank_eval_qrel_path=$full_rank_eval_qrel_path \
        --full_rank_eval_topk=$full_rank_eval_topk \
        --full_rank_index_dir=$full_rank_index_dir \
        --full_rank_out_dir=$full_rank_out_dir \
        --save_steps=25_000