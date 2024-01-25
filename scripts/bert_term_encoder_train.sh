#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-term-encoder

# train 
queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/labeled_queries
teacher_score_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/hard_negatives_scores/qrel_added_teacher_scores.json

output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/"
run_name="bert-term-encoder-01-0"

# need to change every time 
model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/bert-splade-01-0
pretrained_path=$model_dir/checkpoint
docid_to_smtid_path=$model_dir/top_bow/docid_to_tokenids.json
full_rank_eval_qrel_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/val_retrieval/qrel.json


echo $teacher_score_path

max_steps=250_000
python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --run_name=$run_name \
        --learning_rate=1e-4 \
        --loss_type=margin_mse \
        --model_name_or_path=$pretrained_path \
        --model_type=bert_term_encoder \
        --output_dir=$output_dir \
        --task_names='["rank"]' \
        --wandb_project_name=term_generative_retriever \
        --use_fp16 \
        --max_length=256 \
        --per_device_train_batch_size=32 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --teacher_score_path=$teacher_score_path \
        --max_step=$max_steps \
        --save_steps=25_000 \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --full_rank_eval_qrel_path=$full_rank_eval_qrel_path