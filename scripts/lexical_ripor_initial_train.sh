#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-lexical-ripor-8-2048


# seq2seq_1
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco
collection_path=$data_root_dir/full_collection/
queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/labeled_queries

smt_data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/t5-dense-1"
smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
lex_data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0/"
lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

# need to change for every experiment
model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/t5-dense-1"
pretrained_path=$model_dir/merged_extended_token_checkpoint/


output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/"
run_name=lexical_ripor_seq2seq_1

# also need to be changed by condition
teacher_score_path=/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/t5-dense-1/out/MSMARCO_TRAIN/qrel_added_merged_teacher_scores.json

python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --epochs=150 \
        --run_name=$run_name \
        --learning_rate=1e-4 \
        --loss_type=margin_mse \
        --model_name_or_path=t5-base \
        --model_type=lexical_ripor \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names='["rank","lexical_rank"]' \
        --wandb_project_name=term_generative_retriever \
        --use_fp16 \
        --collection_path=$collection_path \
        --max_length=64 \
        --per_device_train_batch_size=128 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --num_decoder_layers=1 \
        --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
        --lex_docid_to_smtid_path=$lex_docid_to_smtid_path