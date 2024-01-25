#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-dense

# train 
collection_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/full_collection/
queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/labeled_queries

output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/"

# make change every time 
#model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/t5-dense-0/"
#pretrained_path=$model_dir/checkpoint/
pretrained_path=t5-base
teacher_score_path="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/bm25_run/qrel_added_qid_docids_teacher_scores.train.json"

echo $teacher_score_path

lr=5e-4
run_name="t5-dense-0-"$lr"-12l"

python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --epochs=100 \
        --run_name=$run_name \
        --learning_rate=$lr \
        --loss_type=margin_mse \
        --model_name_or_path=t5-base \
        --model_type=t5_dense \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names='["rank"]' \
        --wandb_project_name=term_generative_retriever \
        --use_fp16 \
        --collection_path=$collection_path \
        --max_length=128 \
        --per_device_train_batch_size=64 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --num_decoder_layers=12