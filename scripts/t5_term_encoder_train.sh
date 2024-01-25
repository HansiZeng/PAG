#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-term-encoder

# train 
queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/labeled_queries
teacher_score_path=/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-term-encoder/t5-term-encoder-0/out/MSMARCO_TRAIN/qrel_added_teacher_scores.json

docid_to_smtid_path=/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json

# need to change every time 
model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0-12l
pretrained_path=$model_dir/checkpoint_12l

output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/"

echo $teacher_score_path

for lr in 5e-4
run_name=t5-term-encoder-1-"$lr"-12l
do 
        #export CUDA_LAUNCH_BLOCKING=1
        python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
                --epochs=100 \
                --run_name=$run_name \
                --learning_rate=$lr \
                --loss_type=margin_mse \
                --model_name_or_path=$pretrained_path \
                --model_type=t5_term_encoder \
                --output_dir=$output_dir \
                --task_names='["rank"]' \
                --wandb_project_name=term_generative_retriever \
                --use_fp16 \
                --max_length=128 \
                --per_device_train_batch_size=32 \
                --queries_path=$queries_path \
                --pretrained_path=$pretrained_path \
                --teacher_score_path=$teacher_score_path \
                --save_steps=25_000 \
                --docid_to_smtid_path=$docid_to_smtid_path \
                --num_decoder_layers=12 
done
        