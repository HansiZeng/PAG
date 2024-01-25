#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-full-lexical-ripor


# seq2seq_0
task=seq2seq
query_to_docid_path=./data/msmarco-full/doc2query/query_to_docid.train.json
data_dir="./data/$experiment_dir/t5-full-dense-1-5e-4-12l"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_tokenids.json
output_dir="./data/$experiment_dir/"

# need to change for every experiment
model_dir="./data/$experiment_dir/t5-full-dense-1-5e-4-12l"
pretrained_path=$model_dir/extended_token_checkpoint/
run_name=ripor_seq2seq_0

# train
python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --max_steps=250_000 \
        --run_name=$run_name  \
        --learning_rate=1e-3 \
        --loss_type=$task \
        --model_name_or_path=t5-base \
        --model_type=ripor \
        --per_device_train_batch_size=256 \
        --pretrained_path=$pretrained_path \
        --query_to_docid_path=$query_to_docid_path \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --output_dir=$output_dir \
        --save_steps=50_000 \
        --task_names='["rank"]' \
        --wandb_project_name=full_lexical_ripor \
        --use_fp16 \
        --warmup_ratio=0.045