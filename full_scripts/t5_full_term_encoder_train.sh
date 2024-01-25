#!/bin/bash
task=t5seq_aq_encoder_seq2seq
finetune_step=bm25_neg

# train 
data_root_dir=./data/msmarco-full
queries_path=./data/msmarco-full/all_train_queries/train_queries

splade_dir="./data/experiments-splade/t5-splade-0-12l/"
docid_to_smtid_path=$splade_dir/top_bow/docid_to_tokenids.json

experiment_dir=experiments-full-lexical-ripor
output_dir="./data/$experiment_dir/"

if [ finetune_step = bm25_neg ]; then 
        model_dir=./data/experiments-splade/t5-splade-0-12l
        pretrained_path=$model_dir/checkpoint
        teacher_score_path=./data/msmarco-full/bm25_run/qrel_added_qid_docids_teacher_scores.train.json 
        run_name=t5-term-encoder-0-bow-12l
elif [ finetune_step = self_neg ]; then 
        model_dir=./data/experiments-full-lexical-ripor/t5-term-encoder-0-bow-12l
        pretrained_path=$model_dir/checkpoint
        teacher_score_path=$model_dir/out/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json 
        run_name=t5-term-encoder-1-bow-12l
else 
        echo "Error: Unknown task."
        exit 1
fi

python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --epochs=50 \
        --run_name=$run_name \
        --learning_rate=5e-4 \
        --loss_type=margin_mse \
        --model_name_or_path=$pretrained_path \
        --model_type=t5_term_encoder \
        --output_dir=$output_dir \
        --task_names='["rank"]' \
        --wandb_project_name=full_lexical_ripor \
        --use_fp16 \
        --max_length=128 \
        --per_device_train_batch_size=32 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --teacher_score_path=$teacher_score_path \
        --save_steps=25_000 \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --num_decoder_layers=12 
        