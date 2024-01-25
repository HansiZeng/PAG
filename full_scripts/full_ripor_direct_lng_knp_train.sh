# seq2seq_1
experiment_dir=experiments-full-lexical-ripor

data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
queries_path=./data/msmarco-full/all_train_queries/train_queries

data_dir="./data/$experiment_dir/t5-full-dense-1-5e-4-12l"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_tokenids.json
output_dir="./data/$experiment_dir/"

# need to change for every experiment
model_dir="./data/$experiment_dir/ripor_seq2seq_0"
pretrained_path=$model_dir/checkpoint/
run_name=ripor_direct_lng_knp_seq2seq_1

# also need to be changed by condition
teacher_score_path=$data_dir/out/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json

python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --epochs=150 \
        --run_name=$run_name \
        --learning_rate=5e-4 \
        --loss_type=direct_lng_knp_margin_mse \
        --model_name_or_path=t5-base \
        --model_type=ripor \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names='["rank","rank_4"]' \
        --wandb_project_name=full_lexical_ripor \
        --use_fp16 \
        --collection_path=$collection_path \
        --max_length=64 \
        --per_device_train_batch_size=128 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --docid_to_smtid_path=$docid_to_smtid_path