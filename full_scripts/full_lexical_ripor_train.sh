#!/bin/bash
task=lexical_ripor_direct_lng_knp_margin_mse
experiment_dir=experiments-full-lexical-ripor

data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
queries_path=./data/msmarco-full/all_train_queries/train_queries

# seq2seq_1
if [ $task = lexical_ripor_margin_mse ]; then
        smt_data_dir="./data/$experiment_dir/lexical_ripor_dense_1_b"
        smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
        lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
        lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

        # need to change for every experiment
        model_dir="./data/$experiment_dir/lexical_ripor_dp_1b_seqseq_1"
        pretrained_path=$model_dir/checkpoint/

        output_dir="./data/$experiment_dir/"
        run_name=lexical_ripor_dp_1b_seqseq_1_self_mg

        # also need to be changed by condition
        teacher_score_path=$model_dir/lex_ret/MSMARCO_TRAIN/qrel_added_merged_teacher_scores.json

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.main \
                --epochs=120 \
                --run_name=$run_name \
                --learning_rate=5e-4 \
                --loss_type=margin_mse \
                --model_name_or_path=t5-base \
                --model_type=lexical_ripor \
                --teacher_score_path=$teacher_score_path \
                --output_dir=$output_dir \
                --task_names='["rank","lexical_rank"]' \
                --wandb_project_name=full_lexical_ripor \
                --use_fp16 \
                --collection_path=$collection_path \
                --max_length=64 \
                --per_device_train_batch_size=128 \
                --queries_path=$queries_path \
                --pretrained_path=$pretrained_path \
                --docid_to_smtid_path=$docid_to_smtid_path \
                --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path
elif [ $task = lexical_ripor_direct_lng_knp_margin_mse ]; then
        smt_data_dir="./data/$experiment_dir/t5-full-dense-1-5e-4-12l"
        smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
        lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
        lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

        # need to change for every experiment
        model_dir="./data/$experiment_dir/ripor_direct_lng_knp_seq2seq_1"
        pretrained_path=$model_dir/merged_checkpoint/

        output_dir="./data/$experiment_dir/"
        run_name=lexical_ripor_direct_lng_knp_seq2seq_1_ppr_loss_drn

        # also need to be changed by condition
        teacher_score_path=$smt_data_dir/out/MSMARCO_TRAIN/qrel_added_merged_teacher_scores.json

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.main \
                --epochs=120 \
                --run_name=$run_name \
                --learning_rate=5e-4 \
                --loss_type=direct_lng_knp_margin_mse \
                --model_name_or_path=t5-base \
                --model_type=lexical_ripor \
                --teacher_score_path=$teacher_score_path \
                --output_dir=$output_dir \
                --task_names='["rank_4","rank","lexical_rank"]' \
                --wandb_project_name=full_lexical_ripor \
                --use_fp16 \
                --collection_path=$collection_path \
                --max_length=64 \
                --per_device_train_batch_size=128 \
                --queries_path=$queries_path \
                --pretrained_path=$pretrained_path \
                --docid_to_smtid_path=$docid_to_smtid_path \
                --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path

elif [ $task = lexical_ripor_dense_pretrained ]; then 
        lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
        lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

        # need to change for every experiment
        model_dir="./data/$experiment_dir/t5-term-encoder-1-5e-4-12l"
        pretrained_path=$model_dir/checkpoint/

        output_dir="./data/$experiment_dir/"
        run_name=lexical_ripor_dense_0_b

        # also need to be changed by condition
        teacher_score_path=./data/$experiment_dir/t5-term-encoder-1-5e-4-12l/out/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.main \
                --epochs=50 \
                --run_name=$run_name \
                --learning_rate=5e-4 \
                --loss_type="dense_pretrained_margin_mse" \
                --model_name_or_path=t5-base \
                --model_type=lexical_ripor \
                --teacher_score_path=$teacher_score_path \
                --output_dir=$output_dir \
                --task_names='["rank","lexical_rank","dense_rank"]' \
                --wandb_project_name=full_lexical_ripor \
                --use_fp16 \
                --collection_path=$collection_path \
                --max_length=96 \
                --per_device_train_batch_size=48 \
                --queries_path=$queries_path \
                --pretrained_path=$pretrained_path \
                --docid_to_smtid_path=$docid_to_smtid_path \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path 
elif [ $task = lexical_ripor_seq2seq_and_margin_mse ]; then 
        smt_data_dir="./data/$experiment_dir/lexical_ripor_dense_1_b"
        smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
        lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
        lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json
        
        output_dir="./data/$experiment_dir/"
        
        # seq2seq
        model_dir="./data/$experiment_dir/lexical_ripor_dense_1_b"
        pretrained_path=$model_dir/extended_token_checkpoint/

        query_to_docid_path=./data/msmarco-full/doc2query/query_to_docid.train.json
        run_name=lexical_ripor_dp_1b_seqseq_0

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.main \
                --max_steps=250_000 \
                --run_name=$run_name  \
                --learning_rate=1e-3 \
                --loss_type="seq2seq" \
                --model_name_or_path=t5-base \
                --model_type=lexical_ripor \
                --per_device_train_batch_size=180 \
                --pretrained_path=$pretrained_path \
                --query_to_docid_path=$query_to_docid_path \
                --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
                --output_dir=$output_dir \
                --save_steps=50_000 \
                --task_names='["lexical_rank","rank"]' \
                --wandb_project_name=full_lexical_ripor \
                --use_fp16 \
                --warmup_ratio=0.045


        # margin_mse 
        model_dir="./data/$experiment_dir/$run_name"
        pretrained_path=$model_dir/checkpoint/
        
        teacher_score_path=./data/experiments-full-lexical-ripor/lexical_ripor_dense_1_b/lex_ret/MSMARCO_TRAIN/qrel_added_merged_teacher_scores.json
        run_name=lexical_ripor_dp_1b_seqseq_1

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.main \
                --epochs=120 \
                --run_name=$run_name \
                --learning_rate=5e-4 \
                --loss_type=margin_mse \
                --model_name_or_path=t5-base \
                --model_type=lexical_ripor \
                --teacher_score_path=$teacher_score_path \
                --output_dir=$output_dir \
                --task_names='["rank","lexical_rank"]' \
                --wandb_project_name=full_lexical_ripor \
                --use_fp16 \
                --collection_path=$collection_path \
                --max_length=64 \
                --per_device_train_batch_size=128 \
                --queries_path=$queries_path \
                --pretrained_path=$pretrained_path \
                --docid_to_smtid_path=$docid_to_smtid_path \
                --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path

elif [ $task = lexical_ripor_direct_lng_knp_margin_mse_diff_bow ]; then
        bow_topk=128
        smt_data_dir="./data/$experiment_dir/t5-full-dense-1-5e-4-12l"
        smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
        lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
        lex_docid_to_smtid_path=$lex_data_dir/top_bow_"$bow_topk"/docid_to_tokenids.json

        # need to change for every experiment
        model_dir="./data/$experiment_dir/ripor_direct_lng_knp_seq2seq_1"
        pretrained_path=$model_dir/bow_"$bow_topk"_merged_checkpoint/

        output_dir="./data/$experiment_dir/"
        run_name=lexical_ripor_direct_lng_knp_seq2seq_1_bow_"$bow_topk"

        # also need to be changed by condition
        teacher_data_dir="./data/$experiment_dir/t5-term-encoder-1-bow-"$bow_topk"-12l"
        teacher_score_path=$teacher_data_dir/out/MSMARCO_TRAIN/qrel_added_merged_teacher_scores.json

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.main \
                --epochs=120 \
                --run_name=$run_name \
                --learning_rate=5e-4 \
                --loss_type=direct_lng_knp_margin_mse \
                --model_name_or_path=t5-base \
                --model_type=lexical_ripor \
                --teacher_score_path=$teacher_score_path \
                --output_dir=$output_dir \
                --task_names='["rank_4","rank","lexical_rank"]' \
                --wandb_project_name=full_lexical_ripor \
                --use_fp16 \
                --collection_path=$collection_path \
                --max_length=64 \
                --per_device_train_batch_size=128 \
                --queries_path=$queries_path \
                --pretrained_path=$pretrained_path \
                --docid_to_smtid_path=$docid_to_smtid_path \
                --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path

else 
        echo "Error: Unknown task."
        exit 1
fi