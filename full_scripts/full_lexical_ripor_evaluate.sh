task=lexical_constrained_retrieve_and_rerank
experiment_dir=experiments-full-lexical-ripor

# datasets for evaluation
data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["./data/msmarco-full/TREC_DL_2019/queries_2019/","./data/msmarco-full/TREC_DL_2020/queries_2020/","./data/msmarco-full/dev_queries/"]'
eval_qrel_path='["./data/msmarco-full/dev_qrel.json","./data/msmarco-full/TREC_DL_2019/qrel.json","./data/msmarco-full/TREC_DL_2019/qrel_binary.json","./data/msmarco-full/TREC_DL_2020/qrel.json","./data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'

if [ $task = "constrained_beam_search_for_qid_rankdata" ]; then 
    echo "task: $task"
    data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    docid_to_tokenids_path=$data_dir/aq_smtid/docid_to_tokenids.json 

    model_dir=./data/$experiment_dir/ripor_direct_lng_knp_seq2seq_1
    pretrained_path=$model_dir/checkpoint

    # need to modify for a new experiment
    max_new_token=8
    topk=100

    out_dir=$model_dir/sub_tokenid_"${max_new_token}"_out_"${topk}"/

    #export CUDA_VISIBLE_DEVICES=4,5,6,7
    python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir \
        --task=constrained_beam_search_for_qid_rankdata \
        --docid_to_tokenids_path=$docid_to_tokenids_path \
        --q_collection_paths=$q_collection_paths \
        --batch_size=2 \
        --max_new_token_for_docid=$max_new_token \
        --topk=$topk

    python -m t5_pretrainer.evaluate \
        --task="$task"_2 \
        --out_dir=$out_dir \
        --q_collection_paths=$q_collection_paths \
        --eval_qrel_path=$eval_qrel_path 

elif [ $task = "constrained_beam_search_for_qid_rankdata_for_train" ]; then 
    echo "task: $task"
    data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    docid_to_tokenids_path=$data_dir/aq_smtid/docid_to_tokenids.json 

    model_dir=./data/$experiment_dir/ripor_direct_lng_knp_seq2seq_1
    pretrained_path=$model_dir/checkpoint

    # need to modify for a new experiment
    num_gpus=4
    max_new_token=8
    topk=100

    out_dir=$model_dir/sub_tokenid_"${max_new_token}"_out_"${topk}"/

    python -m torch.distributed.launch --nproc_per_node=$num_gpus -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir \
        --task=constrained_beam_search_for_qid_rankdata \
        --docid_to_tokenids_path=$docid_to_tokenids_path \
        --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
        --batch_size=16 \
        --max_new_token_for_docid=$max_new_token \
        --topk=$topk

    python -m t5_pretrainer.evaluate \
        --task=constrained_beam_search_for_qid_rankdata_2 \
        --out_dir=$out_dir \
        --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
        --eval_qrel_path=$eval_qrel_path

    # let's use teacher model to rerank 
    run_path=$out_dir/MSMARCO_TRAIN/run.json
    out_dir=$out_dir/MSMARCO_TRAIN/
    q_collection_path=./data/msmarco-full/all_train_queries/train_queries

    python -m torch.distributed.launch --nproc_per_node=$num_gpus -m t5_pretrainer.rerank \
        --task=rerank_for_create_trainset \
        --run_json_path=$run_path \
        --out_dir=$out_dir \
        --collection_path=$collection_path \
        --q_collection_path=$q_collection_path \
        --json_type=json \
        --batch_size=256

    python -m t5_pretrainer.rerank \
        --task=rerank_for_create_trainset_2 \
        --out_dir=$out_dir

elif [ $task = "constrained_beam_search_for_doc_ret_by_sub_tokens" ]; then 
    echo "task: $task"
    data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    docid_to_tokenids_path=$data_dir/aq_smtid/docid_to_tokenids.json 

    model_dir=./data/$experiment_dir/ripor_direct_lng_knp_seq2seq_1
    pretrained_path=$model_dir/checkpoint

    # need to modify for a new experiment
    for max_new_token in 2 4 6 8
    do 
        for topk in 10 50 100 200
        do

        out_dir=$model_dir/doc_ret_by_sub_tokens/ret_"$topk"_sub_"$max_new_token"/

        #export CUDA_VISIBLE_DEVICES=4,5,6,7
        #python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
        #    --pretrained_path=$pretrained_path \
        #    --out_dir=$out_dir \
        #    --task=constrained_beam_search_for_qid_rankdata \
        #    --docid_to_tokenids_path=$docid_to_tokenids_path \
        #    --q_collection_paths=$q_collection_paths \
        #    --batch_size=2 \
        #    --max_new_token_for_docid=$max_new_token \
        #    --topk=$topk

        python -m t5_pretrainer.evaluate \
            --task=constrained_beam_search_for_qid_rankdata_2 \
            --out_dir=$out_dir \
            --q_collection_paths=$q_collection_paths \
            --eval_qrel_path=$eval_qrel_path
        done 
    done
elif [ $task = "constrained_beam_search_for_qid_rankdata_sub_tokens" ]; then 
    echo "task: $task"
    data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    docid_to_tokenids_path=$data_dir/aq_smtid/docid_to_tokenids.json 

    model_dir=./data/$experiment_dir/ripor_direct_lng_knp_seq2seq_1
    pretrained_path=$model_dir/checkpoint

    # need to modify for a new experiment
    for max_new_token in 2 4 6 8
    do
        for topk in 10 50 100 200
        do

        if [ $max_new_token = 2 ]; then
            eval_qrel_path='["./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/dev_qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/TREC_DL_2019/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/TREC_DL_2019/qrel_binary.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/TREC_DL_2020/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/TREC_DL_2020/qrel_binary.json"]'
        elif [ $max_new_token = 4 ]; then 
            eval_qrel_path='["./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/dev_qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/TREC_DL_2019/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/TREC_DL_2019/qrel_binary.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/TREC_DL_2020/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/TREC_DL_2020/qrel_binary.json"]'
        elif [ $max_new_token = 6 ]; then 
            eval_qrel_path='["./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/dev_qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/TREC_DL_2019/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/TREC_DL_2019/qrel_binary.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/TREC_DL_2020/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/TREC_DL_2020/qrel_binary.json"]'
        elif [ $max_new_token = 8 ]; then 
            eval_qrel_path='["./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/dev_qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/TREC_DL_2019/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/TREC_DL_2019/qrel_binary.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/TREC_DL_2020/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/TREC_DL_2020/qrel_binary.json"]'
        else 
            echo "Error: Unknown task."
            exit 1
        fi

        out_dir=$model_dir/sub_tokens/ret_"$topk"_sub_"$max_new_token"/

        #python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
        #    --pretrained_path=$pretrained_path \
        #    --out_dir=$out_dir \
        #    --task=constrained_beam_search_for_qid_rankdata_sub_tokens \
        #    --docid_to_tokenids_path=$docid_to_tokenids_path \
        #    --q_collection_paths=$q_collection_paths \
        #    --batch_size=2 \
        #    --max_new_token_for_docid=$max_new_token \
        #    --topk=$topk

        python -m t5_pretrainer.evaluate \
            --task=constrained_beam_search_for_qid_rankdata_2 \
            --out_dir=$out_dir \
            --q_collection_paths=$q_collection_paths \
            --eval_qrel_path=$eval_qrel_path 
        done
    done

elif [ $task = "constrained_beam_search_for_train_queries" ]; then 
    echo "task: $task"
    data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    docid_to_tokenids_path=$data_dir/aq_smtid/docid_to_tokenids.json 

    model_dir=./data/$experiment_dir/ripor_seq2seq_1
    pretrained_path=$model_dir/checkpoint
    
    train_queries_path="./data/msmarco-full/all_train_queries/train_queries/raw.tsv"
    
    topk=100
    for max_new_token in 4 8
    do 
        out_dir=$model_dir/sub_smtid_"${max_new_token}"_out/

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
            --pretrained_path=$pretrained_path \
            --out_dir=$out_dir \
            --task=constrained_beam_search_for_qid_rankdata \
            --docid_to_tokenids_path=$docid_to_tokenids_path \
            --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
            --batch_size=2 \
            --max_new_token_for_docid=$max_new_token \
            --topk=$topk \
            --get_qid_smtid_rankdata
        
        python -m t5_pretrainer.evaluate \
            --task=constrained_beam_search_for_train_queries_2 \
            --out_dir=$out_dir 

        python t5_pretrainer/full_preprocess/from_qid_smtid_rank_to_qid_smtid_docids.py \
            --root_dir=$out_dir
    done 

    
    for max_new_token in 4 8
    do 
        qid_smtid_docids_path=$model_dir/sub_smtid_"$max_new_token"_out/qid_smtid_docids.train.json

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.rerank \
            --train_queries_path=$train_queries_path \
            --collection_path=$collection_path \
            --model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --max_length=256 \
            --batch_size=256 \
            --qid_smtid_docids_path=$qid_smtid_docids_path \
            --task=cross_encoder_rerank_for_qid_smtid_docids

        python -m t5_pretrainer.rerank \
            --out_dir=$model_dir/sub_smtid_"$max_new_token"_out \
            --task=cross_encoder_rerank_for_qid_smtid_docids_2
    done

elif [ $task = "lexical_ripor_retrieve_and_rerank" ]; then
    echo "task: $task"

    smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1_no_merge"
    pretrained_path=$model_dir/checkpoint/

    topk=1000
    out_dir=$model_dir/lex_ret_and_rerank/
    lex_out_dir=$model_dir/lex_ret 

    export CUDA_VISIBLE_DEVICES=2
    python -m torch.distributed.launch --nproc_per_node=1 -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir \
        --lex_out_dir=$lex_out_dir \
        --task=lexical_ripor_retrieve_and_rerank \
        --docid_to_tokenids_path=$docid_to_tokenids_path \
        --q_collection_paths=$q_collection_paths \
        --batch_size=8 \
        --topk=$topk \
        --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
        --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
        --max_length=128 
elif [ $task = "lexical_ripor_for_dense_pretrained_retrieve_and_rerank" ]; then
    echo "task: $task"

    #smt_data_dir="./data/experiments-full-lexical-ripor/lexical_ripor_dense_0_b"
    #smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_dense_0_b"
    pretrained_path=$model_dir/checkpoint/

    export CUDA_VISIBLE_DEVICES=2
    topk=1000
    out_dir=$model_dir/lex_ret_and_rerank/
    lex_out_dir=$model_dir/lex_ret 

    python -m torch.distributed.launch --nproc_per_node=1 -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir \
        --lex_out_dir=$lex_out_dir \
        --task=$task \
        --docid_to_tokenids_path=$docid_to_tokenids_path \
        --q_collection_paths=$q_collection_paths \
        --collection_path=$collection_path \
        --batch_size=8 \
        --topk=$topk \
        --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
        --max_length=128 



elif [ $task = "lexical_ripor_retrieve_and_rerank_for_train" ]; then
    echo "task: $task"

    smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1"
    pretrained_path=$model_dir/checkpoint/

    topk=100
    lex_out_dir=$model_dir/lex_ret

    python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --lex_out_dir=$lex_out_dir \
        --task=lexical_ripor_retrieve_parallel \
        --docid_to_tokenids_path=$docid_to_tokenids_path \
        --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
        --batch_size=8 \
        --topk=$topk \
        --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
        --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
        --max_length=128

    python -m t5_pretrainer.evaluate \
        --task=lexical_ripor_for_dense_pretrained_merge_runs \
        --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
        --out_dir=$lex_out_dir

    # let's use teacher model to rerank 
    run_path=$lex_out_dir/MSMARCO_TRAIN/run.json
    out_dir=$lex_out_dir/MSMARCO_TRAIN/
    q_collection_path=./data/msmarco-full/all_train_queries/train_queries

    python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.rerank \
        --task=rerank_for_create_trainset \
        --run_json_path=$run_path \
        --out_dir=$out_dir \
        --collection_path=$collection_path \
        --q_collection_path=$q_collection_path \
        --json_type=json \
        --batch_size=256

    python -m t5_pretrainer.rerank \
        --task=rerank_for_create_trainset_2 \
        --out_dir=$out_dir

elif [ $task = "lexical_constrained_retrieve_and_rerank" ]; then
    echo "task: $task"

    smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    # modify every time 
    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1"
    pretrained_path=$model_dir/checkpoint/
    max_new_token_for_docid=8

    for lex_topk in 1000
    do  
        lex_out_dir=$model_dir/all_lex_rets/lex_ret_$lex_topk
        python -m t5_pretrainer.evaluate \
            --pretrained_path=$pretrained_path \
            --out_dir=$out_dir \
            --lex_out_dir=$lex_out_dir \
            --task=$task \
            --docid_to_tokenids_path=$docid_to_tokenids_path \
            --q_collection_paths=$q_collection_paths \
            --batch_size=8 \
            --topk=$lex_topk \
            --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
            --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
            --max_length=128 \
            --eval_qrel_path=$eval_qrel_path

        for topk in 10 100
        do
        lexical_constrained=lexical_tmp_rescore
        if [ $lexical_constrained = lexical_condition ]; then 
            lex_out_dir=$model_dir/lex_ret_$lex_topk
            smt_out_dir=$model_dir/smt_ret_"$topk"
            out_dir=$model_dir/lex_smt_ret_rerank_"$topk"
        elif [ $lexical_constrained = lexical_incorporate ]; then 
            lex_out_dir=$model_dir/lex_ret_$lex_topk
            smt_out_dir=$model_dir/linc_smt_ret_"$topk"
            out_dir=$model_dir/linc_lex_smt_ret_rerank_"$topk"
        elif [ $lexical_constrained = lexical_tmp_rescore ]; then 
            smt_out_dir=$model_dir/all_lex_rets/lex_ret_"$lex_topk"/ltmp_smt_ret_"$topk"
        else 
            echo "Error: Unknown lexical_constrained $lexical_constrained"
            exit 1
        fi

        python -m torch.distributed.launch --nproc_per_node=1 -m t5_pretrainer.evaluate \
            --pretrained_path=$pretrained_path \
            --out_dir=$smt_out_dir \
            --lex_out_dir=$lex_out_dir \
            --task="$task"_2 \
            --docid_to_tokenids_path=$docid_to_tokenids_path \
            --q_collection_paths=$q_collection_paths \
            --batch_size=16 \
            --topk=$topk \
            --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
            --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
            --max_length=128  \
            --max_new_token_for_docid=$max_new_token_for_docid \
            --eval_qrel_path=$eval_qrel_path \
            --lex_constrained=$lexical_constrained

        python -m t5_pretrainer.evaluate \
            --task="$task"_3 \
            --out_dir=$smt_out_dir \
            --q_collection_paths=$q_collection_paths \
            --eval_qrel_path=$eval_qrel_path

        # for sanity check, let's rerank to smt_ret
        #python -m t5_pretrainer.evaluate \
        #    --pretrained_path=$pretrained_path \
        #    --out_dir=$out_dir \
        #    --smt_out_dir=$smt_out_dir \
        #    --task=lexical_ripor_rerank \
        #    --q_collection_paths=$q_collection_paths \
        #    --batch_size=8 \
        #    --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
        #    --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
        #    --max_length=128 
        done
    done

    #python -m t5_pretrainer.evaluate \
    #    --task="$task"_4 \
    #    --q_collection_paths=$q_collection_paths \
    #    --out_dir=$out_dir \
    #    --lex_out_dir=$lex_out_dir \
    #    --smt_out_dir=$smt_out_dir \
    #    --eval_qrel_path=$eval_qrel_path 

elif [ $task = "lexical_constrained_retrieve_and_rerank_diff_bow" ]; then
    echo "task: $task"

    bow_topk=128
    smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow_"$bow_topk"/docid_to_tokenids.json

    # modify every time 
    model_dir=./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1_bow_$bow_topk
    pretrained_path=$model_dir/checkpoint/
    max_new_token_for_docid=8

    for lex_topk in 1000
    do  
        lex_out_dir=$model_dir/all_lex_rets/lex_ret_$lex_topk
        python -m t5_pretrainer.evaluate \
            --pretrained_path=$pretrained_path \
            --out_dir=$out_dir \
            --lex_out_dir=$lex_out_dir \
            --task=lexical_constrained_retrieve_and_rerank \
            --docid_to_tokenids_path=$docid_to_tokenids_path \
            --q_collection_paths=$q_collection_paths \
            --batch_size=8 \
            --topk=$lex_topk \
            --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
            --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
            --max_length=128 \
            --eval_qrel_path=$eval_qrel_path
        
        for topk in 10 100
        do
        lexical_constrained=lexical_tmp_rescore
        if [ $lexical_constrained = lexical_tmp_rescore ]; then 
            smt_out_dir=$model_dir/all_lex_rets/lex_ret_"$lex_topk"/ltmp_smt_ret_"$topk"
        else 
            echo "Error: Unknown lexical_constrained $lexical_constrained"
            exit 1
        fi

        python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
            --pretrained_path=$pretrained_path \
            --out_dir=$smt_out_dir \
            --lex_out_dir=$lex_out_dir \
            --task=lexical_constrained_retrieve_and_rerank_2 \
            --docid_to_tokenids_path=$docid_to_tokenids_path \
            --q_collection_paths=$q_collection_paths \
            --batch_size=2 \
            --topk=$topk \
            --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
            --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
            --max_length=128  \
            --max_new_token_for_docid=$max_new_token_for_docid \
            --eval_qrel_path=$eval_qrel_path \
            --lex_constrained=$lexical_constrained

        python -m t5_pretrainer.evaluate \
            --task=lexical_constrained_retrieve_and_rerank_3 \
            --out_dir=$smt_out_dir \
            --q_collection_paths=$q_collection_paths \
            --eval_qrel_path=$eval_qrel_path

        # for sanity check, let's rerank to smt_ret
        #python -m t5_pretrainer.evaluate \
        #    --pretrained_path=$pretrained_path \
        #    --out_dir=$out_dir \
        #    --smt_out_dir=$smt_out_dir \
        #    --task=lexical_ripor_rerank \
        #    --q_collection_paths=$q_collection_paths \
        #    --batch_size=8 \
        #    --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
        #    --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
        #    --max_length=128 
        done
    done
elif [ $task = "lexical_constrained_retrieve_and_rerank_ret_doc_by_sub_tokens" ]; then 
    echo "task: $task"

    smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    # modify every time 
    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1"
    pretrained_path=$model_dir/checkpoint/

    lex_out_dir=$model_dir/lex_ret_1000

    for topk in 10 50 100 200
        do
            for max_new_token_for_docid in 2 4 6 8
            do

            lexical_constrained=lexical_tmp_rescore
            if [ $lexical_constrained = lexical_tmp_rescore ]; then 
                smt_out_dir=$model_dir/doc_ret_by_sub_tokens/ltmp_smt_ret_"$topk"_sub_"$max_new_token_for_docid"
            else 
                echo "Error: Unknown lexical_constrained $lexical_constrained"
                exit 1
            fi

            #python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
            #--pretrained_path=$pretrained_path \
            #--out_dir=$smt_out_dir \
            #--lex_out_dir=$lex_out_dir \
            #--task=lexical_constrained_retrieve_and_rerank_2 \
            #--docid_to_tokenids_path=$docid_to_tokenids_path \
            #--q_collection_paths=$q_collection_paths \
            #--batch_size=2 \
            #--topk=$topk \
            #--lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
            #--smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
            #--max_length=128  \
            #--max_new_token_for_docid=$max_new_token_for_docid \
            #--eval_qrel_path=$eval_qrel_path \
            #--lex_constrained=$lexical_constrained

            python -m t5_pretrainer.evaluate \
                --task=lexical_constrained_retrieve_and_rerank_3 \
                --out_dir=$smt_out_dir \
                --q_collection_paths=$q_collection_paths \
                --eval_qrel_path=$eval_qrel_path

            done 
        done 


elif [ $task = "lexical_constrained_retrieve_and_rerank_sub_token" ]; then
    echo "task: $task"

    smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    # modify every time 
    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1"
    pretrained_path=$model_dir/checkpoint/

    for lex_topk in 1000
    do  
        # lexical retrieval
        lex_out_dir=$model_dir/lex_ret_$lex_topk
        python -m t5_pretrainer.evaluate \
                --pretrained_path=$pretrained_path \
                --out_dir=$out_dir \
                --lex_out_dir=$lex_out_dir \
                --task=$task \
                --docid_to_tokenids_path=$docid_to_tokenids_path \
                --q_collection_paths=$q_collection_paths \
                --batch_size=8 \
                --topk=$lex_topk \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
                --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
                --max_length=128 \
                --eval_qrel_path=$eval_qrel_path

        for topk in 10 
        do
            for max_new_token_for_docid in 2 4 6 8
            do

            if [ $max_new_token_for_docid = 2 ]; then
                eval_qrel_path='["./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/dev_qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/TREC_DL_2019/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/TREC_DL_2019/qrel_binary.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/TREC_DL_2020/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_2/TREC_DL_2020/qrel_binary.json"]'
            elif [ $max_new_token_for_docid = 4 ]; then 
                eval_qrel_path='["./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/dev_qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/TREC_DL_2019/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/TREC_DL_2019/qrel_binary.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/TREC_DL_2020/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_4/TREC_DL_2020/qrel_binary.json"]'
            elif [ $max_new_token_for_docid = 6 ]; then 
                eval_qrel_path='["./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/dev_qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/TREC_DL_2019/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/TREC_DL_2019/qrel_binary.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/TREC_DL_2020/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_6/TREC_DL_2020/qrel_binary.json"]'
            elif [ $max_new_token_for_docid = 8 ]; then 
                eval_qrel_path='["./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/dev_qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/TREC_DL_2019/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/TREC_DL_2019/qrel_binary.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/TREC_DL_2020/qrel.json","./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/msmarco-full/sub_token_8/TREC_DL_2020/qrel_binary.json"]'
            else 
                echo "Error: Unknown task."
                exit 1
            fi
        
            lexical_constrained=lexical_tmp_rescore
            if [ $lexical_constrained = lexical_tmp_rescore ]; then 
                smt_out_dir=$model_dir/lex_ret_"$lex_topk"/ltmp_smt_ret_"$topk"_sub_"$max_new_token_for_docid"
            else 
                echo "Error: Unknown lexical_constrained $lexical_constrained"
                exit 1
            fi

            python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
                --pretrained_path=$pretrained_path \
                --out_dir=$smt_out_dir \
                --lex_out_dir=$lex_out_dir \
                --task=lexical_constrained_retrieve_and_rerank_2_for_sub_tokens \
                --docid_to_tokenids_path=$docid_to_tokenids_path \
                --q_collection_paths=$q_collection_paths \
                --batch_size=2 \
                --topk=$topk \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
                --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
                --max_length=128  \
                --max_new_token_for_docid=$max_new_token_for_docid \
                --eval_qrel_path=$eval_qrel_path \
                --lex_constrained=$lexical_constrained

            python -m t5_pretrainer.evaluate \
                --task=lexical_constrained_retrieve_and_rerank_3 \
                --out_dir=$smt_out_dir \
                --q_collection_paths=$q_collection_paths \
                --eval_qrel_path=$eval_qrel_path
            done 
        done
    done
elif [ $task = "lexical_ripor_for_dense_pretrained_retrieve_and_rerank_for_train" ]; then 
    echo "task: $task"

    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_dense_1_b"
    pretrained_path=$model_dir/checkpoint/

    topk=100
    out_dir=$model_dir/lex_ret_and_rerank/
    lex_out_dir=$model_dir/lex_ret 

    q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]'

    python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --lex_out_dir=$lex_out_dir \
        --task=lexical_ripor_for_dense_pretrained_retrieve_and_rerank_1 \
        --q_collection_paths=$q_collection_paths \
        --batch_size=8 \
        --topk=$topk \
        --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
        --collection_path=$collection_path \
        --max_length=128 

    python -m t5_pretrainer.evaluate \
        --out_dir=$lex_out_dir \
        --task=lexical_ripor_for_dense_pretrained_merge_runs \
        --q_collection_paths=$q_collection_paths \
        --collection_path=$collection_path \
        --batch_size=8 \
        --eval_qrel_path=$eval_qrel_path

    # let's use teacher model to rerank 
    run_path=$lex_out_dir/MSMARCO_TRAIN/run.json
    out_dir=$lex_out_dir/MSMARCO_TRAIN/
    q_collection_path=./data/msmarco-full/all_train_queries/train_queries

    python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.rerank \
        --task=rerank_for_create_trainset \
        --run_json_path=$run_path \
        --out_dir=$out_dir \
        --collection_path=$collection_path \
        --q_collection_path=$q_collection_path \
        --json_type=json \
        --batch_size=256

    python -m t5_pretrainer.rerank \
        --task=rerank_for_create_trainset_2 \
        --out_dir=$out_dir

    #python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
    #    --pretrained_path=$pretrained_path \
    #    --out_dir=$out_dir \
    #    --lex_out_dir=$lex_out_dir \
    #    --task=lexical_ripor_for_dense_pretrained_retrieve_and_rerank_2 \
    #    --q_collection_paths=$q_collection_paths \
    #    --batch_size=8 \
    #    --topk=$topk \
    #    --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
    #    --collection_path=$collection_path \
    #    --max_length=128 

    #python -m t5_pretrainer.evaluate \
    #    --out_dir=$out_dir \
    #    --task=lexical_ripor_for_dense_pretrained_merge_runs \
    #    --q_collection_paths=$q_collection_paths \
    #    --collection_path=$collection_path \
    #    --batch_size=8 \
    #    --eval_qrel_path=$eval_qrel_path

    # start to rerank 

    
elif [ $task = "lexical_ripor_dense_pretrained_aq_pipeline" ]; then 
    echo "task: $task"

    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_dense_1_b"
    pretrained_path=$model_dir/checkpoint
    index_dir=$model_dir/aq_index
    mmap_dir=$model_dir/mmap
    out_dir=$model_dir/aq_out

    M=8
    nbits=11
    K=$((2 ** $nbits))
    echo M: $M nbits: $nbits K: $K
    echo $model_dir

    #python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
    #    --pretrained_path=$pretrained_path \
    #    --index_dir=$mmap_dir \
    #    --task=lexical_ripor_dense_index \
    #    --collection_path=$collection_path

    #python -m t5_pretrainer.evaluate \
    #    --task=mmap_2 \
    #    --index_dir=$mmap_dir \
    #    --mmap_dir=$mmap_dir

    #python -m t5_pretrainer.evaluate \
    #    --task=aq_index \
    #    --codebook_num=$M \
    #    --codebook_bits=$nbits \
    #    --index_dir=$index_dir \
    #    --mmap_dir=$mmap_dir

    #python -m t5_pretrainer.evaluate \
    #    --task=lexical_ripor_dense_aq_retrieve \
    #    --pretrained_path=$pretrained_path \
    #    --index_dir=$index_dir \
    #    --out_dir=$out_dir \
    #    --q_collection_paths=$q_collection_paths \
    #    --eval_qrel_path=$eval_qrel_path  \
    #    --mmap_dir=$mmap_dir

    #python t5_pretrainer/preprocess/create_customized_smtid_file.py \
    #    --model_dir=$model_dir \
    #    --M=$M \
    #    --bits=$nbits

    python -m t5_pretrainer.preprocess.change_customized_embed_layer \
        --model_dir=$model_dir \
        --K=$K
elif [ $task = "lexical_ripor_dense_pretrained_dense_pipeline" ]; then
    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_dense_1_b"
    pretrained_path=$model_dir/checkpoint
    index_dir=$model_dir/index
    out_dir=$model_dir/out
    topk=100

    q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]'

    #python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
    #    --pretrained_path=$pretrained_path \
    #    --index_dir=$index_dir \
    #    --task=lexical_ripor_dense_index \
    #    --collection_path=$collection_path

    #python -m t5_pretrainer.evaluate \
    #    --task=index_2 \
    #    --index_dir=$index_dir \
    #    --mmap_dir=$index_dir

    python -m t5_pretrainer.evaluate \
        --task=lexical_ripor_dense_retrieve \
        --pretrained_path=$pretrained_path \
        --index_dir=$index_dir \
        --out_dir=$out_dir \
        --q_collection_paths=$q_collection_paths \
        --eval_qrel_path=$eval_qrel_path \
        --topk=$topk
elif [ $task = "lexical_constrained_retrieve_and_rerank_diff_pooling" ]; then
    echo "task: $task"
    task=lexical_constrained_retrieve_and_rerank

    smt_data_dir="./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="./data/experiments-splade/t5-splade-0-12l/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    # modify every time 
    model_dir="./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1"
    pretrained_path=$model_dir/checkpoint/
    max_new_token_for_docid=8

    for lex_topk in 1000
    do  
        lex_out_dir=$model_dir/all_lex_rets/lex_ret_$lex_topk
        #python -m t5_pretrainer.evaluate \
        #    --pretrained_path=$pretrained_path \
        #    --out_dir=$out_dir \
        #    --lex_out_dir=$lex_out_dir \
        #    --task=$task \
        #    --docid_to_tokenids_path=$docid_to_tokenids_path \
        #    --q_collection_paths=$q_collection_paths \
        #    --batch_size=8 \
        #    --topk=$lex_topk \
        #    --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
        #    --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
        #    --max_length=128 \
        #    --eval_qrel_path=$eval_qrel_path
        for pooling in "mean" "min" "max"
        do
            for topk in 10 100
            do
            lexical_constrained=lexical_tmp_rescore
            if [ $lexical_constrained = lexical_condition ]; then 
                lex_out_dir=$model_dir/lex_ret_$lex_topk
                smt_out_dir=$model_dir/smt_ret_"$topk"
                out_dir=$model_dir/lex_smt_ret_rerank_"$topk"
            elif [ $lexical_constrained = lexical_incorporate ]; then 
                lex_out_dir=$model_dir/lex_ret_$lex_topk
                smt_out_dir=$model_dir/linc_smt_ret_"$topk"
                out_dir=$model_dir/linc_lex_smt_ret_rerank_"$topk"
            elif [ $lexical_constrained = lexical_tmp_rescore ]; then 
                smt_out_dir=$model_dir/all_lex_rets/lex_ret_"$lex_topk"/ltmp_smt_ret_"$topk"_"$pooling"
            else 
                echo "Error: Unknown lexical_constrained $lexical_constrained"
                exit 1
            fi

            python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
                --pretrained_path=$pretrained_path \
                --out_dir=$smt_out_dir \
                --lex_out_dir=$lex_out_dir \
                --task="$task"_2 \
                --docid_to_tokenids_path=$docid_to_tokenids_path \
                --q_collection_paths=$q_collection_paths \
                --batch_size=16 \
                --topk=$topk \
                --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
                --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
                --max_length=128  \
                --max_new_token_for_docid=$max_new_token_for_docid \
                --eval_qrel_path=$eval_qrel_path \
                --lex_constrained=$lexical_constrained \
                --pooling=$pooling

            python -m t5_pretrainer.evaluate \
                --task="$task"_3 \
                --out_dir=$smt_out_dir \
                --q_collection_paths=$q_collection_paths \
                --eval_qrel_path=$eval_qrel_path
            done
        done
    done

    #python -m t5_pretrainer.evaluate \
    #    --task="$task"_4 \
    #    --q_collection_paths=$q_collection_paths \
    #    --out_dir=$out_dir \
    #    --lex_out_dir=$lex_out_dir \
    #    --smt_out_dir=$smt_out_dir \
    #    --eval_qrel_path=$eval_qrel_path 
else 
    echo "Error: Unknown task."
    exit 1
fi