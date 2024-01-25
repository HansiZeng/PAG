task=lexical_constrained_retrieve_and_rerank
experiment_dir=experiments-ripor-8-2048

# datasets for evaluation
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco
collection_path=$data_root_dir/full_collection/
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/qrel_binary.json"]'

if [ $task = "constrained_beam_search_for_qid_rankdata" ]; then 
    echo "task: $task"
    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/t5-dense-1"
    docid_to_tokenids_path=$data_dir/aq_smtid/docid_to_tokenids.json 

    model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/$experiment_dir/ripor_seq2seq_1
    pretrained_path=$model_dir/checkpoint

    # need to modify for a new experiment
    max_new_token=8
    topk=10

    out_dir=$model_dir/sub_tokenid_"${max_new_token}"_out_"${topk}"/

    #export CUDA_VISIBLE_DEVICES=4,5,6,7
    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.evaluate \
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

elif [ $task = "lexical_ripor_retrieve_and_rerank" ]; then
    echo "task: $task"
    experiment_dir=experiments-lexical-ripor-8-2048

    smt_data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/t5-dense-1"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-lexical-ripor-8-2048/lexical_ripor_seq2seq_1"
    pretrained_path=$model_dir/checkpoint/

    topk=1000
    out_dir=$model_dir/lex_ret_and_rerank/
    lex_out_dir=$model_dir/lex_ret 

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

elif [ $task = "lexical_constrained_retrieve_and_rerank" ]; then
    echo "task: $task"
    experiment_dir=experiments-lexical-ripor-8-2048

    smt_data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/t5-dense-1"
    smt_docid_to_smtid_path=$smt_data_dir/aq_smtid/docid_to_tokenids.json
    lex_data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0/"
    lex_docid_to_smtid_path=$lex_data_dir/top_bow/docid_to_tokenids.json

    # modify every time 
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-lexical-ripor-8-2048/lexical_ripor_seq2seq_1"
    pretrained_path=$model_dir/checkpoint/
    max_new_token_for_docid=8

    topk=100
    lex_topk=1000
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
        lex_out_dir=$model_dir/lex_ret_$lex_topk
        smt_out_dir=$model_dir/ltmp_smt_ret_"$topk"
        out_dir=$model_dir/ltmp_lex_smt_ret_rerank_"$topk"
    else 
        echo "Error: Unknown lexical_constrained $lexical_constrained"
        exit 1
    fi

    export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
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

    python -m torch.distributed.launch --nproc_per_node=6 -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --out_dir=$smt_out_dir \
        --lex_out_dir=$lex_out_dir \
        --task="$task"_2 \
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
        --task="$task"_3 \
        --out_dir=$smt_out_dir \
        --q_collection_paths=$q_collection_paths \
        --eval_qrel_path=$eval_qrel_path

    # for sanity check, let's rerank to smt_ret
    python -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir \
        --smt_out_dir=$smt_out_dir \
        --task=lexical_ripor_rerank \
        --q_collection_paths=$q_collection_paths \
        --batch_size=8 \
        --lex_docid_to_smtid_path=$lex_docid_to_smtid_path \
        --smt_docid_to_smtid_path=$smt_docid_to_smtid_path \
        --max_length=128 

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