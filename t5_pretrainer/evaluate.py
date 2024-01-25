import os 
import json
import ujson
import time
import copy
import pickle

import faiss
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import HfArgumentParser
from transformers import T5ForConditionalGeneration
import torch
import numpy as np
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from .dataset.dataloader import (
   CollectionDataLoader,
   T5SpladeCollectionDataLoader,
   T5DenseCollectionDataLoader,
   CollectionDataLoaderForRiporGeneration,
   LexicalRiporRerankDataLoader,
   LexicalConditionCollectionDataLoader,
   LexicalRiporDensePretrainedRerankDataLoader,
   LexicalRiporDenseCollectionDataLoader,
)
from .dataset.dataset import (
    CollectionDatasetPreLoad,
    LexicalRiporRerankDataset,
    LexicalRiporDensePretrainedRerankDataset,
)

from .modeling.t5_term_encoder import BertSplade, T5Splade, T5DenseEncoder
from .modeling.t5_generative_retriever import Ripor, LexicalRipor, LexicalRiporForDensePretrained
from .tasks.evaluator import (
    SparseIndexing, 
    SparseRetrieval, 
    TermEncoderRetriever,
    DenseIndexing,
    DenseRetriever,
    AddictvieQuantizeIndexer
)

from .tasks.generation import (
    generate_for_constrained_prefix_beam_search, 
    generate_for_lexical_condition_beam_search,
    generate_for_lexical_inc_beam_search,
    generate_for_lex_tmp_rescore
)

from .utils.utils import get_dataset_name, convert_ptsmtids_to_strsmtid, from_qrel_to_qsmtid_rel, to_device, is_first_worker
from .utils.metrics import load_and_evaluate, load_and_evaluate_for_qid_smtid
from .arguments import ModelArguments, EvalArguments
from .utils.inverted_index import merge_inverted_indexes
from .utils.prefixer import Prefixer, BatchPrefixer, BatchPrefixerForLexInc
from .utils.sequence_rescorer import BatchLexicalReScorer, BatchLexTmpReScorer
from .utils.utils import get_qid_smtid_scores

def constrained_decode_doc(model, 
                            dataloader,
                            prefixer,
                            smtid_to_docids,
                            max_new_token,
                            device, 
                            out_dir,
                            local_rank,
                            topk=100,
                            get_qid_smtid_rankdata=False):
    
    qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}
            outputs = generate_for_constrained_prefix_beam_search(
                        model,
                        prefix_allowed_tokens_fn=prefixer,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk,
                    )
        batch_qids = batch["id"].cpu().tolist()
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            qid_to_rankdata[qid] = {}
            if get_qid_smtid_rankdata:
                for smtid, rel_score in zip(ranked_smtids, rel_scores):
                    qid_to_rankdata[qid][smtid] = {}
                    if smtid in smtid_to_docids:
                        for docid in smtid_to_docids[smtid]:
                            qid_to_rankdata[qid][smtid][docid] = rel_score * max_new_token
                    else:
                        print(f"smtid: {smtid} not in smtid_to_docid")
            else:
                for smtid, rel_score in zip(ranked_smtids, rel_scores):
                    if smtid not in smtid_to_docids:
                        #pass 
                        print(f"smtid: {smtid} not in smtid_to_docid")
                    else:
                        for docid in smtid_to_docids[smtid]:
                            qid_to_rankdata[qid][docid] = rel_score * max_new_token
    
    if get_qid_smtid_rankdata:
        out_path = os.path.join(out_dir, f"qid_smtid_rankdata_{local_rank}.json")
        with open(out_path, "w") as fout:
            ujson.dump(qid_to_rankdata, fout)
    else:
        out_path = os.path.join(out_dir, f"run_{local_rank}.json")
        with open(out_path, "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

def constrained_decode_sub_tokens(model, 
                            dataloader,
                            prefixer,
                            smtid_to_docids,
                            max_new_token,
                            device, 
                            out_dir,
                            local_rank,
                            topk=100,
                            get_qid_smtid_rankdata=False):
    
    qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}
            outputs = generate_for_constrained_prefix_beam_search(
                        model,
                        prefix_allowed_tokens_fn=prefixer,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk,
                    )
        batch_qids = batch["id"].cpu().tolist()
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            qid_to_rankdata[qid] = {}
            for smtid, rel_score in zip(ranked_smtids, rel_scores):
                if smtid not in smtid_to_docids:
                    #pass 
                    print(f"smtid: {smtid} not in smtid_to_docid")
                else:
                    qid_to_rankdata[qid][smtid] = rel_score * max_new_token
    
    if get_qid_smtid_rankdata:
        raise NotImplementedError
        out_path = os.path.join(out_dir, f"qid_smtid_rankdata_{local_rank}.json")
        with open(out_path, "w") as fout:
            ujson.dump(qid_to_rankdata, fout)
    else:
        out_path = os.path.join(out_dir, f"run_{local_rank}.json")
        with open(out_path, "w") as fout:
            ujson.dump(qid_to_rankdata, fout)


def lexical_condition_decode_doc(model, 
                            dataloader,
                            qid_to_rankdata,
                            docid_to_tokenids,
                            tokenizer,
                            smtid_to_docids,
                            max_new_token,
                            device, 
                            out_dir,
                            local_rank,
                            topk=100):
    
    out_qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        batch_qids = batch["id"].cpu().tolist()
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}

            batch_prefixer = BatchPrefixer(docid_to_tokenids=docid_to_tokenids,
                                           qid_to_rankdata=qid_to_rankdata,
                                           qids=batch_qids,
                                           tokenizer=tokenizer,
                                           apply_stats=False)
            outputs = generate_for_lexical_condition_beam_search(
                        model,
                        prefix_allowed_tokens_fn=batch_prefixer,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        decoder_input_ids=inputs["decoder_input_ids"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk,
                    )
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            out_qid_to_rankdata[qid] = {}
            for smtid, rel_score in zip(ranked_smtids, rel_scores):
                if smtid not in smtid_to_docids:
                    #pass 
                    print(f"smtid: {smtid} not in smtid_to_docid")
                else:
                    for docid in smtid_to_docids[smtid]:
                        out_qid_to_rankdata[qid][docid] = rel_score * max_new_token
                    
    out_path = os.path.join(out_dir, f"run_{local_rank}.json")
    with open(out_path, "w") as fout:
        ujson.dump(out_qid_to_rankdata, fout)

def lexical_inc_decode_doc(model, 
                            dataloader,
                            qid_to_rankdata,
                            docid_to_tokenids,
                            tokenizer,
                            smtid_to_docids,
                            max_new_token,
                            device, 
                            out_dir,
                            local_rank,
                            topk=100):
    
    out_qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        batch_qids = batch["id"].cpu().tolist()
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}

            batch_prefixer = BatchPrefixer(docid_to_tokenids=docid_to_tokenids,
                                           qid_to_rankdata=qid_to_rankdata,
                                           qids=batch_qids,
                                           tokenizer=tokenizer,
                                           apply_stats=False)
            batch_lex_rescorer = BatchLexicalReScorer(batch_prefixer, num_beams=topk)
            outputs = generate_for_lexical_inc_beam_search(
                        model,
                        prefix_allowed_tokens_fn=batch_prefixer,
                        lex_rescorer=batch_lex_rescorer,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        decoder_input_ids=inputs["decoder_input_ids"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk,
                    )
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            out_qid_to_rankdata[qid] = {}
            for smtid, rel_score in zip(ranked_smtids, rel_scores):
                if smtid not in smtid_to_docids:
                    #pass 
                    print(f"smtid: {smtid} not in smtid_to_docid")
                else:
                    for docid in smtid_to_docids[smtid]:
                        out_qid_to_rankdata[qid][docid] = rel_score * max_new_token
                    
    out_path = os.path.join(out_dir, f"run_{local_rank}.json")
    with open(out_path, "w") as fout:
        ujson.dump(out_qid_to_rankdata, fout)

def lexical_tmp_rescore_decode_doc(model, 
                            dataloader,
                            qid_to_rankdata,
                            docid_to_tokenids,
                            tokenizer,
                            smtid_to_docids,
                            max_new_token,
                            device, 
                            out_dir,
                            local_rank,
                            pooling="max",
                            topk=100):
    print("pooling: ", pooling)
    
    lex_qid_to_smtid_to_score = get_qid_smtid_scores(qid_to_rankdata, docid_to_tokenids)
    out_qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        batch_qids = batch["id"].cpu().tolist()
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}

            batch_prefixer = BatchPrefixerForLexInc(docid_to_tokenids=docid_to_tokenids,
                                           qid_to_rankdata=qid_to_rankdata,
                                           qids=batch_qids,
                                           tokenizer=tokenizer,
                                           pooling=pooling)
            batch_lex_rescorer = BatchLexTmpReScorer(batch_prefixer, num_beams=topk)
            outputs = generate_for_lex_tmp_rescore(
                        model,
                        prefix_allowed_tokens_fn=batch_prefixer,
                        logit_tmp_rescorer=batch_lex_rescorer,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        decoder_input_ids=inputs["decoder_input_ids"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk,
                    )
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            out_qid_to_rankdata[qid] = {}
            for smtid, rel_score in zip(ranked_smtids, rel_scores):
                if smtid not in smtid_to_docids:
                    #pass 
                    print(f"smtid: {smtid} not in smtid_to_docid")
                else:
                    for docid in smtid_to_docids[smtid]:
                        out_qid_to_rankdata[qid][docid] = rel_score + lex_qid_to_smtid_to_score[str(qid)][str(smtid)]
                        #out_qid_to_rankdata[qid][docid] = rel_score 
                    
    out_path = os.path.join(out_dir, f"run_{local_rank}.json")
    with open(out_path, "w") as fout:
        ujson.dump(out_qid_to_rankdata, fout)

def lexical_tmp_rescore_decode_sub_tokens(model, 
                            dataloader,
                            qid_to_rankdata,
                            docid_to_tokenids,
                            tokenizer,
                            smtid_to_docids,
                            max_new_token,
                            device, 
                            out_dir,
                            local_rank,
                            topk=100):
    
    lex_qid_to_smtid_to_score = get_qid_smtid_scores(qid_to_rankdata, docid_to_tokenids)
    out_qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        batch_qids = batch["id"].cpu().tolist()
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}

            batch_prefixer = BatchPrefixerForLexInc(docid_to_tokenids=docid_to_tokenids,
                                           qid_to_rankdata=qid_to_rankdata,
                                           qids=batch_qids,
                                           tokenizer=tokenizer)
            batch_lex_rescorer = BatchLexTmpReScorer(batch_prefixer, num_beams=topk)
            outputs = generate_for_lex_tmp_rescore(
                        model,
                        prefix_allowed_tokens_fn=batch_prefixer,
                        logit_tmp_rescorer=batch_lex_rescorer,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        decoder_input_ids=inputs["decoder_input_ids"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk,
                    )
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            out_qid_to_rankdata[qid] = {}
            for smtid, rel_score in zip(ranked_smtids, rel_scores):
                if smtid not in smtid_to_docids:
                    #pass 
                    print(f"smtid: {smtid} not in smtid_to_docid")
                else:
                    out_qid_to_rankdata[qid][smtid] = rel_score + lex_qid_to_smtid_to_score[str(qid)][str(smtid)]
                    
    out_path = os.path.join(out_dir, f"run_{local_rank}.json")
    with open(out_path, "w") as fout:
        ujson.dump(out_qid_to_rankdata, fout)

def ddp_setup():
    init_process_group(backend="nccl")

def multiprocess_sparse_index(rank, args):
    init_process_group(backend="nccl", rank=rank, world_size=args.world_size)
    assert args.world_size == torch.distributed.get_world_size()

    # read model_config first
    with open(os.path.join(args.pretrained_path, "config.json")) as fin:
        model_config = ujson.load(fin)
    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        model = T5Splade.from_pretrained(args.pretrained_path)
    elif model_config["architectures"][0] == "BertForMaskedLM":
        model = BertSplade.from_pretrained(args.pretrained_path)
    else:
        raise ValueError("model architecture: {} is not predefined".format(model_config["architectures"][0]))
    
    d_collection = CollectionDatasetPreLoad(data_dir=args.collection_path, id_style="row_id")
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=args.index_retrieve_batch_size,
                                        shuffle=False, num_workers=1,
                                        sampler=DistributedSampler(d_collection, shuffle=False))
        
    evaluator = SparseIndexing(model=model, config={"index_dir": args.index_dir + f"_{rank}"}, compute_stats=True, device=rank, )
    out = evaluator.multi_process_index(d_loader)

def sparse_index(args):
    # read model_config first
    with open(os.path.join(args.pretrained_path, "config.json")) as fin:
        model_config = ujson.load(fin)
    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        model = T5Splade.from_pretrained(args.pretrained_path)
    elif model_config["architectures"][0] == "BertForMaskedLM":
        model = BertSplade.from_pretrained(args.pretrained_path)
    else:
        raise ValueError("model architecture: {} is not predefined".format(model_config["architectures"][0]))
    
    d_collection = CollectionDatasetPreLoad(data_dir=args.collection_path, id_style="row_id")

    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        d_loader = T5SpladeCollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=args.index_retrieve_batch_size,
                                        shuffle=False, num_workers=1) 
    elif model_config["architectures"][0] == "BertForMaskedLM":
        d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=args.index_retrieve_batch_size,
                                        shuffle=False, num_workers=1)
    else:
        raise NotImplementedError
    
    config = {
        "index_dir": args.index_dir
    }
    evaluator = SparseIndexing(model=model, config=config, compute_stats=True, device=args.local_rank if args.local_rank != -1 else 0)
    evaluator.index(d_loader)

def sparse_retrieve_and_evaluate(args):
    # read model_config first
    with open(os.path.join(args.pretrained_path, "config.json")) as fin:
        model_config = ujson.load(fin)
    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        model = T5Splade.from_pretrained(args.pretrained_path)
    elif model_config["architectures"][0] == "BertForMaskedLM":
        model = BertSplade.from_pretrained(args.pretrained_path)
    else:
        raise ValueError("model architecture: {} is not predefined".format(model_config["architectures"][0]))
    
    # read q_collection paths
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    batch_size = 1
    config = {
        "index_dir": args.index_dir,
        "out_dir": args.out_dir
    }
    # NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)
    for data_dir in set(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        if model_config["architectures"][0] == "T5ForConditionalGeneration":
            q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=batch_size,
                                        shuffle=False, num_workers=1) 
        else:
            q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length, batch_size=batch_size,
                                            shuffle=False, num_workers=1)
        evaluator = SparseRetrieval(config=config, model=model, dataset_name=get_dataset_name(data_dir),
                                    compute_stats=True, dim_voc=model.output_dim, device=args.local_rank if args.local_rank != -1 else 0)
        evaluator.retrieve(q_loader, top_k=args.topk, threshold=args.splade_threshold)

    evaluate(args)

def evaluate(args):
    if len(args.eval_qrel_path) == 1:
        args.eval_qrel_path = ujson.loads(args.eval_qrel_path[0])
    eval_qrel_path = args.eval_qrel_path
    eval_metric = args.eval_metric
    out_dir = args.out_dir

    res_all_datasets = {}
    for i, (qrel_file_path, eval_metrics) in enumerate(zip(eval_qrel_path, eval_metric)):
        if qrel_file_path is not None:
            res = {}
            dataset_name = get_dataset_name(qrel_file_path)
            print(eval_metrics)
            for metric in eval_metrics:
                res.update(load_and_evaluate(qrel_file_path=qrel_file_path,
                                             run_file_path=os.path.join(out_dir, dataset_name, "run.json"),
                                             metric=metric))
            if dataset_name in res_all_datasets.keys():
                res_all_datasets[dataset_name].update(res)
            else:
                res_all_datasets[dataset_name] = res
            json.dump(res, open(os.path.join(out_dir, dataset_name, "perf.json"), "a"))
    json.dump(res_all_datasets, open(os.path.join(out_dir, "perf_all_datasets.json"), "a"))
    return res_all_datasets

def spalde_get_bow_rep(args):
    # read model_config first
    with open(os.path.join(args.pretrained_path, "config.json")) as fin:
        model_config = ujson.load(fin)
    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        model = T5Splade.from_pretrained(args.pretrained_path)
        print("Read Splade with T5 as the backbone")
    elif model_config["architectures"][0] == "BertForMaskedLM":
        model = BertSplade.from_pretrained(args.pretrained_path)
        print("Read Splade model with BERT as the backbone")
    else:
        raise ValueError("model architecture: {} is not predefined".format(model_config["architectures"][0]))
    
    d_collection = CollectionDatasetPreLoad(data_dir=args.collection_path, id_style="row_id")
    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        d_loader = T5SpladeCollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=args.index_retrieve_batch_size,
                                        shuffle=False, num_workers=1) 
    elif model_config["architectures"][0] == "BertForMaskedLM":
        d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=args.index_retrieve_batch_size,
                                        shuffle=False, num_workers=1)
    else:
        raise NotImplementedError
    
    device = "cuda:0"
    model.eval()
    model.to(device)
    topk = args.bow_topk
    docid_to_tokenids = {}
    for batch in tqdm(d_loader, total=len(d_loader)):
        docids = batch["id"].tolist()
        tokenized_inputs = {k: v.to(device) for k, v in batch.items() if k != "id"}

        with torch.no_grad():
            doc_rep = model.encode(**tokenized_inputs)
        values, indices = torch.sort(doc_rep, dim=-1, descending=True)
        top_values, top_indices = values[:, :topk].cpu().tolist(), indices[:, :topk].cpu().tolist()

        assert len(docids) == len(top_values) == len(top_indices)

        for i in range(len(docids)):
            docid = str(docids[i])
            token_ids = top_indices[i]
            docid_to_tokenids[docid] = token_ids

    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    with open(os.path.join(args.out_dir, "docid_to_tokenids.json"), "w") as fout:
        ujson.dump(docid_to_tokenids, fout)
    with open(os.path.join(args.out_dir, "meta_data.json"), "w") as fout:
        info = {"topk": topk}
        ujson.dump(info, fout)

        #for i in range(len(top_values)):
        #    print(f"{i}: ")
        #    decoded_pairs = [(reverse_voc[k], v) for k, v in zip(top_indices[i], top_values[i])]
        #    print(decoded_pairs)
    
def term_encoder_retrieve(args):
    # read model_config first
    with open(os.path.join(args.pretrained_path, "config.json")) as fin:
        model_config = ujson.load(fin)
    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        model = T5Splade.from_pretrained(args.pretrained_path)
        print("Read TermEncoder with T5 as the backbone")
    elif model_config["architectures"][0] == "BertForMaskedLM":
        model = BertSplade.from_pretrained(args.pretrained_path)
        print("Read TermEncoder with BERT as the backbone")
    else:
        raise ValueError("model architecture: {} is not predefined".format(model_config["architectures"][0]))
    
    device = "cuda:0"
    model.eval()
    model.to(device)


    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    with open(os.path.join(args.docid_to_smtid_path)) as fin:
        docid_to_smtids = ujson.load(fin)
    
    # read q_collection paths
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    retriever = TermEncoderRetriever(model, args)
    for data_dir in set(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        if model_config["architectures"][0] == "T5ForConditionalGeneration":
            q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                                max_length=args.max_length,
                                                batch_size=16,
                                                shuffle=False, num_workers=1) 
        else:
            q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length, batch_size=16,
                                            shuffle=False, num_workers=1)
        retriever.retrieve(q_loader, docid_to_smtids, args.topk, out_dir=os.path.join(args.out_dir, get_dataset_name(data_dir)))
    
    evaluate(args)

def term_encoder_parallel_retrieve(args):
    ddp_setup()
    # read model_config first
    with open(os.path.join(args.pretrained_path, "config.json")) as fin:
        model_config = ujson.load(fin)
    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        model = T5Splade.from_pretrained(args.pretrained_path)
        print("Read TermEncoder with T5 as the backbone")
    elif model_config["architectures"][0] == "BertForMaskedLM":
        model = BertSplade.from_pretrained(args.pretrained_path)
        print("Read TermEncoder with BERT as the backbone")
    else:
        raise ValueError("model architecture: {} is not predefined".format(model_config["architectures"][0]))
    
    device = args.local_rank
    model.eval()
    model.to(device)


    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    with open(os.path.join(args.docid_to_smtid_path)) as fin:
        docid_to_smtids = ujson.load(fin)
    
    # read q_collection paths
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    retriever = TermEncoderRetriever(model, args)
    for data_dir in set(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        if model_config["architectures"][0] == "T5ForConditionalGeneration":
            q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                                max_length=args.max_length,
                                                batch_size=16,
                                                shuffle=False, num_workers=1,
                                                sampler=DistributedSampler(q_collection, shuffle=False)) 
        else:
            q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length, batch_size=16,
                                            shuffle=False, num_workers=1,
                                            sampler=DistributedSampler(q_collection, shuffle=False))
        retriever.retrieve(q_loader, docid_to_smtids, args.topk, out_dir=os.path.join(args.out_dir, get_dataset_name(data_dir)),
                           run_name=f"run_{args.local_rank}.json")
    
    #evaluate(args)

def term_encoder_parallel_retrieve_2(args):
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
    for data_dir in args.q_collection_paths:
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))

        # remove old 
        if os.path.exists(os.path.join(out_dir, "run.json")):
            print("old run.json exisit.")
            os.remove(os.path.join(out_dir, "run.json"))
        
        # merge
        qid_to_rankdata = {}
        sub_paths = [p for p in os.listdir(out_dir) if "run" in p]
        assert len(sub_paths) == torch.cuda.device_count()
        for sub_path in sub_paths:
            with open(os.path.join(out_dir, sub_path)) as fin:
                sub_qid_to_rankdata = ujson.load(fin)
            if len(qid_to_rankdata) == 0:
                qid_to_rankdata.update(sub_qid_to_rankdata)
            else:
                for qid, rankdata in sub_qid_to_rankdata.items():
                    if qid not in qid_to_rankdata:
                        qid_to_rankdata[qid] = rankdata
                    else:
                        qid_to_rankdata[qid].update(rankdata)
        print("length of pids and avg rankdata length in qid_to_rankdata: {}, {}".format(
        len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

        for sub_path in sub_paths:
            sub_path = os.path.join(out_dir, sub_path)
            os.remove(sub_path)

def index(args):
    ddp_setup()

    # parallel initialiaztion
    assert args.local_rank != -1
    model = T5DenseEncoder.from_pretrained(args.pretrained_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    d_collection = CollectionDatasetPreLoad(data_dir=args.collection_path, id_style="row_id")
    d_loader = T5DenseCollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                    max_length=args.max_length,
                                    batch_size=args.index_retrieve_batch_size,
                                    num_workers=1,
                                sampler=DistributedSampler(d_collection, shuffle=False))
    evaluator = DenseIndexing(model=model, args=args)
    evaluator.store_embs(d_loader, args.local_rank, use_fp16=False)
    
    destroy_process_group()

def index_2(args):
    DenseIndexing.aggregate_embs_to_index(args.index_dir)

def lexical_ripor_dense_index(args):
    ddp_setup()

    assert args.local_rank != -1
    model = LexicalRiporForDensePretrained.from_pretrained(args.pretrained_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    d_collection = CollectionDatasetPreLoad(data_dir=args.collection_path, id_style="row_id")
    d_loader = LexicalRiporDenseCollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                    max_length=args.max_length,
                                    batch_size=args.index_retrieve_batch_size,
                                    num_workers=1,
                                sampler=DistributedSampler(d_collection, shuffle=False))
    evaluator = DenseIndexing(model=model, args=args)
    evaluator.store_embs(d_loader, args.local_rank, use_fp16=False)
    
    destroy_process_group()

def dense_retrieve(args):
    model = T5DenseEncoder.from_pretrained(args.pretrained_path)
    model.cuda()

    batch_size = 128

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
        
    for data_dir in args.q_collection_paths:
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5DenseCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
        evaluator = DenseRetriever(model=model, args=args, dataset_name=get_dataset_name(data_dir))
        evaluator.retrieve(q_loader, 
                        topk=args.topk)
    evaluate(args)

def lexical_ripor_dense_retrieve(args):
    model = LexicalRiporForDensePretrained.from_pretrained(args.pretrained_path)
    model.cuda()

    batch_size = 128

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
        
    for data_dir in args.q_collection_paths:
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = LexicalRiporDenseCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
        
        evaluator = DenseRetriever(model=model, args=args, dataset_name=get_dataset_name(data_dir))
        evaluator.retrieve(q_loader, 
                          topk=args.topk)

    evaluate(args)

def mmap_2(args):
    DenseIndexing.aggregate_embs_to_mmap(args.mmap_dir)

def aq_index(args):
    if not os.path.exists(args.index_dir):
        os.mkdir(args.index_dir)
    AddictvieQuantizeIndexer.index(args.mmap_dir, index_dir=args.index_dir, codebook_num=args.codebook_num,
                                   codebook_bits=args.codebook_bits)
    
def aq_retrieve(args):
    model = T5DenseEncoder.from_pretrained(args.pretrained_path)
    model.cuda()

    aq_indexer = AddictvieQuantizeIndexer(model, args)
    batch_size = 128

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
        
    for data_dir in args.q_collection_paths:
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5DenseCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
        aq_indexer.search(q_loader, 
                          topk=1000, 
                          index_path=os.path.join(args.index_dir, "model.index"),
                          out_dir=os.path.join(args.out_dir, get_dataset_name(data_dir)),
                          index_ids_path=os.path.join(args.mmap_dir, "text_ids.tsv")
                          )

    evaluate(args)

def lexical_ripor_dense_aq_retrieve(args):
    model = LexicalRiporForDensePretrained.from_pretrained(args.pretrained_path)
    model.cuda()

    aq_indexer = AddictvieQuantizeIndexer(model, args)
    batch_size = 128

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
        
    for data_dir in args.q_collection_paths:
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = LexicalRiporDenseCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
        aq_indexer.search(q_loader, 
                          topk=1000, 
                          index_path=os.path.join(args.index_dir, "model.index"),
                          out_dir=os.path.join(args.out_dir, get_dataset_name(data_dir)),
                          index_ids_path=os.path.join(args.mmap_dir, "text_ids.tsv")
                          )

    evaluate(args)

def constrained_beam_search_for_qid_rankdata(args):
    ddp_setup()

    # read model 
    model = Ripor.from_pretrained(args.pretrained_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    # load prefixer 
    prefix_dir = os.path.dirname(args.docid_to_tokenids_path)
    if os.path.exists(os.path.join(prefix_dir, "prefix.pickle")):
        prefixer = Prefixer(docid_to_tokenids_path=None, tokenizer=None, prefix_path=os.path.join(prefix_dir, "prefix.pickle"))
    else:
        prefixer = Prefixer(docid_to_tokenids_path=args.docid_to_tokenids_path, tokenizer=tokenizer)

    #if not os.path.exists(args.out_dir):
    #os.mkdir(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # define parameters for decoding
    with open(args.docid_to_tokenids_path) as fin:
        docid_to_tokenids = ujson.load(fin)
    assert args.max_new_token_for_docid in [2, 4, 6, 8, 16, 32], args.max_new_token_for_docid
    smtid_to_docids = {}
    for docid, tokenids in docid_to_tokenids.items():
        assert tokenids[0] != -1, tokenids
        sid = "_".join([str(x) for x in tokenids[:args.max_new_token_for_docid]])
        if sid not in smtid_to_docids:
            smtid_to_docids[sid] = [docid]
        else:
            smtid_to_docids[sid] += [docid]

    max_new_token = args.max_new_token_for_docid
    assert len(sid.split("_")) == max_new_token, (sid, max_new_token)

    if args.local_rank <= 0:
        print("distribution of docids length per smtid: ", 
              np.quantile([len(xs) for xs in smtid_to_docids.values()], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        print("avergen length = {:.3f}".format(np.mean([len(xs) for xs in smtid_to_docids.values()])))
        print("smtid: ", sid)

    
    #if not os.path.exists(args.out_dir):
    #os.mkdir(args.out_dir)

    # start to retrieve
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    # retrieve sub_smtid only when we handle with train_query
    if len(args.q_collection_paths) == 1:
        assert not args.get_qid_smtid_rankdata 
    else:
        assert not args.get_qid_smtid_rankdata
    
    for data_dir in args.q_collection_paths:
        dev_dataset = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        dev_loader =  CollectionDataLoaderForRiporGeneration(dataset=dev_dataset, 
                                                    tokenizer_type=args.pretrained_path,
                                                    max_length=64,
                                                    batch_size=args.batch_size,
                                                    num_workers=1,
                                                    sampler=DistributedSampler(dev_dataset, shuffle=False))

        model.to(args.local_rank)
        if args.get_qid_smtid_rankdata:
            out_dir = args.out_dir
        else:
            out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        print("out_dir: ", out_dir)

        #if not os.path.exists(out_dir):
        #os.mkdir(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        model.base_model.return_logits = True
        constrained_decode_doc(model.base_model, 
                            dev_loader,
                            prefixer,
                            smtid_to_docids,
                            max_new_token,
                            device=args.local_rank, 
                            out_dir=out_dir,
                            local_rank=args.local_rank,
                            topk=args.topk,
                            get_qid_smtid_rankdata=args.get_qid_smtid_rankdata)

def constrained_beam_search_for_qid_rankdata_sub_tokens(args):
    ddp_setup()

    # read model 
    model = Ripor.from_pretrained(args.pretrained_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    # load prefixer 
    prefix_dir = os.path.dirname(args.docid_to_tokenids_path)
    if os.path.exists(os.path.join(prefix_dir, "prefix.pickle")):
        prefixer = Prefixer(docid_to_tokenids_path=None, tokenizer=None, prefix_path=os.path.join(prefix_dir, "prefix.pickle"))
    else:
        prefixer = Prefixer(docid_to_tokenids_path=args.docid_to_tokenids_path, tokenizer=tokenizer)

    os.makedirs(args.out_dir, exist_ok=True)
    
    # define parameters for decoding
    with open(args.docid_to_tokenids_path) as fin:
        docid_to_tokenids = ujson.load(fin)
    assert args.max_new_token_for_docid in [2, 4, 6, 8, 16, 32], args.max_new_token_for_docid
    smtid_to_docids = {}
    for docid, tokenids in docid_to_tokenids.items():
        assert tokenids[0] != -1, tokenids
        sid = "_".join([str(x) for x in tokenids[:args.max_new_token_for_docid]])
        if sid not in smtid_to_docids:
            smtid_to_docids[sid] = [docid]
        else:
            smtid_to_docids[sid] += [docid]

    max_new_token = args.max_new_token_for_docid
    assert len(sid.split("_")) == max_new_token, (sid, max_new_token)

    if args.local_rank <= 0:
        print("distribution of docids length per smtid: ", 
              np.quantile([len(xs) for xs in smtid_to_docids.values()], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        print("avergen length = {:.3f}".format(np.mean([len(xs) for xs in smtid_to_docids.values()])))
        print("smtid: ", sid)

    if args.local_rank <= 0:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

    # start to retrieve
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    # retrieve sub_smtid only when we handle with train_query
    if len(args.q_collection_paths) == 1:
        assert args.get_qid_smtid_rankdata 
    else:
        assert not args.get_qid_smtid_rankdata
    
    for data_dir in args.q_collection_paths:
        dev_dataset = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        dev_loader =  CollectionDataLoaderForRiporGeneration(dataset=dev_dataset, 
                                                    tokenizer_type=args.pretrained_path,
                                                    max_length=64,
                                                    batch_size=args.batch_size,
                                                    num_workers=1,
                                                    sampler=DistributedSampler(dev_dataset, shuffle=False))

        model.to(args.local_rank)
        if args.get_qid_smtid_rankdata:
            out_dir = args.out_dir
        else:
            out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        print("out_dir: ", out_dir, args.local_rank)
        
        os.makedirs(out_dir, exist_ok=True)

        model.base_model.return_logits = True
        constrained_decode_sub_tokens(model.base_model, 
                            dev_loader,
                            prefixer,
                            smtid_to_docids,
                            max_new_token,
                            device=args.local_rank, 
                            out_dir=out_dir,
                            local_rank=args.local_rank,
                            topk=args.topk,
                            get_qid_smtid_rankdata=args.get_qid_smtid_rankdata)

def constrained_beam_search_for_qid_rankdata_2(args):
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
    for data_dir in args.q_collection_paths:
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))

        # remove old 
        if os.path.exists(os.path.join(out_dir, "run.json")):
            print("old run.json exisit.")
            os.remove(os.path.join(out_dir, "run.json"))
        
        # merge
        qid_to_rankdata = {}
        sub_paths = [p for p in os.listdir(out_dir) if "run" in p]
        assert len(sub_paths) == torch.cuda.device_count()
        for sub_path in sub_paths:
            with open(os.path.join(out_dir, sub_path)) as fin:
                sub_qid_to_rankdata = ujson.load(fin)
            if len(qid_to_rankdata) == 0:
                qid_to_rankdata.update(sub_qid_to_rankdata)
            else:
                for qid, rankdata in sub_qid_to_rankdata.items():
                    if qid not in qid_to_rankdata:
                        qid_to_rankdata[qid] = rankdata
                    else:
                        qid_to_rankdata[qid].update(rankdata)
        print("length of pids and avg rankdata length in qid_to_rankdata: {}, {}".format(
        len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

        for sub_path in sub_paths:
            sub_path = os.path.join(out_dir, sub_path)
            os.remove(sub_path)

    evaluate(args)

def constrained_beam_search_for_train_queries_2(args):
    out_dir = args.out_dir

    # remove old 
    if os.path.exists(os.path.join(out_dir, "qid_smtid_rankdata.json")):
        print("old run.json exisit.")
        os.remove(os.path.join(out_dir, "qid_smtid_rankdata.json"))
    
    # merge
    qid_to_smtid_rankdata = {}
    sub_paths = [p for p in os.listdir(out_dir) if "qid_smtid_rankdata" in p]
    print("out_dir: ", [p for p in os.listdir(out_dir)], out_dir)
    assert len(sub_paths) == torch.cuda.device_count(), (len(sub_paths), torch.cuda.device_count())
    for sub_path in sub_paths:
        with open(os.path.join(out_dir, sub_path)) as fin:
            sub_qid_to_smtid_rankdata = ujson.load(fin)
        for qid in sub_qid_to_smtid_rankdata:
            if qid not in qid_to_smtid_rankdata:
                qid_to_smtid_rankdata[qid] = sub_qid_to_smtid_rankdata[qid]
            else:
                for smtid in sub_qid_to_smtid_rankdata[qid]:
                    if smtid not in qid_to_smtid_rankdata[qid]:
                        qid_to_smtid_rankdata[qid][smtid] = sub_qid_to_smtid_rankdata[qid][smtid]
                    else:
                        for docid, score in sub_qid_to_smtid_rankdata[qid][smtid].items():
                            qid_to_smtid_rankdata[qid][smtid][docid] = score
    
    smtid_lengths = []
    doc_lenghts = []
    for qid in qid_to_smtid_rankdata:
        smtid_lengths.append(len(qid_to_smtid_rankdata[qid]))
        for smtid in qid_to_smtid_rankdata[qid]:
            doc_lenghts.append(len(qid_to_smtid_rankdata[qid][smtid]))
    
    print("smtid_length per query: ", np.quantile(smtid_lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    print("doc_length per smtid: ", np.quantile(doc_lenghts, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))


    with open(os.path.join(out_dir, "qid_smtid_rankdata.json"), "w") as fout:
        ujson.dump(qid_to_smtid_rankdata, fout)

    for sub_path in sub_paths:
        sub_path = os.path.join(out_dir, sub_path)
        os.remove(sub_path)

def lexical_ripor_retrieve_and_rerank(args):
    # initialize model
    model = LexicalRipor.from_pretrained(args.pretrained_path)
    model.eval()
    device = "cuda:0"
    model.to(device)
    model.base_model.mode = "lex_retrieval"
    # retrive top k docids using lexical semantic ids
    if not os.path.exists(args.lex_out_dir):
        os.mkdir(args.lex_out_dir)
    with open(os.path.join(args.lex_docid_to_smtid_path)) as fin:
        lex_docid_to_smtids = ujson.load(fin)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    retriever = TermEncoderRetriever(model, args)
    for data_dir in set(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1) 
        retriever.retrieve(q_loader, lex_docid_to_smtids, args.topk, out_dir=os.path.join(args.lex_out_dir, get_dataset_name(data_dir)))

    # hack 
    rerank_out_dir = args.out_dir
    args.out_dir = args.lex_out_dir
    #evaluate(args)
    args.out_dir = rerank_out_dir
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # start to reranking 
    model.base_model.mode = "lex_smt_outputs"
    for data_dir in set(args.q_collection_paths):
        run_path = os.path.join(args.lex_out_dir, get_dataset_name(data_dir), "run.json")
        dataset = LexicalRiporRerankDataset(run_path, query_dir=data_dir, smt_docid_to_smtid_path=args.smt_docid_to_smtid_path,
                                            lex_docid_to_smtid_path=args.lex_docid_to_smtid_path)
        dataloader = LexicalRiporRerankDataLoader(dataset=dataset, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=64,
                                            shuffle=False, num_workers=1)

        qid_to_docid_to_score = {}
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = to_device(batch, device)
            with torch.no_grad():
                scores = model.rerank_forward(**batch)["score"].cpu().tolist()
            
            qids = batch["qid"].cpu().tolist()
            docids = batch["docid"].cpu().tolist()

            for qid, docid, score in zip(qids, docids, scores):
                qid, docid = str(qid), str(docid)
                if qid not in qid_to_docid_to_score:
                    qid_to_docid_to_score[qid] = {docid: score}
                else:
                    qid_to_docid_to_score[qid][docid] = score

        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_docid_to_score, fout)
        
    evaluate(args)

def lexical_ripor_retrieve_parallel(args):
    ddp_setup()

    # initialize model
    model = LexicalRipor.from_pretrained(args.pretrained_path)
    model.eval()
    device = args.local_rank
    model.to(device)
    model.base_model.mode = "lex_retrieval"

    # retrive top k docids using lexical semantic ids
    if is_first_worker():
        if not os.path.exists(args.lex_out_dir):
            os.mkdir(args.lex_out_dir)
    with open(os.path.join(args.lex_docid_to_smtid_path)) as fin:
        lex_docid_to_smtids = ujson.load(fin)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    retriever = TermEncoderRetriever(model, args)
    for data_dir in set(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1,
                                            sampler=DistributedSampler(q_collection, shuffle=False)) 
        retriever.retrieve(q_loader, lex_docid_to_smtids, args.topk, out_dir=os.path.join(args.lex_out_dir, get_dataset_name(data_dir)),
                           run_name=f"run_{args.local_rank}.json")


def lexical_ripor_for_dense_pretrained_retrieve_and_rerank(args):
    # initialize model
    model = LexicalRiporForDensePretrained.from_pretrained(args.pretrained_path)
    model.eval()
    device = "cuda:0"
    model.to(device)
    model.base_model.mode = "lex_retrieval"
    # retrive top k docids using lexical semantic ids
    if not os.path.exists(args.lex_out_dir):
        os.mkdir(args.lex_out_dir)
    with open(os.path.join(args.lex_docid_to_smtid_path)) as fin:
        lex_docid_to_smtids = ujson.load(fin)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
    
    retriever = TermEncoderRetriever(model, args)
    for data_dir in set(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1) 
        retriever.retrieve(q_loader, lex_docid_to_smtids, args.topk, out_dir=os.path.join(args.lex_out_dir, get_dataset_name(data_dir)))

    # hack 
    rerank_out_dir = args.out_dir
    args.out_dir = args.lex_out_dir
    evaluate(args)
    args.out_dir = rerank_out_dir
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # start to reranking 
    model.base_model.mode = "lex_smt_outputs"
    for data_dir in set(args.q_collection_paths):
        run_path = os.path.join(args.lex_out_dir, get_dataset_name(data_dir), "run.json")
        dataset = LexicalRiporDensePretrainedRerankDataset(run_path, query_dir=data_dir, 
                                                           document_dir=args.collection_path,
                                                            lex_docid_to_smtid_path=args.lex_docid_to_smtid_path)
        dataloader = LexicalRiporDensePretrainedRerankDataLoader(dataset=dataset, tokenizer_type=args.pretrained_path,
                                                            max_length=args.max_length,
                                                            batch_size=64,
                                                            shuffle=False, num_workers=1)

        qid_to_docid_to_score = {}
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = to_device(batch, device)
            with torch.no_grad():
                scores = model.rerank_forward(**batch)["score"].cpu().tolist()
            
            qids = batch["qid"].cpu().tolist()
            docids = batch["docid"].cpu().tolist()

            for qid, docid, score in zip(qids, docids, scores):
                qid, docid = str(qid), str(docid)
                if qid not in qid_to_docid_to_score:
                    qid_to_docid_to_score[qid] = {docid: score}
                else:
                    qid_to_docid_to_score[qid][docid] = score

        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_docid_to_score, fout)
    
    evaluate(args)


def lexical_ripor_rerank(args):
    # init model 
    model = LexicalRipor.from_pretrained(args.pretrained_path)
    model.eval()
    device = "cuda:0"
    model.to(device)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # start to reranking 
    model.base_model.mode = "lex_smt_outputs"
    for data_dir in set(args.q_collection_paths):
        run_path = os.path.join(args.smt_out_dir, get_dataset_name(data_dir), "run.json")
        dataset = LexicalRiporRerankDataset(run_path, query_dir=data_dir, smt_docid_to_smtid_path=args.smt_docid_to_smtid_path,
                                            lex_docid_to_smtid_path=args.lex_docid_to_smtid_path)
        dataloader = LexicalRiporRerankDataLoader(dataset=dataset, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=64,
                                            shuffle=False, num_workers=1)

        qid_to_docid_to_score = {}
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = to_device(batch, device)
            with torch.no_grad():
                scores = model.rerank_forward(**batch)["score"].cpu().tolist()
            
            qids = batch["qid"].cpu().tolist()
            docids = batch["docid"].cpu().tolist()

            for qid, docid, score in zip(qids, docids, scores):
                qid, docid = str(qid), str(docid)
                if qid not in qid_to_docid_to_score:
                    qid_to_docid_to_score[qid] = {docid: score}
                else:
                    qid_to_docid_to_score[qid][docid] = score

        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_docid_to_score, fout)
        
    evaluate(args)

def lexical_constrained_retrieve_and_rerank(args):
    # lexical retrieve 
    model = LexicalRipor.from_pretrained(args.pretrained_path)
    
    model.eval()
    device = "cuda:0"
    model.to(device)
    model.base_model.mode = "lex_retrieval"
    # retrive top k docids using lexical semantic ids
    #if not os.path.exists(args.lex_out_dir):
    #os.mkdir(args.lex_out_dir)
    os.makedirs(args.lex_out_dir, exist_ok=True)

    with open(os.path.join(args.lex_docid_to_smtid_path)) as fin:
        lex_docid_to_smtids = ujson.load(fin)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)


    retriever = TermEncoderRetriever(model, args)
    for data_dir in set(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1) 
        retriever.retrieve(q_loader, lex_docid_to_smtids, args.topk, out_dir=os.path.join(args.lex_out_dir, get_dataset_name(data_dir)))

        ## run.json --> sub_docid_to_smtids

    
    args.out_dir = args.lex_out_dir 

    evaluate(args)

def lexical_constrained_retrieve_and_rerank_2(args):
    ddp_setup()

    # read model 
    model = LexicalRipor.from_pretrained(args.pretrained_path)
    model.eval()
    device = args.local_rank #"cpu"
    model.to(device)
    model.base_model.mode = "smt_retrieval"
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    # define parameters for decoding
    with open(args.smt_docid_to_smtid_path) as fin:
        docid_to_tokenids = ujson.load(fin)
    assert args.max_new_token_for_docid in [2, 4, 6, 8, 16, 32], args.max_new_token_for_docid

    smtid_to_docids = {}
    docid_to_sub_tokenids = {}
    for docid, tokenids in docid_to_tokenids.items():
        assert tokenids[0] != -1, tokenids
        sid = "_".join([str(x) for x in tokenids[:args.max_new_token_for_docid]])
        if sid not in smtid_to_docids:
            smtid_to_docids[sid] = [docid]
        else:
            smtid_to_docids[sid] += [docid]

        docid_to_sub_tokenids[docid] = tokenids[:args.max_new_token_for_docid]

    max_new_token = args.max_new_token_for_docid
    assert len(sid.split("_")) == max_new_token, (sid, max_new_token)

    if args.local_rank <= 0:
        print("distribution of docids length per smtid: ", 
              np.quantile([len(xs) for xs in smtid_to_docids.values()], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        print("avergen length = {:.3f}".format(np.mean([len(xs) for xs in smtid_to_docids.values()])))
        print("smtid: ", sid)

    #if args.local_rank <= 0:
    #if not os.path.exists(args.out_dir):
    #os.mkdir(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

     # start to retrieve
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    # lexical_constrained semantic retrieve
    for data_dir in list(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = LexicalConditionCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1,
                                            sampler=DistributedSampler(q_collection, shuffle=False))
        
        ## out_dir
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        print("out_dir: ", out_dir)
        #if args.local_rank <= 0:
        #if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

        ## get lex_run_data
        lex_run_path = os.path.join(args.lex_out_dir, get_dataset_name(data_dir), "run.json")
        with open(lex_run_path) as fin:
            qid_to_rankdata = ujson.load(fin)
        #with open(args.smt_docid_to_smtid_path) as fin:
        #    docid_to_tokenids = ujson.load(fin)

        print("size of q_collection: ", len(q_collection))
        print("size of qid_to_rankdata: ", len(qid_to_rankdata))
        #assert '1108939' in qid_to_rankdata
        
        if args.lex_constrained == "lexical_condition":
            lexical_condition_decode_doc(model=model.base_model, 
                                dataloader=q_loader,
                                qid_to_rankdata=qid_to_rankdata,
                                docid_to_tokenids=docid_to_tokenids,
                                tokenizer=tokenizer,
                                smtid_to_docids=smtid_to_docids,
                                max_new_token=max_new_token,
                                device=args.local_rank, 
                                out_dir=out_dir,
                                local_rank=args.local_rank,
                                topk=args.topk)
        elif args.lex_constrained == "lexical_incorporate":
            lexical_inc_decode_doc(model=model.base_model, 
                                dataloader=q_loader,
                                qid_to_rankdata=qid_to_rankdata,
                                docid_to_tokenids=docid_to_tokenids,
                                tokenizer=tokenizer,
                                smtid_to_docids=smtid_to_docids,
                                max_new_token=max_new_token,
                                device=args.local_rank, 
                                out_dir=out_dir,
                                local_rank=args.local_rank,
                                topk=args.topk)
        elif args.lex_constrained == "lexical_tmp_rescore":
            lexical_tmp_rescore_decode_doc(model=model.base_model, 
                                dataloader=q_loader,
                                qid_to_rankdata=qid_to_rankdata,
                                docid_to_tokenids=docid_to_sub_tokenids, #docid_to_tokenids,
                                tokenizer=tokenizer,
                                smtid_to_docids=smtid_to_docids,
                                max_new_token=max_new_token,
                                device=device, 
                                out_dir=out_dir,
                                local_rank=args.local_rank,
                                topk=args.topk,
                                pooling=args.pooling)
        else:
            raise ValueError(f"the {args.lex_constrained} is not predefined.")

def lexical_constrained_retrieve_and_rerank_3(args):
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
    for data_dir in args.q_collection_paths:
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))

        # remove old 
        if os.path.exists(os.path.join(out_dir, "run.json")):
            print("old run.json exisit.")
            os.remove(os.path.join(out_dir, "run.json"))
        
        # merge
        qid_to_rankdata = {}
        sub_paths = [p for p in os.listdir(out_dir) if "run" in p]
        assert len(sub_paths) == torch.cuda.device_count()
        for sub_path in sub_paths:
            with open(os.path.join(out_dir, sub_path)) as fin:
                sub_qid_to_rankdata = ujson.load(fin)
            if len(qid_to_rankdata) == 0:
                qid_to_rankdata.update(sub_qid_to_rankdata)
            else:
                for qid, rankdata in sub_qid_to_rankdata.items():
                    if qid not in qid_to_rankdata:
                        qid_to_rankdata[qid] = rankdata
                    else:
                        qid_to_rankdata[qid].update(rankdata)
        print("length of pids and avg rankdata length in qid_to_rankdata: {}, {}".format(
        len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

        for sub_path in sub_paths:
            sub_path = os.path.join(out_dir, sub_path)
            os.remove(sub_path)

    evaluate(args)

def lexical_constrained_retrieve_and_rerank_4(args):
    # let's do reranking 
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
    for data_dir in args.q_collection_paths:
        smt_out_dir = os.path.join(args.smt_out_dir, get_dataset_name(data_dir))
        lex_out_dir = os.path.join(args.lex_out_dir, get_dataset_name(data_dir))

        with open(os.path.join(smt_out_dir, "run.json")) as fin:
            smt_run = ujson.load(fin)
        
        with open(os.path.join(lex_out_dir, "run.json")) as fin:
            lex_run = ujson.load(fin)
        
        result_run = {}
        for qid in tqdm(smt_run):
            result_run[qid] = {}
            for docid, smt_score in smt_run[qid].items():
                #print(len(lex_run[qid]), qid)
                lex_score = lex_run[qid][docid]
                result_run[qid][docid] = smt_score + lex_score
    
    evaluate(args)

def lexical_constrained_retrieve_and_rerank_2_for_sub_tokens(args):
    ddp_setup()

    # read model 
    model = LexicalRipor.from_pretrained(args.pretrained_path)
    model.eval()
    device = args.local_rank #"cpu"
    model.to(device)
    model.base_model.mode = "smt_retrieval"
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    # define parameters for decoding
    with open(args.smt_docid_to_smtid_path) as fin:
        docid_to_tokenids = ujson.load(fin)
    assert args.max_new_token_for_docid in [2, 4, 6, 8, 16, 32], args.max_new_token_for_docid
    smtid_to_docids = {}
    docid_to_sub_tokenids = {}
    for docid, tokenids in docid_to_tokenids.items():
        assert tokenids[0] != -1, tokenids
        sid = "_".join([str(x) for x in tokenids[:args.max_new_token_for_docid]])
        if sid not in smtid_to_docids:
            smtid_to_docids[sid] = [docid]
        else:
            smtid_to_docids[sid] += [docid]
            
        docid_to_sub_tokenids[docid] = tokenids[:args.max_new_token_for_docid]

    max_new_token = args.max_new_token_for_docid
    assert len(sid.split("_")) == max_new_token, (sid, max_new_token)

    if args.local_rank <= 0:
        print("distribution of docids length per smtid: ", 
              np.quantile([len(xs) for xs in smtid_to_docids.values()], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        print("avergen length = {:.3f}".format(np.mean([len(xs) for xs in smtid_to_docids.values()])))
        print("smtid: ", sid)

    if args.local_rank <= 0:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

     # start to retrieve
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    # lexical_constrained semantic retrieve
    for data_dir in list(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = LexicalConditionCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1,
                                            sampler=DistributedSampler(q_collection, shuffle=False))
        
        ## out_dir
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        print("out_dir: ", out_dir)
        if args.local_rank <= 0:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

        ## get lex_run_data
        lex_run_path = os.path.join(args.lex_out_dir, get_dataset_name(data_dir), "run.json")
        with open(lex_run_path) as fin:
            qid_to_rankdata = ujson.load(fin)

        print("size of q_collection: ", len(q_collection))
        print("size of qid_to_rankdata: ", len(qid_to_rankdata))
        #assert '1108939' in qid_to_rankdata
        
        if args.lex_constrained == "lexical_tmp_rescore":
            lexical_tmp_rescore_decode_sub_tokens(model=model.base_model, 
                                dataloader=q_loader,
                                qid_to_rankdata=qid_to_rankdata,
                                docid_to_tokenids=docid_to_sub_tokenids,
                                tokenizer=tokenizer,
                                smtid_to_docids=smtid_to_docids,
                                max_new_token=max_new_token,
                                device=device, 
                                out_dir=out_dir,
                                local_rank=args.local_rank,
                                topk=args.topk)
        else:
            raise ValueError(f"the {args.lex_constrained} is not predefined.")

# this is for parallel setting
def lexical_ripor_for_dense_pretrained_retrieve_and_rerank_1(args):
    ddp_setup()
    # initialize model
    model = LexicalRiporForDensePretrained.from_pretrained(args.pretrained_path)
    model.eval()
    device = args.local_rank
    model.to(device)
    model.base_model.mode = "lex_retrieval"
    # retrive top k docids using lexical semantic ids
    if is_first_worker():
        if not os.path.exists(args.lex_out_dir):
            os.mkdir(args.lex_out_dir)
    with open(os.path.join(args.lex_docid_to_smtid_path)) as fin:
        lex_docid_to_smtids = ujson.load(fin)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    retriever = TermEncoderRetriever(model, args)
    for data_dir in set(args.q_collection_paths):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1,
                                            sampler=DistributedSampler(q_collection, shuffle=False)) 
        retriever.retrieve(q_loader, lex_docid_to_smtids, args.topk, out_dir=os.path.join(args.lex_out_dir, get_dataset_name(data_dir)),
                           run_name=f"run_{args.local_rank}.json")

def lexical_ripor_for_dense_pretrained_retrieve_and_rerank_2(args):
    ddp_setup()
    # initialize model
    model = LexicalRiporForDensePretrained.from_pretrained(args.pretrained_path)
    model.eval()
    device = args.local_rank
    model.to(device)
    model.base_model.mode = "lex_smt_outputs"

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)

    if is_first_worker():
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

    for data_dir in list(args.q_collection_paths):
        run_path = os.path.join(args.lex_out_dir, get_dataset_name(data_dir), "run.json")
        dataset = LexicalRiporDensePretrainedRerankDataset(run_path, query_dir=data_dir, 
                                                           document_dir=args.collection_path,
                                                            lex_docid_to_smtid_path=args.lex_docid_to_smtid_path)
        dataloader = LexicalRiporDensePretrainedRerankDataLoader(dataset=dataset, tokenizer_type=args.pretrained_path,
                                                            max_length=args.max_length,
                                                            batch_size=64,
                                                            shuffle=False, num_workers=1,
                                                            sampler=DistributedSampler(dataset, shuffle=False))

        qid_to_docid_to_score = {}
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = to_device(batch, device)
            with torch.no_grad():
                scores = model.rerank_forward(**batch)["score"].cpu().tolist()
            
            qids = batch["qid"].cpu().tolist()
            docids = batch["docid"].cpu().tolist()

            for qid, docid, score in zip(qids, docids, scores):
                qid, docid = str(qid), str(docid)
                if qid not in qid_to_docid_to_score:
                    qid_to_docid_to_score[qid] = {docid: score}
                else:
                    qid_to_docid_to_score[qid][docid] = score

        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        run_name = f"run_{args.local_rank}.json"
        print(run_name)
        with open(os.path.join(out_dir, run_name), "w") as fout:
            ujson.dump(qid_to_docid_to_score, fout)

def lexical_ripor_for_dense_pretrained_merge_runs(args):
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
    for data_dir in args.q_collection_paths:
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))

        # remove old 
        if os.path.exists(os.path.join(out_dir, "run.json")):
            print("old run.json exisit.")
            os.remove(os.path.join(out_dir, "run.json"))
        
        # merge
        qid_to_rankdata = {}
        sub_paths = [p for p in os.listdir(out_dir) if "run" in p]
        assert len(sub_paths) == torch.cuda.device_count()
        for sub_path in sub_paths:
            with open(os.path.join(out_dir, sub_path)) as fin:
                sub_qid_to_rankdata = ujson.load(fin)
            if len(qid_to_rankdata) == 0:
                qid_to_rankdata.update(sub_qid_to_rankdata)
            else:
                for qid, rankdata in sub_qid_to_rankdata.items():
                    if qid not in qid_to_rankdata:
                        qid_to_rankdata[qid] = rankdata
                    else:
                        qid_to_rankdata[qid].update(rankdata)
        print("length of pids and avg rankdata length in qid_to_rankdata: {}, {}".format(
        len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

        for sub_path in sub_paths:
            sub_path = os.path.join(out_dir, sub_path)
            os.remove(sub_path)

    evaluate(args)

if __name__ == "__main__":
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]

    if args.task == "sparse_index":
        sparse_index(args)
    elif args.task == "multiprocess_sparse_index":
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'   

        args.world_size = torch.cuda.device_count()
        print("We have {} GPUs!".format(args.world_size))
        mp.spawn(multiprocess_sparse_index,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True
                 )
        merge_inverted_indexes(args.index_dir, args.world_size)
    elif args.task == "sparse_retrieve_and_evaluate":
        sparse_retrieve_and_evaluate(args)
    elif args.task == "spalde_get_bow_rep":
        spalde_get_bow_rep(args)
    elif args.task == "term_encoder_retrieve":
        term_encoder_retrieve(args)
    elif args.task == "term_encoder_parallel_retrieve":
        term_encoder_parallel_retrieve(args)
    elif args.task == "term_encoder_parallel_retrieve_2":
        term_encoder_parallel_retrieve_2(args)
    elif args.task == "index":
        index(args)
    elif args.task == "index_2":
        index_2(args)
    elif args.task == "dense_retrieve":
        dense_retrieve(args)
    elif args.task == "mmap":
        assert "mmap" in args.index_dir, args.index_dir
        index(args)
    elif args.task == "mmap_2":
        assert args.mmap_dir == args.index_dir, (args.mmap_dir, args.index_dir)
        mmap_2(args)
    elif args.task == "aq_index":
        aq_index(args)
    elif args.task == "aq_retrieve":
        aq_retrieve(args)
    elif args.task == "constrained_beam_search_for_qid_rankdata":
        constrained_beam_search_for_qid_rankdata(args)
    elif args.task == "constrained_beam_search_for_qid_rankdata_sub_tokens":
        constrained_beam_search_for_qid_rankdata_sub_tokens(args)
    elif args.task == "constrained_beam_search_for_qid_rankdata_2":
        constrained_beam_search_for_qid_rankdata_2(args)
    elif args.task == "constrained_beam_search_for_train_queries_2":
        constrained_beam_search_for_train_queries_2(args)
    elif args.task == "lexical_ripor_retrieve_and_rerank":
        lexical_ripor_retrieve_and_rerank(args)
    elif args.task == "lexical_ripor_retrieve_parallel":
        lexical_ripor_retrieve_parallel(args)
    elif args.task == "lexical_ripor_for_dense_pretrained_retrieve_and_rerank":
        lexical_ripor_for_dense_pretrained_retrieve_and_rerank(args)
    elif args.task == "lexical_constrained_retrieve_and_rerank":
        lexical_constrained_retrieve_and_rerank(args)
    elif args.task == "lexical_constrained_retrieve_and_rerank_2":
        lexical_constrained_retrieve_and_rerank_2(args)
    elif args.task == "lexical_constrained_retrieve_and_rerank_3":
        lexical_constrained_retrieve_and_rerank_3(args)
    elif args.task == "lexical_constrained_retrieve_and_rerank_4":
        lexical_constrained_retrieve_and_rerank_4(args)
    elif args.task == "lexical_constrained_retrieve_and_rerank_2_for_sub_tokens":
        lexical_constrained_retrieve_and_rerank_2_for_sub_tokens(args)
    elif args.task == "lexical_ripor_rerank":
        lexical_ripor_rerank(args)
    elif args.task == "lexical_ripor_dense_index":
        lexical_ripor_dense_index(args)
    elif args.task == "lexical_ripor_dense_aq_retrieve":
        lexical_ripor_dense_aq_retrieve(args)
    elif args.task == "lexical_ripor_for_dense_pretrained_retrieve_and_rerank_1":
        lexical_ripor_for_dense_pretrained_retrieve_and_rerank_1(args)
    elif args.task == "lexical_ripor_for_dense_pretrained_retrieve_and_rerank_2":
        lexical_ripor_for_dense_pretrained_retrieve_and_rerank_2(args)
    elif args.task == "lexical_ripor_for_dense_pretrained_merge_runs":
        lexical_ripor_for_dense_pretrained_merge_runs(args)
    elif args.task == "lexical_ripor_dense_index":
        lexical_ripor_dense_index(args)
    elif args.task == "lexical_ripor_dense_retrieve":
        lexical_ripor_dense_retrieve(args)
    else:
        raise ValueError(f"task: {args.task} is not valid.")
