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
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

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
    #CollectionDatasetPreLoad,
    LexicalRiporRerankDataset,
    LexicalRiporDensePretrainedRerankDataset,
    BeirDataset
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
                            topk=100,
                            id_to_key=None):
    
    lex_qid_to_smtid_to_score = get_qid_smtid_scores(qid_to_rankdata, docid_to_tokenids)
    out_qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        batch_qids = batch["id"].cpu().tolist()
        if id_to_key is not None:
            batch_qids = [id_to_key[x] for x in batch_qids]
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
            #if id_to_key is not None:
            #    qid = id_to_key[qid]
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

def ddp_setup():
    init_process_group(backend="nccl")

def index(args):
    ddp_setup()

    # parallel initialiaztion
    assert args.local_rank != -1
    model = T5DenseEncoder.from_pretrained(args.pretrained_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        args.beir_dataset)
    out_dir = args.beir_dataset_path
    data_path = util.download_and_unzip(url, out_dir)

    os.makedirs(args.index_dir, exist_ok=True)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    d_collection = BeirDataset(corpus, information_type="document")

    d_loader = T5DenseCollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                    max_length=args.max_length,
                                    batch_size=args.index_retrieve_batch_size,
                                    num_workers=1,
                                sampler=DistributedSampler(d_collection, shuffle=False))
    evaluator = DenseIndexing(model=model, args=args)
    evaluator.store_embs(d_loader, args.local_rank, use_fp16=False)

    if is_first_worker():
        with open(os.path.join(args.index_dir, "idx_to_key.json"), "w") as fout:
            ujson.dump(d_collection.idx_to_key, fout)
    
    destroy_process_group()

def index_2(args):
    DenseIndexing.aggregate_embs_to_index(args.index_dir)

def lexical_ripor_dense_index(args):
    ddp_setup()

    assert args.local_rank != -1
    model = LexicalRiporForDensePretrained.from_pretrained(args.pretrained_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        args.beir_dataset)
    out_dir = args.beir_dataset_path
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    d_collection = BeirDataset(corpus, information_type="document")

    d_loader = LexicalRiporDenseCollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                    max_length=args.max_length,
                                    batch_size=args.index_retrieve_batch_size,
                                    num_workers=1,
                                sampler=DistributedSampler(d_collection, shuffle=False))
    evaluator = DenseIndexing(model=model, args=args)
    evaluator.store_embs(d_loader, args.local_rank, use_fp16=False)
    
    destroy_process_group()

def aq_index(args):
    if not os.path.exists(args.index_dir):
        os.mkdir(args.index_dir)
    AddictvieQuantizeIndexer.index(args.mmap_dir, index_dir=args.index_dir, codebook_num=args.codebook_num,
                                   codebook_bits=args.codebook_bits)

def spalde_get_bow_rep(args):
    # read model_config first
    with open(os.path.join(args.pretrained_path, "config.json")) as fin:
        model_config = ujson.load(fin)
    if model_config["architectures"][0] == "T5ForConditionalGeneration":
        model = T5Splade.from_pretrained(args.pretrained_path)
        print("Read Splade with T5 as the backbone")
    else:
        raise ValueError("model architecture: {} is not predefined".format(model_config["architectures"][0]))
    
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        args.beir_dataset)
    out_dir = args.beir_dataset_path
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    d_collection = BeirDataset(corpus, information_type="document")
    d_loader = T5SpladeCollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                    max_length=args.max_length,
                                    batch_size=args.index_retrieve_batch_size,
                                    shuffle=False, num_workers=1) 
    
    device = "cuda:0"
    model.eval()
    model.to(device)
    topk = 64
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
            docid = d_collection.idx_to_key[docids[i]]
            token_ids = top_indices[i]
            docid_to_tokenids[docid] = token_ids

    
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "docid_to_tokenids.json"), "w") as fout:
        ujson.dump(docid_to_tokenids, fout)
    with open(os.path.join(args.out_dir, "meta_data.json"), "w") as fout:
        info = {"topk": topk}
        ujson.dump(info, fout)

def mmap_2(args):
    DenseIndexing.aggregate_embs_to_mmap(args.mmap_dir)

def lexical_constrained_retrieve_and_rerank(args):
    # lexical retrieve 
    model = LexicalRipor.from_pretrained(args.pretrained_path)
    
    model.eval()
    device = "cuda:0"
    model.to(device)
    model.base_model.mode = "lex_retrieval"

    os.makedirs(args.lex_out_dir, exist_ok=True)
    with open(os.path.join(args.lex_docid_to_smtid_path)) as fin:
        lex_docid_to_smtids = ujson.load(fin)

    retriever = TermEncoderRetriever(model, args)
    
    # dataset 
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        args.beir_dataset)
    out_dir = args.beir_dataset_path
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    q_collection = BeirDataset(queries, information_type="query")
    q_loader = T5SpladeCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=args.batch_size,
                                        shuffle=False, num_workers=1) 

    retriever.retrieve(q_loader, lex_docid_to_smtids, args.topk, out_dir=args.lex_out_dir)

    # let's re-write the "run.json", since we store qidx instead of qid     
    with open(os.path.join(args.lex_out_dir, "run.json")) as reader:
        old_run = json.load(reader)

    run = {}
    for qidx, rankdata in old_run.items():
        qid = q_collection.idx_to_key[int(qidx)]
        run[qid] = rankdata
    with open(os.path.join(args.lex_out_dir, "run.json"), "w") as fout:
        ujson.dump(run ,fout)
    new_run = dict()

    print("Removing query id from document list")
    for query_id, doc_dict in tqdm(run.items()):
        query_dict = dict()
        for doc_id, doc_values in doc_dict.items():
            if query_id != doc_id:
                query_dict[doc_id] = doc_values
        new_run[query_id] = query_dict

    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, new_run, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, new_run, [1, 10, 100, 1000], metric="r_cap")
    res = {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@100": recall["Recall@100"],
        "R_cap@100": results2["R_cap@100"]
    }
    print(res)
    json.dump(res, open(os.path.join(args.lex_out_dir, "perf.json"), "w"))

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
    assert args.max_new_token_for_docid in [4, 8, 16, 32], args.max_new_token_for_docid

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
        print(args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)

    # lexical_constrained semantic retrieve
    # dataset 
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        args.beir_dataset)
    out_dir = args.beir_dataset_path
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    out_dir = args.out_dir
    
    q_collection = BeirDataset(queries, information_type="query")
    q_loader = LexicalConditionCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=args.batch_size,
                                        shuffle=False, num_workers=1,
                                        sampler=DistributedSampler(q_collection, shuffle=False))

    ## get lex_run_data
    lex_run_path = os.path.join(args.lex_out_dir, "run.json")
    with open(lex_run_path) as fin:
        qid_to_rankdata = ujson.load(fin)
    with open(args.smt_docid_to_smtid_path) as fin:
        docid_to_tokenids = ujson.load(fin)

    print("size of q_collection: ", len(q_collection))
    print("size of qid_to_rankdata: ", len(qid_to_rankdata))
        
    lexical_tmp_rescore_decode_doc(model=model.base_model, 
                        dataloader=q_loader,
                        qid_to_rankdata=qid_to_rankdata,
                        docid_to_tokenids=docid_to_tokenids,
                        tokenizer=tokenizer,
                        smtid_to_docids=smtid_to_docids,
                        max_new_token=max_new_token,
                        device=device, 
                        out_dir=out_dir,
                        local_rank=args.local_rank,
                        topk=args.topk,
                        id_to_key=q_collection.idx_to_key)
        
def lexical_constrained_retrieve_and_rerank_3(args):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        args.beir_dataset)
    out_dir = args.beir_dataset_path
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    out_dir = args.out_dir
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

    with open(os.path.join(args.out_dir, "run.json")) as reader:
        run = json.load(reader)
    new_run = dict()

    print("Removing query id from document list")
    for query_id, doc_dict in tqdm(run.items()):
        query_dict = dict()
        for doc_id, doc_values in doc_dict.items():
            if query_id != doc_id:
                query_dict[doc_id] = doc_values
        new_run[query_id] = query_dict

    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, new_run, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, new_run, [1, 10, 100, 1000], metric="r_cap")
    res = {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@100": recall["Recall@100"],
        "R_cap@100": results2["R_cap@100"]
    }
    print(res)
    json.dump(res, open(os.path.join(args.out_dir, "perf.json"), "w"))

if __name__ == "__main__":
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]

    if args.task in {"index", "mmap"}:
        if args.task == "mmap":
            assert "mmap" in args.index_dir, args.index_dir 
        index(args)
    elif args.task == "mmap_2":
        mmap_2(args)
    elif args.task == "lexical_ripor_dense_index":
        lexical_ripor_dense_index(args)
    elif args.task == "aq_index":
        aq_index(args)
    elif args.task == "spalde_get_bow_rep":
        spalde_get_bow_rep(args)
    elif args.task == "lexical_constrained_retrieve_and_rerank":
        lexical_constrained_retrieve_and_rerank(args)
    elif args.task == "lexical_constrained_retrieve_and_rerank_2":
        lexical_constrained_retrieve_and_rerank_2(args)
    elif args.task == "lexical_constrained_retrieve_and_rerank_3":
        lexical_constrained_retrieve_and_rerank_3(args)
    else:
        print(args.task)
        raise NotImplementedError