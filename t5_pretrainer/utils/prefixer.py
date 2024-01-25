import os
from typing import Any 
import ujson 
import pickle 
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np

import torch

from .utils import is_first_worker

class Prefixer():
    def __init__(self, docid_to_tokenids_path, tokenizer, prefix_path=None, apply_stats=True, save_prefix=True):
        self.prefix_dict = defaultdict(set)
        if prefix_path is not None:
            print("directly read prefix_tree from {}".format(prefix_path))
            with open(prefix_path, "rb") as fin:
                self.prefix_dict = pickle.load(fin)
        else:
            with open(docid_to_tokenids_path) as fin:
                docid_to_tokenids = ujson.load(fin)
            
            for docid, tokenids in tqdm(docid_to_tokenids.items(), total=len(docid_to_tokenids), disable=not is_first_worker()):
                assert len(tokenids) in {8, 16 ,32} and tokenids[0] != -1 and tokenids[0] != 0, tokenids
                assert tokenids[0] >= 32000
                extended_tokenids = [tokenizer.pad_token_id] + tokenids
                for i in range(1, len(extended_tokenids)):
                    self.prefix_dict[tuple(extended_tokenids[:i])].add(extended_tokenids[i])
            

            if save_prefix:
                prefix_dir = os.path.dirname(docid_to_tokenids_path)
                with open(os.path.join(prefix_dir, "prefix.pickle"), "wb") as fout:
                    pickle.dump(self.prefix_dict, fout)
        # stats
        if apply_stats:
            prefix_counter = Counter()
            for prefix_id in tqdm(self.prefix_dict, total=len(self.prefix_dict), disable=not is_first_worker()):
                prefix_counter[len(prefix_id)] += 1
            for len_prefix in prefix_counter:
                if is_first_worker():
                    print("prefix_length = {}, tokenids_size = {}".format(len_prefix, prefix_counter[len_prefix]))

    def __call__(self, batch_id, sent):
        allowed_tokenids = list(self.prefix_dict[tuple(sent.cpu().tolist())])
        #if len(sent.cpu().tolist()) >= 1:
        #    print("allowed_tokenids: ", tuple(sent.cpu().tolist()), allowed_tokenids)
        return allowed_tokenids
    
class BatchPrefixer():
    def __init__(self, docid_to_tokenids, qid_to_rankdata, qids, tokenizer, apply_stats=False,
                 vectorized_map=False):
        self.list_prefix_dict = []
        self._list_prefix_docids = []
        self._qids = qids
        self._qid_to_rankdata = qid_to_rankdata

        for qid in qids:
            qid = str(qid)
            rankdata = qid_to_rankdata[qid]
            docids = list(rankdata.keys())
            
            prefix_dict = defaultdict(set)
            prefix_to_docids = defaultdict(set)
            for docid in docids:
                tokenids = docid_to_tokenids[docid]
                assert len(tokenids) in {8, 16 ,32} and tokenids[0] != -1 and tokenids[0] != 0, tokenids
                assert tokenids[0] >= 32000
                extended_tokenids = [tokenizer.pad_token_id] + tokenids
                #print("extened_tokenids: ", extended_tokenids)
                for i in range(1, len(extended_tokenids)):
                    prefix_dict[tuple(extended_tokenids[:i])].add(extended_tokenids[i])
                    prefix_to_docids[tuple(extended_tokenids[:i])].add(docid)
            
            self.list_prefix_dict.append(prefix_dict)
            self._list_prefix_docids.append(prefix_to_docids)
        
        # stats
        if apply_stats:
            prefix_counter = Counter()
            for prefix_id in tqdm(prefix_dict, total=len(prefix_dict), disable=not is_first_worker()):
                prefix_counter[len(prefix_id)] += 1
            for len_prefix in prefix_counter:
                if is_first_worker():
                    print("prefix_length = {}, tokenids_size = {}".format(len_prefix, prefix_counter[len_prefix]))

    def __call__(self, batch_id, sent):
        prefix_dict = self.list_prefix_dict[batch_id]
        #print("batch_id: ", batch_id)
        #print("prefix_dict: ", prefix_dict)
        allowed_tokenids = list(prefix_dict[tuple(sent.cpu().tolist())])
        return allowed_tokenids
    
    def _get_docids(self, batch_id, sent):
        prefix_to_docids = self._list_prefix_docids[batch_id]
        allowed_docids = list(prefix_to_docids[tuple(sent.cpu().tolist())])
        return allowed_docids


class BatchPrefixerForLexInc():
    # prefix --> next_token_ids 
    # prefix --> list of (next_token_id, lex_score)
    def __init__(self, docid_to_tokenids, qid_to_rankdata, qids, tokenizer,
                 pooling="max"):
        """
        qid_to_rankdata: this is from ranking data from lexical approach
        """
        self.list_prefix_dict = []
        self._list_prefix_to_pairs = []
        
        assert pooling in {"min", "max", "mean"}

        for qid in qids:
            qid = str(qid)
            rankdata = qid_to_rankdata[qid]
            docids = list(rankdata.keys())

            prefix_dict = defaultdict(set)
            prefix_to_pairs = {}
            for docid in docids:
                tokenids = docid_to_tokenids[docid]
                assert len(tokenids) in {2, 4, 6, 8, 16 ,32} and tokenids[0] != -1 and tokenids[0] != 0, tokenids
                assert tokenids[0] >= 32000
                extended_tokenids = [tokenizer.pad_token_id] + tokenids
                #print("extened_tokenids: ", extended_tokenids)
                for i in range(1, len(extended_tokenids)):
                    prefix = tuple(extended_tokenids[:i])
                    next_token_id = extended_tokenids[i]

                    prefix_dict[prefix].add(next_token_id)
                    if prefix not in prefix_to_pairs:
                        if pooling in {"min", "max"}:
                            prefix_to_pairs[prefix] = {next_token_id: rankdata[docid]}
                        else:
                            prefix_to_pairs[prefix] = {next_token_id: [rankdata[docid]]}
                    else:
                        if pooling in {"min", "max"}:
                            if next_token_id not in prefix_to_pairs[prefix]:
                                prefix_to_pairs[prefix][next_token_id] = rankdata[docid]
                            else:
                                if pooling == "max":
                                    prefix_to_pairs[prefix][next_token_id] = max(rankdata[docid], prefix_to_pairs[prefix][next_token_id])
                                else:
                                    prefix_to_pairs[prefix][next_token_id] = min(rankdata[docid], prefix_to_pairs[prefix][next_token_id])
                        else:
                            assert pooling == "mean", pooling
                            if next_token_id not in prefix_to_pairs[prefix]:
                                prefix_to_pairs[prefix][next_token_id] = [rankdata[docid]]
                            else:
                                prefix_to_pairs[prefix][next_token_id] += [rankdata[docid]]
                                
            if pooling == "mean":
                for prefix in prefix_to_pairs:
                    for next_token_id, scores in prefix_to_pairs[prefix].items():
                        prefix_to_pairs[prefix][next_token_id] = np.mean(scores)
                        
            self.list_prefix_dict.append(prefix_dict)
            self._list_prefix_to_pairs.append(prefix_to_pairs)
    
    def __call__(self, batch_id, sent):
        prefix_dict = self.list_prefix_dict[batch_id]
        #print("batch_id: ", batch_id)
        #print("prefix_dict: ", prefix_dict)
        allowed_tokenids = list(prefix_dict[tuple(sent.cpu().tolist())])
        return allowed_tokenids
    
    def _get_tokenids_and_scores(self, batch_id, sent):
        prefix_to_pair = self._list_prefix_to_pairs[batch_id]
        tokenid_to_score = prefix_to_pair[tuple(sent.cpu().tolist())]

        token_ids, scores = list(tokenid_to_score.keys()), list(tokenid_to_score.values())

        return token_ids, torch.FloatTensor(scores).to(sent.device)

def generate_special_token_list(num_code, codebook_size):
    token_list = []
    for i in range(num_code):
        for j in range(codebook_size):
            token_list.append(f"<docid_{i}_{j}>")
    return token_list 

def construct_sub_docid_to_tokenids_from_run(run_dir, docid_to_tokenids_path):
    with open(docid_to_tokenids_path) as fin:
        docid_to_tokenids = ujson.load(fin)
    
    with open(os.path.join(run_dir, "run.json")) as fin:
        run = ujson.load(fin)

    sub_docid_to_tokenids = {}
    for qid in run:
        for docid in run[qid]:
            tokenids = docid_to_tokenids[docid]
            assert tokenids[0] >= 32_000 and tokenids[1] >= 32_000 
            sub_docid_to_tokenids[docid] = tokenids

    print("size of docids = {}".format(len(sub_docid_to_tokenids)))
    with open(os.path.join(run_dir, "sub_docid_to_tokenids.json"), "w") as fout:
        ujson.dump(sub_docid_to_tokenids, fout)

    return os.path.join(run_dir, "sub_docid_to_tokenids.json")





