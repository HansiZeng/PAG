import gzip
import json
import os
import pickle
import random
import copy

import ujson
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer

#QUERY_PREFIX = "query: "
#DOC_PREFIX = "document: "

# for rerank 
class RerankDataset(Dataset):
    def __init__(self, run_json_path, document_dir, query_dir, json_type="jsonl"):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        if json_type == "jsonl":
            self.all_pair_ids = []
            with open(run_json_path) as fin:
                for line in fin:
                    example = ujson.loads(line)
                    qid, docids = example["qid"], example["docids"]
                    for docid in docids:
                        self.all_pair_ids.append((qid, docid))
        else:
            with open(run_json_path) as fin:
                qid_to_rankdata = ujson.load(fin)
            
            self.all_pair_ids = []
            for qid, rankdata in qid_to_rankdata.items():
                for pid, _ in rankdata.items():
                    self.all_pair_ids.append((str(qid), str(pid)))
        
    def __len__(self):
        return len(self.all_pair_ids)
    
    def __getitem__(self, idx):
        pair_id = self.all_pair_ids[idx]
        qid, pid = pair_id
        
        query = self.query_dataset[qid][1]
        doc = self.document_dataset[pid][1]
        
        return {
            "pair_id": pair_id,
            "query": query,
            "doc": doc,
        }

class PseudoQueryForScoreDataset(Dataset):
    def __init__(self, document_dir, pseudo_queries_path, docid_pseudo_qids_path):
        docid_to_doc = {}
        with open(os.path.join(document_dir, "raw.tsv")) as fin:
            for line in fin:
                docid, doc = line.strip().split("\t")
                docid_to_doc[docid] = doc 
        self.docid_to_doc = docid_to_doc
        
        qid_to_query = {}
        with open(pseudo_queries_path) as fin:
            for line in tqdm(fin, total=442090001):
                try:
                    qid, query = line.strip().split("\t")
                    qid_to_query[qid] = query 
                except:
                    print(line)
                    raise ValueError("wrong here")
        self.qid_to_query = qid_to_query

        self.out_pairs = [] 
        with open(docid_pseudo_qids_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                docid = example["docid"]
                qids = example["qids"]

                for qid in qids:
                    self.out_pairs.append((qid, docid))
    
    def __len__(self):
        return len(self.out_pairs)

    def __getitem__(self, idx):
        qid, docid = self.out_pairs[idx]
        query, doc = self.qid_to_query[qid], self.docid_to_doc[docid]

        return (qid, docid, query, doc)

class QueryToSmtidRerankDataset(Dataset):
    def __init__(self, qid_docids_path, queries_path, docid_to_smtids, docid_to_strsmtid):
        self.examples = []
        with open(qid_docids_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                self.examples.append(
                    {"qid": example["qid"], "docids": example["docids"]}
                )


        self.qid_to_query = {}
        with open(queries_path) as fin:
            for line in fin:
                qid, query = line.strip().split("\t")
                self.qid_to_query[qid] = query

        self.qid_strsmtid_pairs = []
        self.qid_smtids_pairs = []
        for example in self.examples:
            qid = example["qid"]
            for docid in example["docids"]:
                self.qid_strsmtid_pairs.append((qid, docid_to_strsmtid[docid]))
                self.qid_smtids_pairs.append((qid, docid_to_smtids[docid]))

    def __len__(self):
        return len(self.qid_strsmtid_pairs)

    def __getitem__(self, idx):
        qid, strsmtid = self.qid_strsmtid_pairs[idx]
        _, smtids = self.qid_smtids_pairs[idx]
        query = self.qid_to_query[qid]

        assert smtids[0] == -1, smtids 
        decoder_input_ids = smtids[:-1]
        labels = smtids[1:]

        return query, qid, strsmtid, decoder_input_ids, labels

class TeacherRerankFromQidSmtidsDataset(Dataset):
    def __init__(self, qid_smtid_rank_path, docid_to_smtids_path, queries_path, collection_path):
        with open(docid_to_smtids_path) as fin:
            docid_to_smtids = ujson.load(fin)
        smtid_to_docids = {}
        for docid, smtids in docid_to_smtids.items():
            assert smtids[0] == -1, smtids 
            smtid = "_".join([str(x) for x in smtids[1:]])
            if smtid not in smtid_to_docids:
                smtid_to_docids[smtid] = [docid]
            else:
                smtid_to_docids[smtid] += [docid]
        self.docid_to_smtids = docid_to_smtids

        self.qid_to_query = {}
        with open(queries_path) as fin:
            for line in fin:
                qid, query = line.strip().split("\t")
                self.qid_to_query[qid] = query 
        
        self.docid_to_doc = {}
        with open(collection_path) as fin:
            for line in fin:
                docid, doc = line.strip().split("\t")
                self.docid_to_doc[docid] = doc 

        smtid_lenghts = []
        self.all_pair_ids = []
        with open(qid_smtid_rank_path) as fin:
            rank_data = ujson.load(fin)
        for qid in rank_data:
            smtid_lenghts.append(len(rank_data[qid]))
            for smtid in rank_data[qid]:
                docids = smtid_to_docids[smtid]
                for docid in docids:
                    self.all_pair_ids.append((str(qid), str(docid)))

        print("number of qids = {}".format(len(rank_data)))
        print("average number of smtids_per_query = {:.3f}".format(np.mean(smtid_lenghts)))
        print("average docs per query = {:.3f}".format(len(self.all_pair_ids) / len(rank_data)))
        

    def __len__(self):
        return len(self.all_pair_ids)
    
    def __getitem__(self, idx):
        pair_id = self.all_pair_ids[idx]
        qid, docid = pair_id
        
        query = self.qid_to_query[qid]
        doc = self.docid_to_doc[docid]
        
        return {
            "pair_id": pair_id,
            "query": query,
            "doc": doc,
        }

class LexicalRiporRerankDataset(Dataset):
    def __init__(self, run_path, query_dir, smt_docid_to_smtid_path, lex_docid_to_smtid_path):
        with open(run_path) as fin:
            self.run = ujson.load(fin)
        
        self.qid_docid_pairs = []
        for qid in self.run:
            for docid, _ in self.run[qid].items():
                self.qid_docid_pairs.append((qid, docid))
        
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")

        with open(smt_docid_to_smtid_path) as fin:
            self.smt_docid_to_smtid = ujson.load(fin)
        with open(lex_docid_to_smtid_path) as fin:
            self.lex_docid_to_smtid = ujson.load(fin)
    
    def __len__(self):
        return len(self.qid_docid_pairs)

    def __getitem__(self, idx):
        qid, docid = self.qid_docid_pairs[idx]

        query = self.query_dataset[qid][1]
        lex_doc_encoding = self.lex_docid_to_smtid[docid]
        smt_doc_encoding = self.smt_docid_to_smtid[docid]

        return int(qid), int(docid), query, lex_doc_encoding, smt_doc_encoding

class LexicalRiporDensePretrainedRerankDataset(Dataset):
    def __init__(self, run_path, query_dir, document_dir, lex_docid_to_smtid_path):
        with open(run_path) as fin:
            self.run = ujson.load(fin)
        
        self.qid_docid_pairs = []
        for qid in self.run:
            for docid, _ in self.run[qid].items():
                self.qid_docid_pairs.append((qid, docid))
        
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")

        with open(lex_docid_to_smtid_path) as fin:
            self.lex_docid_to_smtid = ujson.load(fin)
    
    def __len__(self):
        return len(self.qid_docid_pairs)

    def __getitem__(self, idx):
        qid, docid = self.qid_docid_pairs[idx]

        query = self.query_dataset[qid][1]
        doc = self.document_dataset[docid][1]
        lex_doc_encoding = self.lex_docid_to_smtid[docid]

        return int(qid), int(docid), query, doc, lex_doc_encoding


class CrossEncRerankForSamePrefixPair(Dataset):
    def __init__(self, qid_to_smtid_to_docids, queries_path, collection_path):
        self.qid_to_query = {}
        with open(queries_path) as fin:
            for line in fin:
                qid, query = line.strip().split("\t")
                self.qid_to_query[qid] = query 
        
        self.docid_to_doc = {}
        with open(collection_path) as fin:
            for line in fin:
                docid, doc = line.strip().split("\t")
                self.docid_to_doc[docid] = doc 

        self.qid_to_smtid_to_docids = qid_to_smtid_to_docids
        self.triples = []
        
        for i, qid in enumerate(tqdm(self.qid_to_smtid_to_docids, total=len(self.qid_to_smtid_to_docids))):
            for smtid, docids in self.qid_to_smtid_to_docids[qid].items():
                for docid in docids:
                    triple = (qid, docid, smtid)
                    self.triples.append(triple)


    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        qid, docid, smtid = self.triples[idx]
        query = self.qid_to_query[qid]
        doc = self.docid_to_doc[docid]

        return {
            "triple_id": (qid, docid, smtid),
            "query": query,
            "doc": doc
        }
        
# for evaluate
class CollectionDatasetPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir, id_style):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        print("Preloading dataset")
        with open(os.path.join(self.data_dir, "raw.tsv")) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[i] = data
                        self.line_dict[i] = id_.strip()
                    else:
                        self.data_dict[id_] = data.strip()
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]

class BeirDataset(Dataset):
    """
    dataset to iterate over a BEIR collection
    we preload everything in memory at init
    """

    def __init__(self, value_dictionary, information_type="document"):
        assert information_type in ["document", "query"]
        self.value_dictionary = value_dictionary
        self.information_type = information_type
        if self.information_type == "document":
            self.value_dictionary = dict()
            for key, value in value_dictionary.items():
                self.value_dictionary[key] = value["title"] + " " + value["text"]
        self.idx_to_key = {idx: key for idx, key in enumerate(self.value_dictionary)}

    def __len__(self):
        return len(self.value_dictionary)

    def __getitem__(self, idx):
        true_idx = self.idx_to_key[idx]
        return idx, self.value_dictionary[true_idx]


# for training 
class MarginMSEDataset(Dataset):
    def __init__(self, example_path, document_dir, query_dir):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")

        self.examples = []
        with open(example_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qid, docids, scores = self.examples[idx]["qid"], self.examples[idx]["docids"], self.examples[idx]["scores"]

        pos_docid = docids[0]
        pos_score = scores[0]

        neg_idx = random.sample(range(1, len(docids)), k=1)[0]
        neg_docid = docids[neg_idx]
        neg_score = scores[neg_idx]
        
        query = self.query_dataset[qid][1]
        pos_doc = self.document_dataset[pos_docid][1]
        neg_doc = self.document_dataset[neg_docid][1]

        return query, pos_doc, neg_doc, pos_score, neg_score

class TermEncoderForMarginMSEDataset(Dataset):
    def __init__(self, example_path, docid_to_smtid_path, query_dir):
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")

        self.examples = []
        with open(example_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                self.examples.append(example)

        with open(docid_to_smtid_path) as fin:
            self.docid_to_smtids = ujson.load(fin)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        qid, docids, scores = self.examples[idx]["qid"], self.examples[idx]["docids"], self.examples[idx]["scores"]

        pos_docid = docids[0]
        pos_score = scores[0]

        neg_idx = random.sample(range(1, len(docids)), k=1)[0]
        neg_docid = docids[neg_idx]
        neg_score = scores[neg_idx]
        
        query = self.query_dataset[qid][1]
        pos_doc_encoding = self.docid_to_smtids[pos_docid]
        neg_doc_encoding = self.docid_to_smtids[neg_docid]

        return query, pos_doc_encoding, neg_doc_encoding, pos_score, neg_score

class RiporForMarginMSEDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir,
                 docid_to_smtid_path, smtid_as_docid=False):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        self.smtid_as_docid = smtid_as_docid

        if self.smtid_as_docid:
            self.docid_to_smtid = None
            assert docid_to_smtid_path == None
        else:
            if docid_to_smtid_path is not None:
                with open(docid_to_smtid_path) as fin: 
                    self.docid_to_smtid = ujson.load(fin)
                tmp_docids = list(self.docid_to_smtid.keys())
                assert self.docid_to_smtid[tmp_docids[0]][0] != -1, self.docid_to_smtid[tmp_docids[0]]
            else:
                self.docid_to_smtid = None 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        query = example["qid"]
        if self.smtid_as_docid:
            positive = example["smtids"][0]
        else:
            positive = example["docids"][0]
        s_pos = example["scores"][0]

        if self.smtid_as_docid:
            neg_idx = random.sample(range(1, len(example["smtids"])), k=1)[0]
            negative = example["smtids"][neg_idx]
        else:
            neg_idx = random.sample(range(1, len(example["docids"])), k=1)[0]
            negative = example["docids"][neg_idx]
        s_neg = example["scores"][neg_idx]

        q = self.query_dataset[str(query)][1]

        if self.smtid_as_docid:
            pos_doc_encoding = [int(x) for x in positive.split("_")]
            neg_doc_encoding = [int(x) for x in negative.split("_")]
        else:
            pos_doc_encoding = self.docid_to_smtid[str(positive)]
            neg_doc_encoding = self.docid_to_smtid[str(negative)]
        
        assert len(pos_doc_encoding) == len(neg_doc_encoding)
        assert len(pos_doc_encoding) in {4, 8, 16, 32}, len(pos_doc_encoding)
        q_pos = q.strip()
        q_neg = q.strip()

        return q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg

class RiporForLngKnpMarginMSEDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir,
                 docid_to_smtid_path, smtid_as_docid=False):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        self.smtid_as_docid = smtid_as_docid

        if self.smtid_as_docid:
            self.docid_to_smtid = None
            assert docid_to_smtid_path == None
        else:
            if docid_to_smtid_path is not None:
                with open(docid_to_smtid_path) as fin: 
                    self.docid_to_smtid = ujson.load(fin)
                tmp_docids = list(self.docid_to_smtid.keys())
                assert self.docid_to_smtid[tmp_docids[0]][0] != -1, self.docid_to_smtid[tmp_docids[0]]
            else:
                self.docid_to_smtid = None 

        # sanity check 
        smtid_len = len(self.examples[0]["smtids"][0].split("_"))
        if smtid_len == 32:
            print("smtid_len is 32")
            assert "smtid_4_scores" in self.examples[0] and "smtid_8_scores" in self.examples[0] and "smtid_16_scores" in self.examples[0]
        elif smtid_len == 16:
            print("smtid_len is 16")
            assert "smtid_4_scores" in self.examples[0] and "smtid_8_scores" in self.examples[0]
            assert "smtid_16_scores" not in self.examples[0] 
        elif smtid_len == 8:
            print("smtid_len is 8")
            assert "smtid_4_scores" in self.examples[0]
            assert "smtid_8_scores" not in self.examples[0] and "smtid_16_scores" not in self.examples[0] 
        else:
            raise ValueError("not valid smtid_len = {}".format(smtid_len))
        self.smtid_len = smtid_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        query = example["qid"]
        if self.smtid_as_docid:
            positive = example["smtids"][0]
        else:
            positive = example["docids"][0]
        s_pos = example["scores"][0]
        if self.smtid_len == 8:
            smtid_4_s_pos = example["smtid_4_scores"][0]
        else:
            raise NotImplementedError

        if self.smtid_as_docid:
            neg_idx = random.sample(range(1, len(example["smtids"])), k=1)[0]
            negative = example["smtids"][neg_idx]
        else:
            neg_idx = random.sample(range(1, len(example["docids"])), k=1)[0]
            negative = example["docids"][neg_idx]
        s_neg = example["scores"][neg_idx]

        if self.smtid_len == 8:
            smtid_4_s_neg = example["smtid_4_scores"][neg_idx]
        else:
            raise NotImplementedError

        q = self.query_dataset[str(query)][1]

        if self.smtid_as_docid:
            pos_doc_encoding = [int(x) for x in positive.split("_")]
            neg_doc_encoding = [int(x) for x in negative.split("_")]
        else:
            pos_doc_encoding = self.docid_to_smtid[str(positive)]
            neg_doc_encoding = self.docid_to_smtid[str(negative)]
        
        assert len(pos_doc_encoding) == len(neg_doc_encoding)
        assert len(pos_doc_encoding) in {8, 16, 32}, len(pos_doc_encoding)

        q_pos = q.strip()
        q_neg = q.strip()
        
        if self.smtid_len == 8:
            return q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, smtid_4_s_pos, smtid_4_s_neg 
        else:
            raise NotImplementedError

class LexicalRiporForMarginMSEDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir,
                 smt_docid_to_smtid_path, lex_docid_to_smtid_path, 
                 smtid_as_docid=False):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        self.smtid_as_docid = smtid_as_docid

        if self.smtid_as_docid:
            self.docid_to_smtid = None
            assert smt_docid_to_smtid_path == None
        else:
            with open(smt_docid_to_smtid_path) as fin:
                self.smt_docid_to_smtid = ujson.load(fin)
            with open(lex_docid_to_smtid_path) as fin:
                self.lex_docid_to_smtid = ujson.load(fin)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        query = example["qid"]
        if self.smtid_as_docid:
            smt_positive = example["smt_smtids"][0]
            lex_positive = example["lex_smtids"][0]
        else:
            positive = example["docids"][0]
        s_pos = example["scores"][0]

        if self.smtid_as_docid:
            neg_idx = random.sample(range(1, len(example["smtids"])), k=1)[0]
            smt_negative = example["smt_smtids"][neg_idx]
            lex_negative = example["lex_smtids"][neg_idx]
        else:
            neg_idx = random.sample(range(1, len(example["docids"])), k=1)[0]
            negative = example["docids"][neg_idx]
        s_neg = example["scores"][neg_idx]

        q = self.query_dataset[str(query)][1]

        if self.smtid_as_docid:
            smt_pos_doc_encoding = [int(x) for x in smt_positive.split("_")]
            smt_neg_doc_encoding = [int(x) for x in smt_negative.split("_")]

            lex_pos_doc_encoding = [int(x) for x in lex_positive.split("_")]
            lex_neg_doc_encoding = [int(x) for x in lex_negative.split("_")]
        else:
            smt_pos_doc_encoding = self.smt_docid_to_smtid[str(positive)]
            smt_neg_doc_encoding = self.smt_docid_to_smtid[str(negative)]

            lex_pos_doc_encoding = self.lex_docid_to_smtid[str(positive)]
            lex_neg_doc_encoding = self.lex_docid_to_smtid[str(negative)]
        
        assert len(smt_pos_doc_encoding) == len(smt_neg_doc_encoding) and len(lex_pos_doc_encoding) == len(lex_neg_doc_encoding)
        assert len(smt_pos_doc_encoding) in {4, 8, 16}, len(smt_pos_doc_encoding)
        assert len(lex_pos_doc_encoding) in {16, 32, 64, 128}
        assert smt_pos_doc_encoding[0] >= 32000

        q_pos = q.strip()
        q_neg = q.strip()

        return q_pos, q_neg, lex_pos_doc_encoding, lex_neg_doc_encoding, smt_pos_doc_encoding, smt_neg_doc_encoding, s_pos, s_neg

class LexicalRiporForKLDivDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir,
                 smt_docid_to_smtid_path, lex_docid_to_smtid_path, 
                 smtid_as_docid=False, nway=30):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        self.smtid_as_docid = smtid_as_docid

        if self.smtid_as_docid:
            self.docid_to_smtid = None
            assert smt_docid_to_smtid_path == None
        else:
            with open(smt_docid_to_smtid_path) as fin:
                self.smt_docid_to_smtid = ujson.load(fin)
            with open(lex_docid_to_smtid_path) as fin:
                self.lex_docid_to_smtid = ujson.load(fin)

        self.nway = nway

    def __len__(self):
        return len(self.examples)
    
    def _get_2d_encoding(self, smtids):
        out_smtids = []
        for smtid in smtids:
            out_smtids.append([int(x) for x in smtid])
        
        return out_smtids

    def __getitem__(self, idx):
        example = self.examples[idx]
        qid = example["qid"]
        assert len(example["docids"]) >= self.nway, (self.nway, len(example["docids"]))
        if self.smtid_as_docid:
            smt_smtids = example["smt_smtids"][:self.nway]
            lex_smtids = example["lex_smtids"][:self.nway]
        else:
            docids = example["docids"][:self.nway]
            smt_smtids = [self.smt_docid_to_smtid[str(x)] for x in docids]
            lex_smtids = [self.lex_docid_to_smtid[str(x)] for x in docids]
        scores = example["scores"][:self.nway]

        smt_doc_encoding = self._get_2d_encoding(smt_smtids)
        lex_doc_encoding = self._get_2d_encoding(lex_smtids)
        queries = [self.query_dataset[str(qid)][1].strip() for _ in range(self.nway)]

        assert len(smt_doc_encoding[0]) in {4,8,16}, len(smt_doc_encoding[0])
        assert len(lex_doc_encoding[0]) in {64}
        assert smt_doc_encoding[0][0] >= 32000
        
        return queries, lex_doc_encoding, smt_doc_encoding, scores

class RiporForSeq2seqDataset(Dataset):
    def __init__(self, example_path, docid_to_smtid_path):
        with open(docid_to_smtid_path) as fin:
            docid_to_smtid = ujson.load(fin)

        self.examples = []
        with open(example_path) as fin:
            for i, line in tqdm(enumerate(fin)):
                example = ujson.loads(line)
                docid, query = example["docid"], example["query"]
                smtid = docid_to_smtid[docid]
                self.examples.append((query, smtid))
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, smtid = self.examples[idx]

        assert len(smtid) in {4, 8, 16}
        assert smtid[0] != -1 and smtid[0] >= 32000, smtid
        
        return query, smtid
    
class LexicalRiporForDensePretrainedMarginMSEDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir,
                 lex_docid_to_smtid_path):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        with open(lex_docid_to_smtid_path) as fin:
            self.lex_docid_to_smtid = ujson.load(fin)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        qid = example["qid"]
        query = self.query_dataset[str(qid)][1]
        
        pos_docid  = example["docids"][0]
        pos_doc = self.document_dataset[str(pos_docid)][1]
        s_pos = example["scores"][0]

        neg_idx = random.sample(range(1, len(example["docids"])), k=1)[0]
        neg_docid = example["docids"][neg_idx]
        neg_doc = self.document_dataset[str(neg_docid)][1]
        s_neg = example["scores"][neg_idx]

        pos_doc_encoding = self.lex_docid_to_smtid[str(pos_docid)]
        neg_doc_encoding = self.lex_docid_to_smtid[str(neg_docid)]
        
        assert len(pos_doc_encoding) == len(neg_doc_encoding)
        assert len(pos_doc_encoding) in {64}, len(pos_doc_encoding)

        return query, pos_doc, neg_doc, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg
    
class LexicalRiporForSeq2seqDataset(Dataset):
    def __init__(self, example_path, smt_docid_to_smtid_path,
                 lex_docid_to_smtid_path):
        with open(smt_docid_to_smtid_path) as fin:
            smt_docid_to_smtid = ujson.load(fin)
        
        with open(lex_docid_to_smtid_path) as fin:
            lex_docid_to_smtid = ujson.load(fin)

        self.examples = []
        with open(example_path) as fin:
            for i, line in tqdm(enumerate(fin)):
                example = ujson.loads(line)
                docid, query = example["docid"], example["query"]
                
                lex_smtid = lex_docid_to_smtid[docid]
                smt_smtid = smt_docid_to_smtid[docid]
                self.examples.append((query, lex_smtid, smt_smtid))
        print("len of example: ", len(self.examples))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, lex_smtid, smt_smtid = self.examples[idx]

        return query, lex_smtid, smt_smtid