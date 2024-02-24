import ujson 
import pickle 
import os 
import gzip
from tqdm import tqdm
from copy import deepcopy
import numpy as np

teacher_score_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
with gzip.open(teacher_score_path, "rb") as fin:
    qid_to_reldocid_to_score = pickle.load(fin)
    
qrel_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/qrels.json"
with open(qrel_path) as fin:
    qid_to_reldocids = ujson.load(fin)

qid_to_rerank = {}
rerank_path = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/t5-full-dense-1-5e-4-12l/out/MSMARCO_TRAIN/qrel_added_merged_teacher_scores.json"
with open(rerank_path) as fin:
    for line in fin:
        example = ujson.loads(line)
        qid = example["qid"]
        qid_to_rerank[qid] = {}
        for docid, score in zip(example["docids"], example["scores"]):
            qid_to_rerank[qid][docid] = score

rerank_path_2 = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-term-encoder/t5-term-encoder-0/out/MSMARCO_TRAIN/qid_docids_teacher_scores.train.json"
with open(rerank_path_2) as fin:
    for line in fin: 
        example = ujson.loads(line)
        qid = example["qid"]
        for docid, score in zip(example["docids"], example["scores"]):
            if docid not in qid_to_rerank[qid]:
                qid_to_rerank[qid][docid] = score

train_examples = []
lengths = []
for qid, reldocids in tqdm(qid_to_reldocids.items(), total=len(qid_to_reldocids)):
    sorted_pairs = sorted(qid_to_rerank[qid].items(), key=lambda x: x[1], reverse=True)
    for rel_docid in reldocids:
        rel_score = qid_to_reldocid_to_score[int(qid)][int(rel_docid)]

        neg_docids, neg_scores = [], []
        for docid, score in sorted_pairs:
            if docid != rel_docid:
                neg_docids.append(docid)
                neg_scores.append(score)
        
        example = {
            "qid": qid,
            "docids": [rel_docid] + neg_docids[:200],
            "scores": [rel_score] + neg_scores[:200]
        }
        train_examples.append(example)
        lengths.append(1+len(neg_docids))

print("number of examples = {}".format(len(train_examples)))
print("distribution of lengths: ", np.quantile(lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
with open("/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-dense/t5-dense-1/out/MSMARCO_TRAIN/qrel_added_merged_teacher_scores.json", "w") as fout:
    for example in train_examples:
        fout.write(ujson.dumps(example) + "\n")