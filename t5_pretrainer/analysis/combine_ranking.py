import ujson 
import os 
import pytrec_eval
from pytrec_eval import RelevanceEvaluator
import numpy as np 

def truncate_run(run, k):
    """truncates run file to only contain top-k results for each query"""
    temp_d = {}
    for q_id in run:
        sorted_run = {k: v for k, v in sorted(run[q_id].items(), key=lambda item: item[1], reverse=True)}
        temp_d[q_id] = {k: sorted_run[k] for k in list(sorted_run.keys())[:k]}
    return temp_d


def mrr_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recip_rank"})
    truncated = truncate_run(run, k)
    
    mrr = evaluator.evaluate(truncated)
    if agg:
        mrr = sum([d["recip_rank"] for d in mrr.values()]) / max(1, len(mrr))
    return mrr

def recall_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recall"})
    out_eval = evaluator.evaluate(run)

    total_v = 0.
    included_k = f"recall_{k}"
    for q, v_dict in out_eval.items():
        for k, v in v_dict.items():
            if k == included_k:
                total_v += v 

    return total_v / len(out_eval) 

def merge_run(bow_run, ripor_run, alpha=1.0):
    new_run = {}
    assert len(bow_run) == len(ripor_run), (len(bow_run), len(ripor_run))

    avg_lengths = []
    for qid in bow_run:
        new_run[qid] = {}
        for docid, bow_score in bow_run[qid].items():
            if docid in ripor_run[qid]:
                ripor_score = ripor_run[qid][docid]

                score = bow_score * alpha + ripor_score
                new_run[qid][docid] = score 
        
        avg_lengths.append(len(new_run[qid]))

    #print("average number of docids per query = {}".format(np.mean(avg_lengths)))

    return new_run

if __name__ == "__main__":
    bow_run_path = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/t5-term-encoder-1-5e-4-12l/out/MSMARCO/run.json"
    ripor_run_path = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/ripor_seq2seq_1/sub_tokenid_8_out_1000/MSMARCO/run.json"
    qrel_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json"

    with open(bow_run_path) as fin:
        bow_run = ujson.load(fin)

    with open(ripor_run_path) as fin:
        ripor_run = ujson.load(fin)

    with open(qrel_path) as fin:
        qrel = ujson.load(fin)

    for alpha in [0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 2., 4., 6., 8., 10.]:
        merged_run = merge_run(bow_run=bow_run, ripor_run=ripor_run, alpha=alpha)

        mrr_10 = mrr_k(merged_run, qrel, k=10)
        recall_10 = recall_k(merged_run, qrel, k=10)

        print("alpha: ", alpha)
        print("mrr@10: {:.3f}, recall@10: {:.3f}".format(mrr_10, recall_10))
        print("="*100)


    