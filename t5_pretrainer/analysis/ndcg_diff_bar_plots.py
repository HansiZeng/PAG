import ujson 
import os 
import numpy as np
import matplotlib.pyplot as plt
import pytrec_eval 

def get_mrr_10(reldocids, docids):
    best_rank = 1001 
    for rel_docid in reldocids:
        if rel_docid in docids:
            best_rank = min(best_rank, docids.index(rel_docid) + 1)
    
    if best_rank > 10:
        mrr = 0. 
    else:
        mrr = 1. / best_rank 

    return mrr 

def get_mrr_10_diff(qrels, lex_run, other_run):
    mrr_diffs = []

    for qid in lex_run:
        lex_docids = [str(x) for x, _ in sorted(lex_run[qid].items(), key=lambda x: x[1], reverse=True)]
        other_docids = [str(x) for x, _ in sorted(other_run[qid].items(), key=lambda x: x[1], reverse=True)]

        lex_mrr = get_mrr_10(qrels[qid], lex_docids)
        other_mrr = get_mrr_10(qrels[qid], other_docids)

        mrr_diffs.append(other_mrr - lex_mrr)
    
    return mrr_diffs

def get_ndcg_diffs(lex_run_dir, other_run_dir):
    ndcg_diffs = []
    with open("./data/msmarco-full/TREC_DL_2019/qrel.json") as fin:
        trec_19_rel = ujson.load(fin)
    
    with open("./data/msmarco-full/TREC_DL_2020/qrel.json") as fin:
        trec_20_rel = ujson.load(fin)

    for qrel, run_name in [(trec_19_rel, "TREC_DL_2019"), (trec_20_rel, "TREC_DL_2020")]:
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"ndcg_cut"})
        with open(os.path.join(lex_run_dir, f"{run_name}/run.json")) as fin:
            lex_run = ujson.load(fin)
        lex_perf = evaluator.evaluate(lex_run)

        with open(os.path.join(other_run_dir, f"{run_name}/run.json")) as fin:
            other_run = ujson.load(fin)
        other_perf = evaluator.evaluate(other_run)

        for qid in lex_perf:
            diff = other_perf[qid]["ndcg_cut_10"] - lex_perf[qid]["ndcg_cut_10"]
            ndcg_diffs.append(diff)

    print("length of ndcg diffs = {}".format(len(ndcg_diffs)))

    return ndcg_diffs




if __name__ == "__main__":

    lex_run_path = "./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/all_lex_rets/lex_ret_1000/MSMARCO/run.json"
    qrels_path = "./data/msmarco-full/dev_qrel.json"
    other_run_path = "./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/all_lex_rets/lex_ret_1000/ltmp_smt_ret_100/MSMARCO/run.json"

    lex_run_dir = "./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/all_lex_rets/lex_ret_1000/"
    other_run_dir = "./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/all_lex_rets/lex_ret_1000/ltmp_smt_ret_100/"

    with open(lex_run_path) as fin:
        lex_run = ujson.load(fin)

    with open(qrels_path) as fin:
        qrels = ujson.load(fin)

    with open(other_run_path) as fin:
        other_run = ujson.load(fin)

    # for MSMARCO Dev
    mrr_diffs = get_mrr_10_diff(qrels, lex_run, other_run)

    mrr_diffs = np.array(sorted(mrr_diffs[:6980]))
    sub_mrr_diffs = mrr_diffs[np.arange(0, len(mrr_diffs), 1)]

    # for TREC'19 and 20
    ndcg_diffs = get_ndcg_diffs(lex_run_dir, other_run_dir)

    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].bar(range(len(sub_mrr_diffs)), sub_mrr_diffs, edgecolor='lightskyblue')
    axs[0].set_title("MSMARCO Dev", fontsize=20)
    axs[0].set_ylabel("∆MRR@10", fontsize=18)
    axs[0].set_xlabel("Query No.", fontsize=18)

    ndcg_diffs = sorted(ndcg_diffs)
    axs[1].bar(range(len(ndcg_diffs)), ndcg_diffs, color="salmon")
    axs[1].set_title("TREC 19/20 Combine", fontsize=20)
    axs[1].set_ylabel("∆NDCG@10", fontsize=18)
    axs[1].set_xlabel("Query No.", fontsize=18)

    axs[0].set_ylim(-1.1,1.1)
    axs[1].set_ylim(-.5,.5)

    plt.tight_layout()
    plt.savefig("./t5_pretrainer/analysis/images/mrr_ndcg_diff_bar_plots.jpg")
    """
    plt.bar(range(len(sub_mrr_diffs)), sub_mrr_diffs,  edgecolor='violet')
    plt.title("MSMARCO Dev")
    plt.ylabel("∆MRR@10")
    plt.xlabel("Document No.")
    plt.tight_layout()

    plt.savefig("./analysis/images/mrr_diff_bar_plots.jpg")

    plt.cla()

    # for TREC'19 and 20
    ndcg_diffs = get_ndcg_diffs(lex_run_dir, other_run_dir)

    ndcg_diffs = sorted(ndcg_diffs)
    plt.bar(range(len(ndcg_diffs)), ndcg_diffs, color="salmon")#, alpha=0.6)
    plt.title("TREC 19/20 Combine")
    plt.ylabel("∆NDCG@10")
    plt.xlabel("Document No.")
    plt.tight_layout()

    plt.savefig("./analysis/images/ndcg_diff_bar_plots.jpg")
    """