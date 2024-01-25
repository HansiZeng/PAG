import ujson 
import os 
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    lex_run_path = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/all_lex_rets/lex_ret_1000/MSMARCO/run.json"
    qrels_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json"
    other_run_path = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/all_lex_rets/lex_ret_1000/ltmp_smt_ret_100/MSMARCO/run.json"
    next_run_path = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/ripor_direct_lng_knp_seq2seq_1/sub_tokenid_8_out_1000/MSMARCO/run.json"

    with open(lex_run_path) as fin:
        lex_run = ujson.load(fin)

    with open(qrels_path) as fin:
        qrels = ujson.load(fin)

    with open(other_run_path) as fin:
        other_run = ujson.load(fin)

    with open(next_run_path) as fin:
        next_run = ujson.load(fin)

    mrr_diffs = get_mrr_10_diff(qrels, lex_run, other_run)
    next_mrr_diffs = get_mrr_10_diff(qrels, lex_run, next_run)

    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    #print(np.quantile(mrr_diffs, quantiles))
    print(np.mean(mrr_diffs))

    #bins = [-1.0, -0.5, -0.3, -0.1, -0.05, 0., 0.05, 0.1, 0.3, 0.5, 1.0]
    #show_edges = np.array([-1., -.8, -.6, -.4, -.2, 0.] + [.2, .4, .6, .8, 1.0])
    bins = [-1.0, -.5, -.1, -.05, -.01, 0., .01, 0.05, .1, .5, 1.0]
    show_edges = np.array([-1., -.8, -.6, -.4, -.2, 0.] + [.2, .4, .6, .8, 1.0])

    bar_width = 0.08  # Adjust as needed for appearance

    # first bar
    hist, bin_edges = np.histogram(mrr_diffs, bins=bins)
    hist = hist / np.sum(hist)
    bin_centers = 0.5 * (show_edges[:-1] + show_edges[1:]) - 0.05

    plt.bar(bin_centers, hist, width=bar_width, edgecolor='black', label="lexical_ripor", alpha=0.6)

    # second bar 
    hist, bin_edges = np.histogram(next_mrr_diffs, bins=bins)
    hist = hist / np.sum(hist)
    bin_centers = 0.5 * (show_edges[:-1] + show_edges[1:]) + 0.05

    plt.bar(bin_centers, hist, width=bar_width, edgecolor='black', label="ripor", alpha=0.6)

    # Set x-ticks to be at bin centers
    #plt.xticks(bin_centers, labels=[f'{edge:.2f}' for edge in bin_edges[:-1]])
    plt.xticks(show_edges, labels=bins)

    # Small gaps between bars
    plt.xlim(bin_edges[0] - bar_width, bin_edges[-1] + bar_width)

    # Add titles and labels
    plt.title('Histogram of MRR Differences')
    plt.xlabel('MRR Difference')
    plt.ylabel('Frequency')
    plt.legend()

    plt.savefig("./analysis/images/query_wise_perf_comp.jpg")

    


