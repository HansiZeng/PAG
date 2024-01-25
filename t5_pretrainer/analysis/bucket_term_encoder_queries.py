import ujson 
import os 
import numpy as np

def rerank_from_other_retriever(rank_to_qids, other_run, qrels):
    rank_to_other_ranks = {1: [], 5: [], 10: [], 50: [], 100: [], 1000: []}
    for rank, qids in rank_to_qids.items():
        for qid in qids:
            docids = [str(x) for x, _ in sorted(other_run[qid].items(), key=lambda x: x[1], reverse=True)]

            best_rank = 1001 
            for rel_docid in list(qrels[qid].keys()):
                if rel_docid in docids:
                    best_rank = min(best_rank, docids.index(rel_docid) + 1)


            rank_to_other_ranks[rank].append(best_rank)

    # sanity check 
    for rank in rank_to_qids:
        assert len(rank_to_qids[rank]) == len(rank_to_other_ranks[rank])

    return rank_to_other_ranks



if __name__ == "__main__":
    lex_run_path = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/all_lex_rets/lex_ret_1000/MSMARCO/run.json"
    qrels_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json"
    other_run_path = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/all_lex_rets/lex_ret_1000/ltmp_smt_ret_100/MSMARCO/run.json"

    with open(lex_run_path) as fin:
        lex_run = ujson.load(fin)

    with open(qrels_path) as fin:
        qrels = ujson.load(fin)

    with open(other_run_path) as fin:
        other_run = ujson.load(fin)

    rank_to_qids = {1: [], 5: [], 10: [], 50: [], 100: [], 1000: []}
    rank_to_ranks = {1: [], 5: [], 10: [], 50: [], 100: [], 1000: []}

    for qid in lex_run:
        docids = [str(x) for x, _ in sorted(lex_run[qid].items(), key=lambda x: x[1], reverse=True)]

        best_rank = 1001 
        for rel_docid in list(qrels[qid].keys()):
            if rel_docid in docids:
                best_rank = min(best_rank, docids.index(rel_docid) + 1)
        
        if best_rank <= 1:
            rank_to_qids[1].append(qid)
            rank_to_ranks[1].append(best_rank)
        elif best_rank <= 5:
            rank_to_qids[5].append(qid)
            rank_to_ranks[5].append(best_rank)
        elif best_rank <= 10:
            rank_to_qids[10].append(qid)
            rank_to_ranks[10].append(best_rank)
        elif best_rank <= 50:
            rank_to_qids[50].append(qid)
            rank_to_ranks[50].append(best_rank)
        elif best_rank <= 100:
            rank_to_qids[100].append(qid)
            rank_to_ranks[100].append(best_rank)
        else:
            rank_to_qids[1000].append(qid)
            rank_to_ranks[1000].append(best_rank)

    rank_to_size = {k: len(v) for k, v in rank_to_qids.items()}
    print(rank_to_size)
    print("size of queries = {}".format(len(lex_run)))
    print("size of counted qids: ", np.sum([len(xs) for xs in rank_to_qids.values()]))


    rank_to_other_ranks = rerank_from_other_retriever(rank_to_qids, other_run, qrels)

    print("for lexical rank: ")
    for rank, ranks in rank_to_ranks.items():
        print("rank: {}, average rank: {:.3f}".format(rank, np.mean(ranks)))

    print("for other rank: ")
    for rank, ranks in rank_to_other_ranks.items():
        print("rank: {}, average rank: {:.3f}".format(rank, np.mean(ranks)))
        
        
        
