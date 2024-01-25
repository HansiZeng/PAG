import os 
import ujson 
from tqdm import tqdm
from collections import Counter, defaultdict
import pickle
import matplotlib.pyplot as plt 
import numpy as np

#lex_ret_run_path = ""
smt_ret_run_path = "../experiments-full-lexical-ripor/ripor_seq2seq_1/sub_tokenid_8_out_1000/MSMARCO/run.json"
smt_docid_to_smtid_path = "../experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/aq_smtid/docid_to_tokenids.json"
out_dir = "../experiments-full-lexical-ripor/ripor_seq2seq_1/analysis"
qrels_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json"

#with open(lex_ret_run_path) as fin:
#    lex_ret_run = ujson.laod(fin)

with open(smt_ret_run_path) as fin:
    smt_ret_run = ujson.load(fin)

with open(smt_docid_to_smtid_path) as fin:
    docid_to_smtis = ujson.load(fin)

with open(qrels_path) as fin:
    qrels = ujson.load(fin)

os.makedirs(out_dir, exist_ok=True)

rank_mrr_data = []
list_ranks = []
list_mrrs = []
prefix_rank_truncate = 100
for qid in tqdm(smt_ret_run, total=len(smt_ret_run)):
    docid_to_score = smt_ret_run[qid]
    top_docids = [x for x, _ in sorted(docid_to_score.items(), key=lambda x: x[1], reverse=True)]

    prefix_counter = Counter()
    for docid in top_docids:
        smtids = docid_to_smtis[docid]
        assert len(smtids) == 8 and smtids[0] != -1, smtids 

        prefix_key = "{}_{}".format(smtids[0], smtids[1])
        prefix_counter[prefix_key] += 1

    top_prefixes = [x for x,_ in prefix_counter.most_common(prefix_rank_truncate)]
    
    best_rank = 1001
    best_prefix = None
    for rel_docid in list(qrels[qid].keys()):
        if rel_docid in top_docids:
            rank = top_docids.index(rel_docid)
            if rank < best_rank:
                best_rank = rank 

                smtids = docid_to_smtis[rel_docid]
                best_prefix = "{}_{}".format(smtids[0], smtids[1])
        else:
            if best_prefix is None:
                smtids = docid_to_smtis[rel_docid]
                best_prefix = "{}_{}".format(smtids[0], smtids[1])
    
    mrr_10 = 1. / (best_rank + 1.)
    if mrr_10 < 0.1:
        mrr_10 = 0. 

    if best_prefix in top_prefixes:
        prefix_rank = top_prefixes.index(best_prefix) + 1 
    else:
        prefix_rank = prefix_rank_truncate + 1
    
    rank_mrr_data.append((prefix_rank, mrr_10))

with open(os.path.join(out_dir, "rank_mrr_data.pickle"), "wb") as fout:
    pickle.dump(rank_mrr_data, fout)

rank_mrrs = [list() for _ in range(prefix_rank_truncate+1)]
for rank, mrr in rank_mrr_data:
    rank_mrrs[rank-1].append(mrr)

length_dists = [len(xs) for xs in rank_mrrs]
print(length_dists)

mrr_means = np.array([np.mean(xs) for xs in rank_mrrs])
mrr_stds = np.array([np.std(xs) for xs in rank_mrrs])
x_values = range(1, prefix_rank_truncate+2)

plt.plot(x_values, mrr_means)
plt.fill_between(x_values, mrr_means - (mrr_stds / 2.), mrr_means + (mrr_stds /2.), alpha=.1)
#plt.xticks(x_values)
plt.savefig(os.path.join(out_dir, "rank_mrr_data.jpg"))



group_num = 20
assert (len(rank_mrrs) - 1) % group_num == 0, (len(rank_mrrs), group_num)
group_size = (len(rank_mrrs) - 1) // group_num

x_values = range(1, group_num+2)
group_means = []
group_stds = []
for i in range(0, len(rank_mrrs), group_size):

    group_mrrs = []
    for xs in rank_mrrs[i:i+group_size]:
        group_mrrs += xs 

    group_means.append(np.mean(group_mrrs))
    group_stds.append(np.std(group_mrrs))


group_means = np.array(group_means)
group_stds = np.array(group_stds)
plt.plot(x_values, group_means)
plt.fill_between(x_values, group_means - (group_stds / 2.), group_means + (group_stds / 2.), alpha=.1)
plt.savefig(os.path.join(out_dir, "group_rank_mrr_data.jpg"))
