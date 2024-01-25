import numpy as np 
import argparse
import os 
import ujson 
from tqdm import tqdm 
from ..modeling.t5_generative_retriever import LexicalRipor
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

qrel_paths = ["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel.json",
              "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel.json"]

def read_qid_to_reldocids(qrel_paths):
    qid_to_reldocids = {}
    for qrel_path in qrel_paths:
        with open(qrel_path) as fin:
            qrel_data = ujson.load(fin)
        for qid in qrel_data:
            qid_to_reldocids[qid] = []
            for docid, s in qrel_data[qid].items():
                if s >= 2.:
                    qid_to_reldocids[qid].append(docid)

    return qid_to_reldocids

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lex_docid_to_smtid_path",
                        default="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0-12l/top_bow/docid_to_tokenids.json",
                        type=str)
    parser.add_argument("--model_dir",
                        default="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/",
                        type=str)
    parser.add_argument("--out_dir",
                        default="/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/t5_pretrainer/analysis/doc_cluster_images/",
                        type=str)
    args = parser.parse_args()

    return args 

if __name__ == "__main__":
    plt.rcParams['font.size'] = 14       # Default font size
    plt.rcParams['axes.labelsize'] = 16  # Font size for x and y labels
    plt.rcParams['axes.titlesize'] = 18  # Font size for title
    plt.rcParams['legend.fontsize'] = 14 # Font size for legend
    
    args = get_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    checkpoint_path = os.path.join(args.model_dir, "checkpoint")
    model = LexicalRipor.from_pretrained(checkpoint_path)
    model.cuda()

    with open(args.lex_docid_to_smtid_path) as fin:
        lex_docid_to_smtids = ujson.load(fin)

    
    # select 10 queries 
    qid_to_reldocids = read_qid_to_reldocids(qrel_paths)
    N = 20
    selected_qid_to_docids = {}
    for idx, (qid, reldocids) in enumerate(qid_to_reldocids.items()):
        if len(selected_qid_to_docids) == N:
            break 
        
        if len(reldocids) >=20 and len(reldocids) <= 40:
            selected_qid_to_docids[qid] = reldocids 
    
    print("size of selected_qid_to_docids: ", len(selected_qid_to_docids))
    assert len(selected_qid_to_docids) == N, (len(selected_qid_to_docids), N)

    # start to encode 
    labels = []
    doc_lex_encodings = []

    for qid, docids in selected_qid_to_docids.items():
        for docid in docids:
            labels.append(qid)
            doc_lex_encodings.append(lex_docid_to_smtids[docid])

    doc_reps = []
    for i in tqdm(range(0, len(labels), 32), total=len(labels)//32):
        batch_reps = np.array(doc_lex_encodings[i:i+32], dtype=np.float32)
        doc_reps.append(batch_reps)

    doc_reps = np.concatenate(doc_reps)
    assert len(doc_reps) == len(labels), (len(doc_reps), len(labels))

    tsne = TSNE(n_components=2, random_state=42)
    doc_reps_2d = tsne.fit_transform(doc_reps)

    # Plot
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=doc_reps_2d[:, 0], y=doc_reps_2d[:, 1], hue=labels, palette=f"tab{N}", legend=False)
    #plt.legend(title='Query ID', loc='upper left')
    plt.title(f"t-SNE visualization for bow encoding")
    plt.savefig(os.path.join(args.out_dir, f"doc_cluster_by_bow_one_hot.jpg"), format='jpg', dpi=300)
    plt.show()
    


    
