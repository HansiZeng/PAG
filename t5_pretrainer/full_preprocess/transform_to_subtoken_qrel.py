import ujson 
import os 

sub_token = 8
model_dir = "./data/experiments-full-lexical-ripor/t5-full-dense-1-5e-4-12l/"
docid_to_smtids_path = os.path.join(model_dir, "aq_smtid/docid_to_tokenids.json")

data_root_dir = "./data/msmarco-full"
in_qrels_paths = [os.path.join(data_root_dir, "dev_qrel.json"),
                  os.path.join(data_root_dir, "TREC_DL_2019", "qrel_binary.json"),
                  os.path.join(data_root_dir, "TREC_DL_2019", "qrel.json"),
                  os.path.join(data_root_dir, "TREC_DL_2020", "qrel_binary.json"),
                  os.path.join(data_root_dir, "TREC_DL_2020", "qrel.json")]

with open(docid_to_smtids_path) as fin:
    docid_to_smtids = ujson.load(fin)
    
root_dir = os.path.join(model_dir, f"msmarco-full/sub_token_{sub_token}")
os.makedirs(root_dir, exist_ok=True)

trec_19_dir = os.path.join(root_dir, "TREC_DL_2019")
trec_20_dir = os.path.join(root_dir, "TREC_DL_2020")
if not os.path.exists(trec_19_dir):
    os.mkdir(trec_19_dir)
if not os.path.exists(trec_20_dir):
    os.mkdir(trec_20_dir)

out_qrels_paths = [os.path.join(root_dir, "dev_qrel.json"), 
                   os.path.join(trec_19_dir, "qrel_binary.json"),
                   os.path.join(trec_19_dir, "qrel.json"),
                   os.path.join(trec_20_dir, "qrel_binary.json"),
                   os.path.join(trec_20_dir, "qrel.json")]

print(out_qrels_paths)

assert len(in_qrels_paths) == len(out_qrels_paths)

docid_to_sub_smtid = {}
for docid, smtids in docid_to_smtids.items():
    assert len(smtids) == 8 and smtids[0] != -1, smtids 
    
    sub_smtid = "_".join(str(x) for x in smtids[:sub_token])
    docid_to_sub_smtid[docid] = sub_smtid 
    

for i, qrels_path in enumerate(in_qrels_paths):
    with open(qrels_path) as fin:
        qrels = ujson.load(fin)
    qid_to_subsmtid_to_label = {}
    for qid in qrels:
        qid_to_subsmtid_to_label[qid] = {}
        for docid, label in qrels[qid].items():
            sub_smtid = docid_to_sub_smtid[docid]
            
            if sub_smtid in qid_to_subsmtid_to_label[qid]:
                qid_to_subsmtid_to_label[qid][sub_smtid] = max(qid_to_subsmtid_to_label[qid][sub_smtid], label)
            else:
                qid_to_subsmtid_to_label[qid][sub_smtid] = label
                
    out_path = out_qrels_paths[i]
    with open(out_path, "w") as fout:
        ujson.dump(qid_to_subsmtid_to_label, fout)
                
    
    