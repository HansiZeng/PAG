import faiss 
import numpy as np 
import argparse
import os
import ujson

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        default="",
                        type=str)
    parser.add_argument("--M", default=32, type=int)
    parser.add_argument("--bits", default=8, type=int)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--mmap_dir", default=None, type=str)
    parser.add_argument("--id_to_key_path", default=None, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model_dir = args.model_dir 
    M = args.M #8
    bits = args.bits #11

    #assert str(2**bits) in args.model_dir, (bits, args.model_dir)
    print("model_dir: ", model_dir, "codebook_num: ", M, "codebook_size: ", 2**bits)
    #assert str(8) in model_dir and str(2**bits) in model_dir
    if args.mmap_dir is not None:
        mmap_dir = args.mmap_dir
    else:
        mmap_dir = os.path.join(model_dir, "mmap")
    print("mmap_dir: ", mmap_dir)
    
    idx_to_docid = {}
    with open(os.path.join(mmap_dir, "text_ids.tsv")) as fin:
        for i, line in enumerate(fin):
            docid = line.strip()
            idx_to_docid[i] = docid 
    print("size of idx_to_docid = {}".format(len(idx_to_docid)))

    # for beir dataset
    if args.id_to_key_path is not None: 
        with open(args.id_to_key_path) as fin:
            idx_to_key = ujson.load(fin)

    doc_embeds = np.memmap(os.path.join(mmap_dir, "doc_embeds.mmap"), dtype=np.float32, mode="r").reshape(-1,768)
    index = faiss.read_index(os.path.join(model_dir, "aq_index/model.index"))
    rq = index.rq
    
    doc_encodings = []
    unit8_codes = rq.compute_codes(doc_embeds)
    for u8_code in unit8_codes:
        bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), unit8_codes.shape[1])
        code = []
        for i in range(M): 
            code.append(bs.read(bits)) 
        doc_encodings.append(code)
        assert len(doc_encodings[-1]) == M, (len(doc_encodings[-1]), M)

    docid_to_smtid = {}
    for idx, doc_enc in enumerate(doc_encodings):
        docid = idx_to_docid[idx]
        if args.id_to_key_path is not None: 
            docid = idx_to_key[docid]
        docid_to_smtid[docid] = doc_enc
    
    print("size of docid_to_smtid = {}".format(len(docid_to_smtid)))

    if args.out_dir is not None:
        out_dir = args.out_dir 
    else:
        assert NotImplementedError
        out_dir = os.path.join(model_dir, "aq_smtid")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open(os.path.join(out_dir, "docid_to_smtid.json"), "w") as fout:
        ujson.dump(docid_to_smtid, fout)

    smtid_to_docids = {}
    for docid, smtids in docid_to_smtid.items():
        smtid = "_".join([str(x) for x in smtids])
        if smtid not in smtid_to_docids:
            smtid_to_docids[smtid] = [docid]
        else:
            smtid_to_docids[smtid] += [docid]
    
    total_smtid = len(smtid_to_docids)
    lengths = np.array([len(x) for x in smtid_to_docids.values()])
    unique_smtid_num = np.sum(lengths == 1)
    print("unique_smtid_num = {}, total_smtid = {}".format(unique_smtid_num, total_smtid))
    print("percentage of smtid is unique = {:.3f}".format(unique_smtid_num / total_smtid))
    print("distribution of lengths: ", np.quantile(lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))

    