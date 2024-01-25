import torch 
from ..modeling.t5_generative_retriever import Ripor
import torch.nn as nn
import faiss 
import os
from transformers import AutoTokenizer
import argparse
import ujson 
from tqdm import tqdm 

from ..utils.prefixer import generate_special_token_list

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", 
                        default=None,
                        type=str)
    parser.add_argument("--K",
                        default=256,
                        type=int)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model_dir = args.model_dir
    K = args.K
    print("model_dir: ", model_dir, "K: ", K)

    d_model = 768
    #if K != 256:
    #    assert str(K) in model_dir
    #else:
    #    print("using default value for K: ", K)

    pretrained_path = os.path.join(model_dir, "checkpoint")
    out_dir = os.path.join(model_dir, "extended_token_checkpoint")

    index_path = os.path.join(model_dir, "aq_index/model.index")

    index = faiss.read_index(index_path)
    rq = index.rq

    print("M, K, d_model are: ", rq.M, K, d_model)

    centroids = faiss.vector_to_array(rq.codebooks).reshape(rq.M, K, 768)
    centroids = torch.FloatTensor(centroids)
    print("centroids shape = {}".format(centroids.shape))

    model = Ripor.from_pretrained(pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    # change embed layers and corresponding config params
    print("original embedding size = {}".format(len(tokenizer)), model.base_model.get_input_embeddings().weight.size(0))

    new_tokens = generate_special_token_list(num_code=rq.M, codebook_size=K)
    tokenizer.add_tokens(new_tokens)

    assert len(new_tokens) == rq.M * K, (len(new_tokens), rq.M*K)

    # resize model embeds and assign new embeddings
    model.base_model.resize_token_embeddings(len(tokenizer))
    embedding_weight = model.base_model.get_input_embeddings().weight

    for i in range(rq.M):
        for j in range(K):
            # Token to find
            token = f"<docid_{i}_{j}>"

            # Find the index of the token in the tokenizer
            token_index = tokenizer.convert_tokens_to_ids(token)
            assert token_index >= 32100, token_index

            # Get the corresponding centroid embedding
            vec = centroids[i, j]
            
            # Assign the embedding vector to the corresponding index in the embedding weight
            embedding_weight.data[token_index] = vec
    
    print("new embedding size = {}".format(len(tokenizer)), model.base_model.get_input_embeddings().weight.size(0))

    # create docid_to_tokenids 
    with open(os.path.join(args.model_dir, "aq_smtid/docid_to_smtid.json")) as fin:
        docid_to_smtids = ujson.load(fin)
    
    docid_to_tokenids = {}
    for docid, smtids in tqdm(docid_to_smtids.items(), total=len(docid_to_smtids)):
        tokenids = []
        for i, j in enumerate(smtids):
            token = f"<docid_{i}_{j}>"
            tokenids.append(tokenizer.convert_tokens_to_ids(token))
        if docid == "0":
            print(tokenids)
        docid_to_tokenids[docid] = tokenids
    with open(os.path.join(args.model_dir, "aq_smtid/docid_to_tokenids.json"), "w") as fout:
        ujson.dump(docid_to_tokenids, fout)
    # save model 
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
