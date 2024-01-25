import torch 
from modeling.t5_generative_retriever import Ripor
import torch.nn as nn
import faiss 
import os
from transformers import AutoTokenizer
import argparse
import ujson 
from tqdm import tqdm 
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from utils.prefixer import generate_special_token_list

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", 
                        default="../data/experiments-full-lexical-ripor/ripor_direct_lng_knp_seq2seq_1",
                        type=str)
    parser.add_argument("--other_model_dir",
                        default="../data/experiments-full-lexical-ripor/t5-term-encoder-1-5e-4-12l/",
                        type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    pretrained_path = os.path.join(args.model_dir, "checkpoint")
    other_pretrained_path = os.path.join(args.other_model_dir, "checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(other_pretrained_path)
    print("size of tokenizer = {}".format(len(tokenizer)))

    model = T5ForConditionalGeneration.from_pretrained(pretrained_path)
    other_model = T5ForConditionalGeneration.from_pretrained(other_pretrained_path)

    new_model = T5ForConditionalGeneration.from_pretrained(pretrained_path)

    print("embed_weight for model: {}, embed_weight for other_model: {}".format(
        model.get_input_embeddings().weight.size(), 
        other_model.get_input_embeddings().weight.size()
    ))

    for (name1, param1), (name2, param2), (avg_name, avg_param) in zip(model.named_parameters(), 
                                                           other_model.named_parameters(), new_model.named_parameters()):
        if name1 == "shared.weight":
            assert name1 == name2 == avg_name
            assert param1.shape == avg_param.shape 
            assert param1.size()[0] > param2.size()[0]
            print("name1: ", name1)

            avg_param.data[:len(tokenizer)].copy_((param1.data[:len(tokenizer)] + param2.data[:len(tokenizer)]) / 2)
            avg_param.data[len(tokenizer):].copy_(param1.data[len(tokenizer):]) 
        else:
            assert name1 == name2 == avg_name
            assert param1.shape == param2.shape == avg_param.shape, (param1.shape, param2.shape, avg_param.shape)
            avg_param.data.copy_((param1.data + param2.data) / 2)

    out_dir = os.path.join(args.model_dir, "merged_checkpoint")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    new_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
