from typing import Any
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
import numpy as np
from copy import deepcopy

from ..utils.utils import flatten_list


# for training   
class MarginMSECollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        query, pos_doc, neg_doc, pos_score, neg_score = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        pos_tokenized_doc = self.tokenizer(list(pos_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        neg_tokenized_doc = self.tokenizer(list(neg_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        teacher_pos_scores = torch.FloatTensor(pos_score)
        teacher_neg_scores = torch.FloatTensor(neg_score)

        return {
            "tokenized_query": tokenized_query,
            "pos_tokenized_doc": pos_tokenized_doc,
            "neg_tokenized_doc": neg_tokenized_doc,
            "teacher_pos_scores": teacher_pos_scores,
            "teacher_neg_scores": teacher_neg_scores
        }

class T5DenseMarginMSECollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        query, pos_doc, neg_doc, pos_score, neg_score = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        pos_tokenized_doc = self.tokenizer(list(pos_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        neg_tokenized_doc = self.tokenizer(list(neg_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")

        # add special token for decoder_input_ids
        start_token_id = self.tokenizer.pad_token_id
        batch_size = tokenized_query["input_ids"].shape[0]
        #print("start_token_idx for T5: {}".format(start_token_id))
        
        decoder_input_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long)
        tokenized_query["decoder_input_ids"] = decoder_input_ids
        pos_tokenized_doc["decoder_input_ids"] = deepcopy(decoder_input_ids)
        neg_tokenized_doc["decoder_input_ids"] = deepcopy(decoder_input_ids)
        
        # teacher score
        teacher_pos_scores = torch.FloatTensor(pos_score)
        teacher_neg_scores = torch.FloatTensor(neg_score)

        return {
            "tokenized_query": tokenized_query,
            "pos_tokenized_doc": pos_tokenized_doc,
            "neg_tokenized_doc": neg_tokenized_doc,
            "teacher_pos_scores": teacher_pos_scores,
            "teacher_neg_scores": teacher_neg_scores
        }
    
class T5SpladeMarginMSECollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        query, pos_doc, neg_doc, pos_score, neg_score = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        pos_tokenized_doc = self.tokenizer(list(pos_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        neg_tokenized_doc = self.tokenizer(list(neg_doc),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        tokenized_query["decoder_input_ids"] = deepcopy(tokenized_query["input_ids"])
        pos_tokenized_doc["decoder_input_ids"] = deepcopy(pos_tokenized_doc["input_ids"])
        neg_tokenized_doc["decoder_input_ids"] = deepcopy(neg_tokenized_doc["input_ids"])

        #tokenized_query["decoder_attention_mask"] = deepcopy(tokenized_query["attention_mask"])
        #pos_tokenized_doc["decoder_attention_mask"] = deepcopy(pos_tokenized_doc["attention_mask"])
        #neg_tokenized_doc["decoder_attention_mask"] = deepcopy(neg_tokenized_doc["attention_mask"])
        
        teacher_pos_scores = torch.FloatTensor(pos_score)
        teacher_neg_scores = torch.FloatTensor(neg_score)

        return {
            "tokenized_query": tokenized_query,
            "pos_tokenized_doc": pos_tokenized_doc,
            "neg_tokenized_doc": neg_tokenized_doc,
            "teacher_pos_scores": teacher_pos_scores,
            "teacher_neg_scores": teacher_neg_scores
        }

class TermEncoderForMarginMSECollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        query, pos_doc_encoding, neg_doc_encoding, pos_score, neg_score = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        
        pos_doc_encoding = torch.LongTensor(pos_doc_encoding)
        neg_doc_encoding = torch.LongTensor(neg_doc_encoding)
        
        teacher_pos_scores = torch.FloatTensor(pos_score)
        teacher_neg_scores = torch.FloatTensor(neg_score)

        return {
            "tokenized_query": tokenized_query,
            "pos_doc_encoding": pos_doc_encoding,
            "neg_doc_encoding": neg_doc_encoding,
            "teacher_pos_scores": teacher_pos_scores,
            "teacher_neg_scores": teacher_neg_scores
        }

class T5TermEncoderForMarginMSECollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        query, pos_doc_encoding, neg_doc_encoding, pos_score, neg_score = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        tokenized_query["decoder_input_ids"] = deepcopy(tokenized_query["input_ids"])
        
        pos_doc_encoding = torch.LongTensor(pos_doc_encoding)
        neg_doc_encoding = torch.LongTensor(neg_doc_encoding)
        
        teacher_pos_scores = torch.FloatTensor(pos_score)
        teacher_neg_scores = torch.FloatTensor(neg_score)

        return {
            "tokenized_query": tokenized_query,
            "pos_doc_encoding": pos_doc_encoding,
            "neg_doc_encoding": neg_doc_encoding,
            "teacher_pos_scores": teacher_pos_scores,
            "teacher_neg_scores": teacher_neg_scores
        }

class RiporForMarginMSECollator:
    """ Can also be used for T5AQDataset """
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    
    def __call__(self, batch):
        q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg = [list(x) for x in zip(*batch)]
        assert pos_doc_encoding[0][0] >= len(self.tokenizer), (pos_doc_encoding, len(self.tokenizer))

        q_pos = self.tokenizer(q_pos,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        q_neg = self.tokenizer(q_neg,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        pos_doc_encoding = torch.LongTensor(pos_doc_encoding)
        neg_doc_encoding = torch.LongTensor(neg_doc_encoding)
        batch_size = q_pos["input_ids"].size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        q_pos["decoder_input_ids"] = torch.hstack((prefix_tensor, pos_doc_encoding[:, :-1]))
        q_neg["decoder_input_ids"] = torch.hstack((prefix_tensor, neg_doc_encoding[:, :-1]))

        return {
            "pos_tokenized_query": q_pos,
            "neg_tokenized_query": q_neg,
            "pos_doc_encoding": pos_doc_encoding,
            "neg_doc_encoding": neg_doc_encoding,
            "teacher_pos_scores": torch.FloatTensor(s_pos),
            "teacher_neg_scores": torch.FloatTensor(s_neg)
        }
    
class RiporForLngKnpMarginMSECollator:
    """ Can also be used for T5AQDataset """
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    
    def __call__(self, batch):
        q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, smtid_4_s_pos, smtid_4_s_neg = [list(x) for x in zip(*batch)]
        assert pos_doc_encoding[0][0] >= len(self.tokenizer), (pos_doc_encoding, len(self.tokenizer))

        q_pos = self.tokenizer(q_pos,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        q_neg = self.tokenizer(q_neg,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        pos_doc_encoding = torch.LongTensor(pos_doc_encoding)
        neg_doc_encoding = torch.LongTensor(neg_doc_encoding)
        batch_size = q_pos["input_ids"].size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        q_pos["decoder_input_ids"] = torch.hstack((prefix_tensor, pos_doc_encoding[:, :-1]))
        q_neg["decoder_input_ids"] = torch.hstack((prefix_tensor, neg_doc_encoding[:, :-1]))

        return {
            "pos_tokenized_query": q_pos,
            "neg_tokenized_query": q_neg,
            "pos_doc_encoding": pos_doc_encoding,
            "neg_doc_encoding": neg_doc_encoding,
            "teacher_pos_scores": torch.FloatTensor(s_pos),
            "teacher_neg_scores": torch.FloatTensor(s_neg),
            "smtid_4_teacher_pos_scores": torch.FloatTensor(smtid_4_s_pos),
            "smtid_4_teacher_neg_scores": torch.FloatTensor(smtid_4_s_neg),
        }
    
class RiporForSeq2seqCollator():
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    
    def __call__(self, batch):
        query, smtid = [list(x) for x in zip(*batch)]

        tokenized_query = self.tokenizer(query,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        labels = torch.LongTensor(smtid)

        batch_size = labels.size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        tokenized_query["decoder_input_ids"] = torch.hstack((prefix_tensor, labels[:, :-1]))

        return {
            "tokenized_query": tokenized_query,
            "labels": labels,
        }


class LexicalRiporForMarginMSECollator:
    """ Can also be used for T5AQDataset """
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        q_pos, q_neg, lex_pos_doc_encoding, lex_neg_doc_encoding, smt_pos_doc_encoding, smt_neg_doc_encoding, \
            s_pos, s_neg = [list(x) for x in zip(*batch)]

        q_pos = self.tokenizer(q_pos,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        q_neg = self.tokenizer(q_neg,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        pos_doc_encoding = torch.hstack((torch.LongTensor(lex_pos_doc_encoding), torch.LongTensor(smt_pos_doc_encoding)))
        neg_doc_encoding = torch.hstack((torch.LongTensor(lex_neg_doc_encoding), torch.LongTensor(smt_neg_doc_encoding)))

        batch_size = q_pos["input_ids"].size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        q_pos["decoder_input_ids"] = torch.hstack((q_pos["input_ids"],
                                                   prefix_tensor,
                                                   torch.LongTensor(smt_pos_doc_encoding)[:,:-1]))
        q_neg["decoder_input_ids"] = torch.hstack((q_neg["input_ids"],
                                                   prefix_tensor,
                                                   torch.LongTensor(smt_neg_doc_encoding)[:,:-1]))
        
        lexical_encoding_size = len(lex_pos_doc_encoding[0])
        assert lexical_encoding_size in {16, 32, 64, 128}, lexical_encoding_size
        return {
            "pos_tokenized_query": q_pos,
            "neg_tokenized_query": q_neg,
            "pos_doc_encoding": pos_doc_encoding,
            "neg_doc_encoding": neg_doc_encoding,
            "teacher_pos_scores": torch.FloatTensor(s_pos),
            "teacher_neg_scores": torch.FloatTensor(s_neg),
            "lexical_encoding_size": lexical_encoding_size,
        }

class LexicalRiporForKLDivCollator:
    """ Can also be used for T5AQDataset """
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def _to_list_1d(self, list_2d):
        list_1d = []
        for ls in list_2d:
            for x in ls:
                list_1d.append(x)
        return list_1d

    def __call__(self, batch):
        queries, lex_doc_encoding, smt_doc_encoding, scores = [list(x) for x in zip(*batch)]

        tokenized_query = self.tokenizer(self._to_list_1d(queries),
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt") # [B, seq_length]
        scores = torch.FloatTensor(scores) #[bz, nway]
        bz, nway = scores.size()
        B = bz*nway

        lex_doc_encoding = torch.LongTensor(lex_doc_encoding).view(B, -1) #[B, 64]
        smt_doc_encoding = torch.LongTensor(smt_doc_encoding).view(B, -1) # [B, 8]
        doc_encoding = torch.hstack((lex_doc_encoding, smt_doc_encoding))

        assert tokenized_query["input_ids"].size(0) == len(lex_doc_encoding) == len(smt_doc_encoding)

        prefix_tensor = torch.full((B, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        tokenized_query["decoder_input_ids"] = torch.hstack((
                                                tokenized_query["input_ids"],
                                                prefix_tensor,
                                                smt_doc_encoding[:, :-1]
                                            ))
        lexical_encoding_size = len(lex_doc_encoding[0])
        assert lexical_encoding_size in {64}

        return {
            "tokenized_query": tokenized_query,
            "lexical_encoding_size": lexical_encoding_size,
            "doc_encoding": doc_encoding,
            "teacher_scores": scores
        }


class LexicalRiporForDensePretrainedMarginMSECollator:
    """ Can also be used for T5AQDataset """
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        query, pos_doc, neg_doc, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg = [list(x) for x in zip(*batch)]

        tokenized_query = self.tokenizer(query,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        
        tokenized_pos_doc = self.tokenizer(pos_doc,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        tokenized_neg_doc = self.tokenizer(neg_doc,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        pos_doc_encoding = torch.LongTensor(pos_doc_encoding)
        neg_doc_encoding = torch.LongTensor(neg_doc_encoding)

        batch_size = tokenized_query["input_ids"].size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)

        tokenized_query["decoder_input_ids"] = torch.hstack((tokenized_query["input_ids"], prefix_tensor))
        tokenized_pos_doc["decoder_input_ids"] = torch.hstack((tokenized_pos_doc["input_ids"], prefix_tensor))
        tokenized_neg_doc["decoder_input_ids"] = torch.hstack((tokenized_neg_doc["input_ids"], prefix_tensor))                           
        
        return {
            "tokenized_query": tokenized_query,
            "tokenized_pos_doc": tokenized_pos_doc,
            "tokenized_neg_doc": tokenized_neg_doc,
            "pos_doc_encoding": pos_doc_encoding,
            "neg_doc_encoding": neg_doc_encoding,
            "teacher_pos_scores": torch.FloatTensor(s_pos),
            "teacher_neg_scores": torch.FloatTensor(s_neg),
        }


class LexicalRiporForSeq2seqCollator:
    def __init__(self, tokenizer_type, max_length, apply_lex_loss=False):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.apply_lex_loss = apply_lex_loss

    def _generate_neg_encodings(self, lex_encodings, max_val=32000, ratio=4):
        bz, pos_size = lex_encodings.size()

        neg_size = pos_size * ratio
        lex_neg_encodings = torch.empty(bz, neg_size, dtype=lex_encodings.dtype)

        for i in range(bz):
            existing = set(lex_encodings[i].tolist())
            neg_encodings = []
            while len(neg_encodings) < neg_size:
                num = torch.randint(0, max_val, (1,))
                if num.item() not in existing:
                    neg_encodings.append(num)
            lex_neg_encodings[i] = torch.cat(neg_encodings)

        return lex_neg_encodings
    
    def __call__(self, batch):
        query, lex_smtid, smt_smtid = [list(x) for x in zip(*batch)]

        tokenized_query = self.tokenizer(query,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        

        batch_size = tokenized_query["input_ids"].size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)

        tokenized_query["decoder_input_ids"] = torch.hstack((tokenized_query["input_ids"], 
                                                             prefix_tensor,
                                                             torch.LongTensor(smt_smtid)[:, :-1]))

        assert len(lex_smtid[0]) == 64, lex_smtid[0]
        assert len(smt_smtid[0]) in {8} and smt_smtid[0][0] >= 32000, smt_smtid[0] 

        if self.apply_lex_loss:
            lex_encodings = torch.LongTensor(lex_smtid)
            lex_neg_encodings = self._generate_neg_encodings(lex_encodings)
        
            return {
                "tokenized_query": tokenized_query,
                "lex_encodings": lex_encodings,
                "lex_neg_encodings": lex_neg_encodings,
                "smt_labels": torch.LongTensor(smt_smtid)
            }
        else:
            return {
                "tokenized_query": tokenized_query,
                "smt_labels": torch.LongTensor(smt_smtid)
            }
