import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, T5Tokenizer
import numpy as np
from ..utils.utils import flatten_list
from copy import deepcopy

class DataLoaderWrapper(DataLoader):
    def __init__(self, tokenizer_type, max_length, **kwargs):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        """
        try:
            print("use auto tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        except:
            print("use t5 tokenizer")
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_type, cache_dir='cache')
        """
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        raise NotImplementedError("must implement this method")

class CollectionDataLoader(DataLoaderWrapper):
    """
    """

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True)
        return {**{k: torch.tensor(v) for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}

class LexicalConditionCollectionDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True,
                                           return_tensors="pt")
        
        start_token_id = self.tokenizer.pad_token_id
        batch_size =  processed_passage["input_ids"].shape[0]
        processed_passage["decoder_input_ids"] = torch.hstack(
            (processed_passage["input_ids"], torch.full((batch_size, 1), start_token_id, dtype=torch.long)))
        
        return {**{k: v for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}
        

class T5SpladeCollectionDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True)
        
        processed_passage["decoder_input_ids"] = deepcopy(processed_passage["input_ids"])

        return {**{k: torch.tensor(v) for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}

    
class T5DenseCollectionDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True,
                                           return_tensors="pt")
        
        # add special token for decoder_input_ids
        start_token_id = self.tokenizer.pad_token_id
        batch_size =  processed_passage["input_ids"].shape[0]
        processed_passage["decoder_input_ids"] = torch.full((batch_size, 1), start_token_id, dtype=torch.long)

        return {**{k: v for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}
    
class LexicalRiporDenseCollectionDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True,
                                           return_tensors="pt")
        
        # add special token for decoder_input_ids

        batch_size = processed_passage["input_ids"].shape[0]
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        processed_passage["decoder_input_ids"] = torch.hstack((processed_passage["input_ids"], prefix_tensor))

        return {**{k: v for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}
    
class CollectionDataLoaderForRiporGeneration(DataLoaderWrapper):
    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True,
                                           return_tensors="pt")
        
        # add special token for decoder_input_ids
        start_token_id = self.tokenizer.pad_token_id
        batch_size =  processed_passage["input_ids"].shape[0]
        processed_passage["decoder_input_ids"] = torch.full((batch_size, 1), start_token_id, dtype=torch.long)

        return {**{k: v for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}
    
class RerankforT5SeqAQLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        qids, docids, queries, doc_encodings, query_decoder_input_ids = [list(xs) for xs in zip(*batch)]

        tokenized_query = self.tokenizer(queries, 
                                         add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        tokenized_query["decoder_input_ids"] = torch.LongTensor(query_decoder_input_ids)
        pair_ids = [(x,y) for x, y in zip(qids, docids)]

        return {
            "tokenized_query": tokenized_query,
            "doc_encoding": torch.LongTensor(doc_encodings),
            "pair_ids": pair_ids
        }
    
class CollectionDataWithDocIDLoader(DataLoaderWrapper):
    """
    """

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d, smtid = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to self.max_length
                                           max_length=self.max_length,
                                           return_attention_mask=True)
        processed_passage.update({"decoder_input_ids": torch.LongTensor(list(smtid))})
        return {**{k: torch.tensor(v) for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long)}

class CollectionAQLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        docid, doc_encoding = [list(x) for x in zip(*batch)]

        return {
            "docids": list(docid),
            "doc_encodings": torch.LongTensor(doc_encoding)
        }

class QueryToSmtidRerankLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        queries, qids, strsmtids, decoder_input_ids, labels = [list(x) for x in zip(*batch)]

        tokenized_query = self.tokenizer(queries,
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")

        pair_ids = []
        for qid, str_smtid in zip(qids, strsmtids):
            pair_ids.append((qid, str_smtid))

        tokenized_query.update({"decoder_input_ids": torch.LongTensor(decoder_input_ids)})

        return {
            "tokenized_query": tokenized_query,
            "pair_ids": pair_ids,
            "labels": torch.LongTensor(labels)
        }
    
class CrossEncRerankDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        pair_ids, queries, docs = [], [], []
        for elem in batch:
            pair_ids.append(elem["pair_id"])
            queries.append(elem["query"])
            docs.append(elem["doc"])
        
        qd_kwargs = self.tokenizer(queries, docs, padding=True, truncation='longest_first', 
                                    return_attention_mask=True, return_tensors="pt", max_length=self.max_length)
        return {"pair_ids": pair_ids, "qd_kwargs": qd_kwargs}
    
class PseudoQueryForScoreDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        qids, docids, queries, docs = zip(*batch)
        
        tokenized_qd = self.tokenizer(list(queries), list(docs), padding=True,
                                    truncation='longest_first', 
                                    return_attention_mask=True, return_tensors="pt", max_length=self.max_length)
        pair_ids = []
        for qid, docid in zip(qids, docids):
            pair_ids.append((qid, docid))

        return {
            "qd_kwargs": tokenized_qd,
            "pair_ids": pair_ids
        }

class CrossEncRerankForSamePrefixPairLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        triple_ids, queries, docs = [], [], []
        for elem in batch:
            triple_ids.append(elem["triple_id"])
            queries.append(elem["query"])
            docs.append(elem["doc"])
        
        qd_kwargs = self.tokenizer(queries, docs, padding=True, truncation='longest_first', 
                                    return_attention_mask=True, return_tensors="pt", max_length=self.max_length)
        return {"triple_ids": triple_ids, "qd_kwargs": qd_kwargs}

class CondPrevSmtidT5SeqEncRerankLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        queries, docs, pair_ids, prev_smtids = [], [], [], []
        for elem in batch:
            queries.append(elem["query"])
            docs.append(elem["doc"])
            pair_ids.append(elem["pair_id"])
            prev_smtids.append(elem["prev_smtids"])

        tokenized_query = self.tokenizer(queries,
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        tokenized_doc = self.tokenizer(docs,
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        tokenized_query.update({"decoder_input_ids": torch.LongTensor(prev_smtids)})
        tokenized_doc.update({"decoder_input_ids": torch.LongTensor(prev_smtids)})

        return {
            "tokenized_query": tokenized_query,
            "tokenized_doc": tokenized_doc,
            "prev_smtids": torch.LongTensor(prev_smtids)[:, 1:].clone(),
            "pair_ids": pair_ids
        }

class LexicalRiporRerankDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        qid, docid, query, lex_doc_encoding, smt_doc_encoding = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        doc_encoding = torch.hstack((torch.LongTensor(lex_doc_encoding), torch.LongTensor(smt_doc_encoding)))

        batch_size = tokenized_query["input_ids"].size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        tokenized_query["decoder_input_ids"] = torch.hstack((tokenized_query["input_ids"],
                                                   prefix_tensor,
                                                   torch.LongTensor(smt_doc_encoding)[:,:-1]))

        return {
            "qid": torch.LongTensor(qid), 
            "docid": torch.LongTensor(docid),
            "tokenized_query": tokenized_query,
            "doc_encoding": doc_encoding,
            "lexical_encoding_size": len(lex_doc_encoding[0])
        }
    
class LexicalRiporDensePretrainedRerankDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        qid, docid, query, doc, lex_doc_encoding = zip(*batch)

        tokenized_query = self.tokenizer(list(query),
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        tokenized_doc = self.tokenizer(list(doc),
                            add_special_tokens=True,
                            padding="longest",  # pad to max sequence length in batch
                            truncation="longest_first",  # truncates to self.max_length
                            max_length=self.max_length,
                            return_attention_mask=True,
                            return_tensors="pt")

        doc_encoding = torch.LongTensor(lex_doc_encoding)

        batch_size = tokenized_query["input_ids"].size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)

        tokenized_query["decoder_input_ids"] = torch.hstack((tokenized_query["input_ids"], prefix_tensor))
        tokenized_doc["decoder_input_ids"] = torch.hstack((tokenized_doc["input_ids"], prefix_tensor))

        return {
            "qid": torch.LongTensor(qid), 
            "docid": torch.LongTensor(docid),
            "tokenized_query": tokenized_query,
            "tokenized_doc": tokenized_doc,
            "doc_encoding": doc_encoding
        }