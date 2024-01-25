import sys 
import os 
import pickle 
import argparse
import glob
import logging
import time
import random
import yaml
from pathlib import Path
import shutil
from collections import OrderedDict, defaultdict
import warnings
import json

import faiss
import torch
import numpy as np 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, HfArgumentParser
import torch.distributed as dist
import ujson 
from tqdm import tqdm
from transformers.modeling_utils import unwrap_model
import numba
from torch.distributed import get_world_size


from ..losses.regulariaztion import L0, L1
from ..utils.utils import is_first_worker, makedir, to_list
from ..utils.inverted_index import IndexDictOfArray

class TermEncoderRetriever:
    def __init__(self, model, args):
        self.model = model
        self.model.eval()
        self.args = args 

    def get_doc_scores(self, pred_scores, doc_encodings):
        """
        Args:
            pred_scores: [bz, vocab_size]
            doc_encodings: [N, L]
        Returns:
            doc_scores: [bz, N]
        """
        doc_scores = []
        for i in range(0, len(doc_encodings), 1_000_000):
            batch_doc_encodings = doc_encodings[i: i+1_000_000]
            batch_doc_scores = pred_scores[:, batch_doc_encodings].sum(dim=-1) #[bz, 1_000_000]
            doc_scores.append(batch_doc_scores)
        doc_scores = torch.hstack(doc_scores)
        # Use advanced indexing to get the values from pred_scores
        #selected_scores = pred_scores[:, doc_encodings]  # shape: [bz, N, L]

        # Sum over the last dimension to get the document scores
        #doc_scores = selected_scores.sum(dim=-1)  # shape: [bz, N]

        return doc_scores


    def retrieve(self, collection_loader, docid_to_smtids, topk, out_dir, use_fp16=False, run_name=None):
        if is_first_worker():
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

        # get doc_encodings
        doc_encodings = []
        docids = []
        for docid, smtids in docid_to_smtids.items():
            assert len(smtids) in {16, 32, 64, 128}, smtids 
            doc_encodings.append(smtids)
            docids.append(docid)
        print("length of doc_encodings = {}, docids = {}".format(len(doc_encodings), len(docids)))
        doc_encodings = torch.LongTensor(doc_encodings).to(self.model.base_model.device)

        qid_to_rankdata = {}
        for i, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.to(self.model.base_model.device) for k, v in batch.items() if k != "id"}
                    batch_preds = self.model.encode(**inputs) #[bz, vocab_size]
                    if isinstance(batch_preds, tuple):
                        assert batch_preds[1] == None, batch_preds
                        assert len(batch_preds) == 2, len(batch_preds)
                        batch_preds = batch_preds[0]
                    elif isinstance(batch_preds, torch.Tensor):
                        pass 
                    else:
                        raise NotImplementedError 
                
                    batch_doc_scores = self.get_doc_scores(batch_preds, doc_encodings)
                    top_scores, top_idxes = torch.topk(batch_doc_scores, k=topk, dim=-1) 
                
            if isinstance(batch["id"], torch.LongTensor):
                query_ids = batch["id"].tolist()
            elif isinstance(batch["id"], list):
                query_ids = batch["id"]
            else:
                raise ValueError("query_ids with type {} is not valid".format(type(query_ids)))
            
            for qid, scores, idxes in zip(query_ids, top_scores, top_idxes):
                qid_to_rankdata[qid] = {}
                scores = scores.cpu().tolist()
                idxes = idxes.cpu().tolist()
                for s, idx in zip(scores, idxes):
                    qid_to_rankdata[qid][docids[idx]] = s 

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if run_name is None:
            with open(os.path.join(out_dir, "run.json"), "w") as fout:
                ujson.dump(qid_to_rankdata, fout)
        else:
            with open(os.path.join(out_dir, run_name), "w") as fout:
                ujson.dump(qid_to_rankdata, fout)

class DenseIndexing:
    def __init__(self, model, args):
        self.index_dir = args.index_dir
        if is_first_worker():
            if self.index_dir is not None:
                makedir(self.index_dir)
        self.model = model
        self.model.eval()
        self.args = args

    def index(self, collection_loader, use_fp16=False, hidden_dim=768):
        model = self.model
        
        index = faiss.IndexFlatIP(hidden_dim)
        index = faiss.IndexIDMap(index)
        
        for idx, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    reps = model(**inputs)          
                    text_ids = batch["id"].numpy()
            
            index.add_with_ids(reps.cpu().numpy().astype(np.float32), text_ids)
            
        return index
                    
    def store_embs(self, collection_loader, local_rank, chunk_size=50_000, use_fp16=False, is_query=False, idx_to_id=None):
        model = self.model
        index_dir = self.index_dir
        write_freq = chunk_size // collection_loader.batch_size
        if is_first_worker():
            print("write_freq: {}, batch_size: {}, chunk_size: {}".format(write_freq, collection_loader.batch_size, chunk_size))
        
        embeddings = []
        embeddings_ids = []
        
        chunk_idx = 0
        for idx, batch in tqdm(enumerate(collection_loader), disable= not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                                total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.to(model.device) for k, v in batch.items() if k != "id"}
                    if is_query:
                        raise NotImplementedError
                    else:
                        reps = unwrap_model(model).doc_encode(**inputs)
                    #reps = model(**inputs)            
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)

            if (idx + 1) % write_freq == 0:
                embeddings = np.concatenate(embeddings)
                embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
                assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))

                text_path = os.path.join(index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx))
                id_path = os.path.join(index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx))
                np.save(text_path, embeddings)
                np.save(id_path, embeddings_ids)

                del embeddings, embeddings_ids
                embeddings, embeddings_ids = [], []

                chunk_idx += 1 
        
        if len(embeddings) != 0:
            embeddings = np.concatenate(embeddings)
            embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
            assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))
            print("last embedddings shape = {}".format(embeddings.shape))
            text_path = os.path.join(index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx))
            id_path = os.path.join(index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx))
            np.save(text_path, embeddings)
            np.save(id_path, embeddings_ids)

            del embeddings, embeddings_ids
            chunk_idx += 1 
            
        plan = {"nranks": dist.get_world_size(), "num_chunks": chunk_idx, "index_path": os.path.join(index_dir, "model.index")}
        print("plan: ", plan)
        
        if is_first_worker():
            with open(os.path.join(self.index_dir, "plan.json"), "w") as fout:
                ujson.dump(plan, fout)

    def stat_sparse_project_encoder(self, collection_loader, use_fp16=False, apply_log_relu_logit=False):
        model = self.model 
        l0_scores, l1_scores = [], []
        docid_to_info = {}
        l0_fn = L0()
        l1_fn = L1()
        for i, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.to(model.device) for k, v in batch.items() if k != "id"}
                    _, logits = unwrap_model(model).doc_encode_and_logit(**inputs)

            if apply_log_relu_logit:
                l0_scores.append(l0_fn(
                    torch.log(1 + torch.relu(logits))
                    ).cpu().item())
                l1_scores.append(l1_fn(
                    torch.log(1 + torch.relu(logits))
                    ).cpu().item())
            else:
                l0_scores.append(l0_fn(logits).cpu().item())
                l1_scores.append(l1_fn(logits).cpu().item())

            top_scores, top_cids = torch.topk(logits, k=128, dim=1) 
            top_scores, top_cids = top_scores.cpu().tolist(), top_cids.cpu().tolist()
            for docid, scores, cids in zip(batch["id"], top_scores, top_cids):
                docid_to_info[docid.cpu().item()] = {"scores": scores, "cids": cids}
        
        return docid_to_info, np.mean(l0_scores), np.mean(l1_scores)
            

    @staticmethod
    def aggregate_embs_to_index(index_dir):
        with open(os.path.join(index_dir, "plan.json")) as fin:
            plan = ujson.load(fin)
            
        print("index_dir is: {}".format(index_dir))
        print("plan: ", plan)
        
        nranks = plan["nranks"]
        num_chunks = plan["num_chunks"]
        index_path = plan["index_path"]
        
        # start index
        text_embs, text_ids = [], []
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                text_embs.append(np.load(os.path.join(index_dir,"embs_{}_{}.npy".format(i, chunk_idx))))
                text_ids.append(np.load(os.path.join(index_dir,"ids_{}_{}.npy".format(i, chunk_idx))))

        text_embs = np.concatenate(text_embs)
        text_embs = text_embs.astype(np.float32) if text_embs.dtype == np.float16 else text_embs
        text_ids = np.concatenate(text_ids)

        assert len(text_embs) == len(text_ids), (len(text_embs), len(text_ids))
        assert text_ids.ndim == 1, text_ids.shape
        print("embs dtype: ", text_embs.dtype, "embs size: ", text_embs.shape)
        print("ids dtype: ", text_ids.dtype)
        
        index = faiss.IndexFlatIP(text_embs.shape[1])
        index = faiss.IndexIDMap(index)

        #assert isinstance(text_ids, list)
        #text_ids = np.array(text_ids)

        index.add_with_ids(text_embs, text_ids)
        faiss.write_index(index, index_path)

        meta = {"text_ids": text_ids, "num_embeddings": len(text_ids)}
        print("meta data for index: {}".format(meta))
        with open(os.path.join(index_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        # remove embs, ids
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                os.remove(os.path.join(index_dir,"embs_{}_{}.npy".format(i, chunk_idx)))
                os.remove(os.path.join(index_dir,"ids_{}_{}.npy".format(i, chunk_idx)))
                
    @staticmethod
    def aggregate_embs_to_mmap(mmap_dir):
        with open(os.path.join(mmap_dir, "plan.json")) as fin:
            plan = ujson.load(fin)
            
        print("mmap_dir is: {}".format(mmap_dir))
        print("plan: ", plan)
        
        nranks = plan["nranks"]
        num_chunks = plan["num_chunks"]
        index_path = plan["index_path"]
        
        # start index
        text_embs, text_ids = [], []
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                text_embs.append(np.load(os.path.join(mmap_dir,"embs_{}_{}.npy".format(i, chunk_idx))))
                text_ids.append(np.load(os.path.join(mmap_dir,"ids_{}_{}.npy".format(i, chunk_idx))))

        text_embs = np.concatenate(text_embs)
        text_embs = text_embs.astype(np.float32) if text_embs.dtype == np.float16 else text_embs
        text_ids = np.concatenate(text_ids)

        assert len(text_embs) == len(text_ids), (len(text_embs), len(text_ids))
        assert text_ids.ndim == 1, text_ids.shape
        print("embs dtype: ", text_embs.dtype, "embs size: ", text_embs.shape)
        print("ids dtype: ", text_ids.dtype)
        
        fp = np.memmap(os.path.join(mmap_dir, "doc_embeds.mmap"), dtype=np.float32, mode='w+', shape=text_embs.shape)

        total_num = 0
        chunksize = 5_000
        for i in range(0, len(text_embs), chunksize):
            # generate some data or load a chunk of your data here. Replace `np.random.rand(chunksize, shape[1])` with your data.
            data_chunk = text_embs[i: i+chunksize]
            total_num += len(data_chunk)
            # make sure that the last chunk, which might be smaller than chunksize, is handled correctly
            if data_chunk.shape[0] != chunksize:
                fp[i:i+data_chunk.shape[0]] = data_chunk
            else:
                fp[i:i+chunksize] = data_chunk
        assert total_num == len(text_embs), (total_num, len(text_embs))
        
        with open(os.path.join(mmap_dir, "text_ids.tsv"), "w") as fout:
            for tid in text_ids:
                fout.write(f"{tid}\n")
        
        meta = {"text_ids": text_ids, "num_embeddings": len(text_ids)}
        print("meta data for index: {}".format(meta))
        with open(os.path.join(mmap_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        # remove embs, ids
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                os.remove(os.path.join(mmap_dir,"embs_{}_{}.npy".format(i, chunk_idx)))
                os.remove(os.path.join(mmap_dir,"ids_{}_{}.npy".format(i, chunk_idx)))        
                
class DenseRetriever:
    def __init__(self, model, args, dataset_name, is_beir=False):
        self.index_dir = args.index_dir
        self.out_dir = os.path.join(args.out_dir, dataset_name) if (dataset_name is not None and not is_beir) \
            else args.out_dir
        if is_first_worker():
            makedir(self.out_dir)
        if self.index_dir is not None:
            self.index_path = os.path.join(self.index_dir, "model.index")
        self.model = model
        self.model.eval()
        self.args = args
        
    def retrieve(self, collection_loader, topk, save_run=True, index=None, use_fp16=False):
        query_embs, query_ids = self._get_embeddings_from_scratch(collection_loader, use_fp16=use_fp16, is_query=True)
        query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs
        

        if index is None:
            index = faiss.read_index(self.index_path)
            index = self._convert_index_to_gpu(index, list(range(8)), False)
        
        start_time = time.time()
        nn_scores, nn_doc_ids = self._index_retrieve(index, query_embs, topk, batch=128)
        print("Flat index time spend: {:.3f}".format(time.time()-start_time))
        
        qid_to_ranks = {}
        for qid, docids, scores in zip(query_ids, nn_doc_ids, nn_scores):
            for docid, s in zip(docids, scores):
                if str(qid) not in qid_to_ranks:
                    qid_to_ranks[str(qid)] = {str(docid): float(s)}
                else:
                    qid_to_ranks[str(qid)][str(docid)] = float(s)
        
        if save_run:
            with open(os.path.join(self.out_dir, "run.json"), "w") as fout:
                ujson.dump(qid_to_ranks, fout)
            return {"retrieval": qid_to_ranks}
        else:
            return {"retrieval": qid_to_ranks}

    @staticmethod
    def get_first_smtid(model, collection_loader, use_fp16=False, is_query=True):
        model = model.base_model 
        print("the flag for model.decoding: ", model.config.decoding)

        text_ids = []
        semantic_ids = []
        for _, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    logits = model(**inputs).logits[0] #[bz, vocab]
            smtids = torch.argmax(logits, dim=1).cpu().tolist() 
            text_ids.extend(batch["id"].tolist())
            semantic_ids.extend(smtids)
        
        return text_ids, semantic_ids
        
    def _get_embeddings_from_scratch(self, collection_loader, use_fp16=False, is_query=True):
        model = self.model
        
        embeddings = []
        embeddings_ids = []
        for _, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    #reps = model(**inputs)
                    if is_query:
                        reps = unwrap_model(model).query_encode(**inputs)
                    else:
                        raise NotImplementedError
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)


        embeddings = np.concatenate(embeddings)

        assert len(embeddings_ids) == embeddings.shape[0]
        assert isinstance(embeddings_ids[0], int)
        print(f"# nan in embeddings: {np.sum(np.isnan(embeddings))}")

        return embeddings, embeddings_ids
    
    def _convert_index_to_gpu(self, index, faiss_gpu_index, useFloat16=False):
        if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
            faiss_gpu_index = faiss_gpu_index[0]
        if isinstance(faiss_gpu_index, int):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024*1024*1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = useFloat16
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            gpu_resources = []
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256*1024*1024)
                gpu_resources.append(res)
            print(f"length of gpu_resources : {len(gpu_resources)}.")

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = useFloat16
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

        return index
    
    def _index_retrieve(self, index, query_embeddings, topk, batch=None):
        if batch is None:
            nn_scores, nearest_neighbors = index.search(query_embeddings, topk)
        else:
            query_offset_base = 0
            pbar = tqdm(total=len(query_embeddings))
            nearest_neighbors = []
            nn_scores = []
            while query_offset_base < len(query_embeddings):
                batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
                batch_nn_scores, batch_nn = index.search(batch_query_embeddings, topk)
                nearest_neighbors.extend(batch_nn.tolist())
                nn_scores.extend(batch_nn_scores.tolist())
                query_offset_base += len(batch_query_embeddings)
                pbar.update(len(batch_query_embeddings))
            pbar.close()

        return nn_scores, nearest_neighbors

class SparseIndexing:
    """sparse indexing
    """

    def __init__(self, model, config, device, compute_stats=False, dim_voc=None, force_new=True,
                 filename="array_index.h5py", **kwargs):
        self.model = model
        self.model.eval()
        self.index_dir = config["index_dir"] if config is not None else None
        self.sparse_index = IndexDictOfArray(self.index_dir, dim_voc=dim_voc, force_new=force_new, filename=filename)
        self.compute_stats = compute_stats
        self.device = device
        if self.compute_stats:
            self.l0 = L0()
        
        self.model.to(self.device)
        self.local_rank = self.device

        if torch.distributed.is_initialized():
            self.world_size = get_world_size()
            print("world_size: {}, local_rank: {}".format(self.world_size, self.local_rank))
        else:
            self.world_size = None


    def multi_process_index(self, collection_loader, id_dict=None):
        doc_ids = {}
        if self.compute_stats:
            stats = defaultdict(float)

        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader, total=len(collection_loader), disable=self.local_rank >= 1)):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id"}}
                batch_documents = self.model.encode(**inputs) #[bz, vocab_size]
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                row, col = torch.nonzero(batch_documents, as_tuple=True)
                data = batch_documents[row, col]
                
                # to let each process having a unique row_id
                new_row = []
                for r in row.cpu().numpy():
                    new_row.append((count + r)*self.world_size + self.local_rank)
                row = np.array(new_row)

                # update doc_ids map
                batch_ids = to_list(batch["id"])
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                for r, docid in zip(row, batch_ids):
                    doc_ids[r] = docid

                count += len(batch_ids)
                self.sparse_index.add_batch_document(row, col.cpu().numpy(), data.cpu().numpy(),
                                                     n_docs=len(batch_ids))

                if t == 10:
                    break

        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}
        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            print("done iterating over the corpus...")
            print("index contains {} posting lists".format(len(self.sparse_index)))
            print("index contains {} documents".format(len(doc_ids)))
            if self.compute_stats:
                with open(os.path.join(self.index_dir, "index_stats.json"), "w") as handler:
                    ujson.dump(stats, handler)
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key],
                                                                  dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out

    def index(self, collection_loader, id_dict=None):
        doc_ids = []
        if self.compute_stats:
            stats = defaultdict(float)
        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id"}}
                batch_documents = self.model.encode(**inputs) #[bz, vocab_size]
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                row, col = torch.nonzero(batch_documents, as_tuple=True)
                data = batch_documents[row, col]
                row = row + count
                batch_ids = to_list(batch["id"])
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                count += len(batch_ids)
                doc_ids.extend(batch_ids)
                self.sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                                     n_docs=len(batch_ids))
                
                
        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}
        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            print("done iterating over the corpus...")
            print("index contains {} posting lists".format(len(self.sparse_index)))
            print("index contains {} documents".format(len(doc_ids)))
            if self.compute_stats:
                with open(os.path.join(self.index_dir, "index_stats.json"), "w") as handler:
                    ujson.dump(stats, handler)
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key],
                                                                  dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out

class SparseRetrieval:
    """retrieval from SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, model, config, dim_voc, device, dataset_name=None, index_d=None, compute_stats=False, is_beir=False,
                 **kwargs):
        self.model = model 
        self.model.eval()

        assert ("index_dir" in config and index_d is None) or (
                "index_dir" not in config and index_d is not None)
        if "index_dir" in config:
            self.sparse_index = IndexDictOfArray(config["index_dir"], dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(config["index_dir"], "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = index_d["index"]
            self.doc_ids = index_d["ids_mapping"]
            for i in range(dim_voc):
                # missing keys (== posting lists), causing issues for retrieval => fill with empty
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = np.array([], dtype=np.int32)
                    self.sparse_index.index_doc_value[i] = np.array([], dtype=np.float32)
        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value

        
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
        self.doc_stats = index_d["stats"] if (index_d is not None and compute_stats) else None
        self.compute_stats = compute_stats
        if self.compute_stats:
            self.l0 = L0()

        self.device = device
        self.model.to(device)

    def retrieve(self, q_loader, top_k, name=None, return_d=False, id_dict=False, threshold=0):
        makedir(self.out_dir)
        if self.compute_stats:
            makedir(os.path.join(self.out_dir, "stats"))
        res = defaultdict(dict)
        if self.compute_stats:
            stats = defaultdict(float)
        with torch.no_grad():
            for t, batch in enumerate(tqdm(q_loader)):
                q_id = to_list(batch["id"])[0]
                if id_dict:
                    q_id = id_dict[q_id]
                inputs = {k: v for k, v in batch.items() if k not in {"id"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                query = self.model.encode(**inputs)  # we assume ONE query per batch here
                if self.compute_stats:
                    stats["L0_q"] += self.l0(query).item()
                # TODO: batched version for retrieval
                row, col = torch.nonzero(query, as_tuple=True)
                values = query[to_list(row), to_list(col)]
                filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                  self.numba_index_doc_values,
                                                                  col.cpu().numpy(),
                                                                  values.cpu().numpy().astype(np.float32),
                                                                  threshold=threshold,
                                                                  size_collection=self.sparse_index.nb_docs())
                # threshold set to 0 by default, could be better
                filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
                for id_, sc in zip(filtered_indexes, scores):
                    res[str(q_id)][str(self.doc_ids[id_])] = float(sc)
        
        # write to disk 
        if self.compute_stats:
            stats = {key: value / len(q_loader) for key, value in stats.items()}
        if self.compute_stats:
            with open(os.path.join(self.out_dir, "stats",
                                   "q_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                      "w") as handler:
                json.dump(stats, handler)
            if self.doc_stats is not None:
                with open(os.path.join(self.out_dir, "stats",
                                       "d_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                          "w") as handler:
                    json.dump(self.doc_stats, handler)
        with open(os.path.join(self.out_dir, "run{}.json".format("_iter_{}".format(name) if name is not None else "")),
                  "w") as handler:
            json.dump(res, handler)
        
        # return values
        if return_d:
            out = {"retrieval": res}
            if self.compute_stats:
                out["stats"] = stats if self.doc_stats is None else {**stats, **self.doc_stats}
            return out
        
class SparseApproxEvalWrapper:
    """
    wrapper for sparse indexer + retriever during training
    """

    def __init__(self, model, config, collection_loader, q_loader, **kwargs):
        self.model = model 
        self.config = config
        self.collection_loader = collection_loader
        self.q_loader = q_loader
        self.model_output_dim = self.model.module.output_dim if hasattr(self.model, "module") else self.model.output_dim

    def index_and_retrieve(self, i):
        self.model.eval()
        indexer = SparseIndexing(self.model, config=None, restore=False, compute_stats=True)
        sparse_index_d = indexer.index(self.collection_loader)
        retriever = SparseRetrieval(self.model, self.config, dim_voc=self.model_output_dim, index_d=sparse_index_d,
                                    restore=False, compute_stats=True)
        return retriever.retrieve(self.q_loader, top_k=self.config["top_k"], name=i, return_d=True)
    

class AddictvieQuantizeIndexer:
    def __init__(self, model, args):
        super().__init__()
        self.model = model 
        self.args = args

    @staticmethod
    def index(mmap_dir, codebook_num, index_dir, codebook_bits):
        doc_embeds = np.memmap(os.path.join(mmap_dir, "doc_embeds.mmap"), dtype=np.float32,mode="r").reshape(-1,768)
        M = codebook_num
        codebook_bits = codebook_bits
        print("M: {}, codebook_bits: {}".format(M, codebook_bits))
        faiss.omp_set_num_threads(32)
        index = faiss.IndexResidualQuantizer(doc_embeds.shape[1], M, codebook_bits, faiss.METRIC_INNER_PRODUCT)

        print("start training")
        print("shape of doc_embeds: ", doc_embeds.shape)
        index.verbose = True
        index.train(doc_embeds)
        index.add(doc_embeds)

        faiss.write_index(index, os.path.join(index_dir, "model.index"))

    def search(self, collection_dataloader, topk, index_path, out_dir, index_ids_path):
        query_embs, query_ids = self._get_embeddings_from_scratch(collection_dataloader, use_fp16=False, is_query=True)
        query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs

        index = faiss.read_index(index_path)
        #index = self._convert_index_to_gpu(index, list(range(8)), False)
        idx_to_docid = self._read_text_ids(index_ids_path)
        
        qid_to_rankdata = {}
        all_scores, all_idxes = index.search(query_embs, topk)
        for qid, scores, idxes in zip(query_ids, all_scores, all_idxes):
            docids = [idx_to_docid[idx] for idx in idxes]
            qid_to_rankdata[str(qid)] = {}
            for docid, score in zip(docids, scores):
                qid_to_rankdata[str(qid)][str(docid)] = float(score)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

    def flat_index_search(self, collection_loader, topk, index_path, out_dir):
        query_embs, query_ids = self._get_embeddings_from_scratch(collection_loader, use_fp16=False, is_query=True)
        query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs
        
        index = faiss.read_index(index_path)
        index = self._convert_index_to_gpu(index, list(range(8)), False)
        
        nn_scores, nn_doc_ids = index.search(query_embs, topk)

        qid_to_ranks = {}
        for qid, docids, scores in zip(query_ids, nn_doc_ids, nn_scores):
            for docid, s in zip(docids, scores):
                if str(qid) not in qid_to_ranks:
                    qid_to_ranks[str(qid)] = {str(docid): float(s)}
                else:
                    qid_to_ranks[str(qid)][str(docid)] = float(s)
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_ranks, fout)

    def _get_embeddings_from_scratch(self, collection_loader, use_fp16=False, is_query=True):
        model = self.model
        
        embeddings = []
        embeddings_ids = []
        for _, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    #reps = model(**inputs)
                    if is_query:
                        reps = unwrap_model(model).query_encode(**inputs)
                    else:
                        raise NotImplementedError
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)


        embeddings = np.concatenate(embeddings)

        assert len(embeddings_ids) == embeddings.shape[0]
        assert isinstance(embeddings_ids[0], int)
        print(f"# nan in embeddings: {np.sum(np.isnan(embeddings))}")

        return embeddings, embeddings_ids
    
    def _convert_index_to_gpu(self, index, faiss_gpu_index, useFloat16=False):
        if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
            faiss_gpu_index = faiss_gpu_index[0]
        if isinstance(faiss_gpu_index, int):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024*1024*1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = useFloat16
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            gpu_resources = []
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256*1024*1024)
                gpu_resources.append(res)
            print(f"length of gpu_resources : {len(gpu_resources)}.")

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = useFloat16
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

        return index
    
    def _read_text_ids(self, text_ids_path):
        idx_to_docid = {}
        with open(text_ids_path) as fin:
            for idx, line in enumerate(fin):
                docid = line.strip()
                idx_to_docid[idx] = docid
        
        print("size of idx_to_docid = {}".format(len(idx_to_docid)))
        return idx_to_docid
