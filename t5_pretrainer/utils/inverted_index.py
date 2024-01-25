"""
code for inverted index based on arrays, powered by numba based retrieval
"""

import array
import json
import os
import pickle
from collections import defaultdict
import pickle

import h5py
import numpy as np
from tqdm.auto import tqdm
import ujson


def merge_inverted_indexes(index_dir, world_size):
    list_inv_indexes = []
    list_docids_map = []
    list_stats = []
    for rank in range(world_size):
        list_inv_indexes.append(IndexDictOfArray(index_path=index_dir + f"_{rank}"))
        with open(os.path.join(index_dir + f"_{rank}", "doc_ids.pkl"), "rb") as fin:
            list_docids_map.append(pickle.load(fin))
        
        with open(os.path.join(index_dir + f"_{rank}", "index_stats.json"), "r") as fin:
            list_stats.append(ujson.load(fin))

    inv_index = IndexDictOfArray(index_path=index_dir, force_new=True)
    row_to_docid = {}
    stats = {}
    for idx, sub_inv_index in enumerate(list_inv_indexes):
        for dim_id in tqdm(sub_inv_index.index_doc_id, total=len(sub_inv_index.index_doc_id)):
            #for row, score in zip(sub_inv_index.index_doc_id[dim_id], sub_inv_index.index_doc_value[dim_id]):
            #    inv_index.index_doc_id[dim_id].append(row)
            #    inv_index.index_doc_value[dim_id].append(score)
            if dim_id not in inv_index.index_doc_id:
                assert dim_id not in inv_index.index_doc_value
                inv_index.index_doc_id[dim_id] = sub_inv_index.index_doc_id[dim_id]
                inv_index.index_doc_value[dim_id] = sub_inv_index.index_doc_value[dim_id]
            else:
                inv_index.index_doc_id[dim_id] = np.concatenate((inv_index.index_doc_id[dim_id], sub_inv_index.index_doc_id[dim_id]))
                inv_index.index_doc_value[dim_id] = np.concatenate((inv_index.index_doc_value[dim_id], sub_inv_index.index_doc_value[dim_id]))

        # write docid_ids.pkl
        sub_row_to_docid = list_docids_map[idx]
        old_n = len(row_to_docid)
        row_to_docid.update(sub_row_to_docid)
        assert len(row_to_docid) - old_n == len(sub_row_to_docid), (len(row_to_docid) - old_n, len(sub_row_to_docid))

        # write index_stats  
        sub_stats = list_stats[idx]
        for k, v in sub_stats.items():
            if k not in stats:
                stats[k] = v / world_size
            else:
                stats[k] += v / world_size

        
    # save to disk
    inv_index.save()
    with open(os.path.join(index_dir, "doc_ids.pkl"), "wb") as fout:
        pickle.dump(row_to_docid, fout)

    with open(os.path.join(index_dir, "index_stats.json"), "w") as handler:
        ujson.dump(stats, handler)


class IndexDictOfArray:
    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None):
        if index_path is not None:
            self.index_path = index_path
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(self.index_path, filename)
            if os.path.exists(self.filename) and not force_new:
                print("index already exists, loading...")
                self.file = h5py.File(self.filename, "r")
                if dim_voc is not None:
                    dim = dim_voc
                else:
                    dim = self.file["dim"][()]
                self.index_doc_id = dict()
                self.index_doc_value = dict()
                for key in tqdm(range(dim)):
                    try:
                        self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)],
                                                          dtype=np.int32)
                        # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                        self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)],
                                                             dtype=np.float32)
                    except:
                        self.index_doc_id[key] = np.array([], dtype=np.int32)
                        self.index_doc_value[key] = np.array([], dtype=np.float32)
                self.file.close()
                del self.file
                print("done loading index...")
                doc_ids = pickle.load(open(os.path.join(self.index_path, "doc_ids.pkl"), "rb"))
                self.n = len(doc_ids)
            else:
                self.n = 0
                print("initializing new index...")
                self.index_doc_id = defaultdict(lambda: array.array("I"))
                self.index_doc_value = defaultdict(lambda: array.array("f"))
        else:
            self.n = 0
            print("initializing new index...")
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))

    def add_batch_document(self, row, col, data, n_docs=-1):
        """add a batch of documents to the index
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs
        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[dim_id].append(doc_id)
            self.index_doc_value[dim_id].append(value)

    def __len__(self):
        return len(self.index_doc_id)

    def nb_docs(self):
        return self.n

    def save(self, dim=None):
        print("converting to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        print("save to disk")
        with h5py.File(self.filename, "w") as f:
            if dim:
                f.create_dataset("dim", data=int(dim))
            else:
                f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            f.close()
        print("saving index distribution...")  # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))
