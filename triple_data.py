import math
import random
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Set, Tuple

import torch
import numpy as np
from ignite.utils import convert_tensor
from tqdm import tqdm
from util import data_io, util_methods

from kgr.conv_e import Config, ConvE
from pytorch_util import build_BatchingDataLoader


Trees = Dict[int, Dict[int, Set[int]]]
Branch = Tuple[int, int, Set[int]]


@dataclass
class TripleDataset:
    ent2id: Dict[str, int]
    rel2id: Dict[str, int]
    dataset2trees: Dict[str, Trees]


def get_id(x2id: Dict, x):
    if x not in x2id.keys():
        x2id[x] = len(x2id)
    return x2id[x]


def build_triple_dataset(triple_files: Dict[str, str]):
    ent2id, rel2id = {}, {}

    def build_triples(file) -> Trees:
        trees = {}

        def add_triple(subj, predi, obje, triple_dict: Trees):
            if subj not in triple_dict:
                triple_dict[subj] = {}
            if predi not in triple_dict[subj]:
                triple_dict[subj][predi] = set()
            triple_dict[subj][predi].add(obje)

        for line in data_io.read_lines(file):
            s, o, p = line.strip().split("\t")
            s_id, o_id, p_id = get_id(ent2id, s), get_id(ent2id, o), get_id(rel2id, p)
            p_inv_id = get_id(rel2id, p + "_inv")
            add_triple(s_id, p_id, o_id, trees)
            add_triple(o_id, p_inv_id, s_id, trees)

        return trees

    ds2tr = {
        dataset_name: build_triples(triple_file)
        for dataset_name, triple_file in triple_files.items()
    }
    return TripleDataset(ent2id, rel2id, ds2tr)


def convert_to_tensors(batch: List[Branch], num_entities):
    def convert_to_binary(batch):
        z = torch.zeros((len(batch), num_entities))
        for i, objs_idx in enumerate(batch):
            z[i][list(objs_idx)] = 1
        return z

    batch_e1, batch_r, batch_e2 = [list(l) for l in zip(*batch)]
    batch_e1 = torch.LongTensor(batch_e1)
    batch_r = torch.LongTensor(batch_r)
    batch_e2 = convert_to_binary(batch_e2)

    return batch_e1, batch_e2, batch_r


def build_batch_iter(trees: Trees, num_ents, batch_size=32):

    branches: List[Branch] = [
        (e1, r, e2s) for e1, r_to_e2s in trees.items() for r, e2s in r_to_e2s.items()
    ]
    random.shuffle(branches)
    g = (
        convert_to_tensors(raw_batch, num_ents)
        for raw_batch in util_methods.iterable_to_batches(branches, batch_size)
    )
    return iter(g)


def build_resetting_next_fun(iter_supplier):
    ita = [iter_supplier()]

    def get_next(_):
        try:
            batch = next(ita[0])
            return batch
        except StopIteration:
            ita[0] = iter_supplier()
            raise StopIteration

    return get_next


if __name__ == "__main__":

    build_path = lambda ds: "../MultiHopKG/data/umls/%s.triples" % ds
    triple_files = {ds: build_path(ds) for ds in ["train", "dev", "test"]}
    data = build_triple_dataset(triple_files)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_entities = len(data.ent2id.keys())
    get_batch_fun = build_resetting_next_fun(
        lambda: build_batch_iter(data.dataset2trees["dev"], num_entities, 32)
    )

    dl = build_BatchingDataLoader(get_batch_fun=get_batch_fun)

    num_relations = len(data.rel2id.keys())
    config = Config()
    model = ConvE(config, num_entities, num_relations)

    for epoch in range(3):
        for batch in tqdm(dl):
            batch = convert_tensor(list(batch))
