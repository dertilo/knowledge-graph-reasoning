import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
from tqdm import tqdm
from util import data_io, util_methods

from pytorch_util import build_BatchingDataLoader

Trees = Dict[int, Dict[int, Set[int]]]
Branch = Tuple[int, int, Set[int]]


@dataclass
class TreesDataset:
    ent2id: Dict[str, int]
    rel2id: Dict[str, int]
    dataset2trees: Dict[str, Trees]


def get_id(x2id: Dict, x):
    if x not in x2id.keys():
        x2id[x] = len(x2id)
    return x2id[x]


def build_trees_dataset(triple_files: Dict[str, str]):
    ent2id, rel2id = {}, {}

    def build_trees(file,dataset_name) -> Trees:
        trees = {}

        def add_triple(subj, predi, obje, trees: Trees):
            if subj not in trees:
                trees[subj] = {}
            if predi not in trees[subj]:
                trees[subj][predi] = set()
            trees[subj][predi].add(obje)

        for line in data_io.read_lines(file):
            s, o, p = line.strip().split("\t")
            s_id, o_id, p_id = get_id(ent2id, s), get_id(ent2id, o), get_id(rel2id, p)
            add_triple(s_id, p_id, o_id, trees)
            if 'train' in dataset_name:
                p_inv_id = get_id(rel2id, p + "_inv")
                add_triple(o_id, p_inv_id, s_id, trees)

        return trees

    ds2tr = {
        dataset_name: build_trees(triple_file,dataset_name)
        for dataset_name, triple_file in triple_files.items()
    }
    return TreesDataset(ent2id, rel2id, ds2tr)



def build_batch_iter(trees: Trees, batch_size=32):

    branches: List[Branch] = [
        (e1, r, e2s) for e1, r_to_e2s in trees.items() for r, e2s in r_to_e2s.items()
    ]
    random.shuffle(branches)
    return iter(
        raw_batch
        for raw_batch in util_methods.iterable_to_batches(branches, batch_size)
    )


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
    data = build_trees_dataset(triple_files)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_entities = len(data.ent2id.keys())
    get_batch_fun = build_resetting_next_fun(
        lambda: build_batch_iter(data.dataset2trees["dev"], 32)
    )

    dl = build_BatchingDataLoader(get_batch_fun=get_batch_fun)

    for epoch in range(3):
        for raw_batch in tqdm(dl):
            raw_batch
