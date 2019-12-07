import math
import random
from dataclasses import dataclass
from typing import Dict, List, NamedTuple

import torch
import numpy as np
from tqdm import tqdm
from util import data_io, util_methods

from pytorch_util import build_BatchingDataLoader


class Triple(NamedTuple):
    subject: int
    predicate: int
    object: int


@dataclass
class TripleDataset:
    ent2id: Dict[str, int]
    rel2id: Dict[str, int]
    dataset2triples: Dict[str, List[Triple]]


def get_id(x2id: Dict, x):
    if x not in x2id.keys():
        x2id[x] = len(x2id)
    return x2id[x]


def build_triple_dataset(triple_files: Dict[str, str]):
    ent2id, rel2id = {}, {}

    def build_triples(file):
        def process_line(line):
            s, o, p = line.strip().split("\t")
            s_id, o_id, p_id = get_id(ent2id, s), get_id(ent2id, o), get_id(rel2id, p)
            return Triple(s_id, p_id, o_id)

        return [process_line(line) for line in data_io.read_lines(file)]

    ds2tr = {
        dataset_name: build_triples(triple_file)
        for dataset_name, triple_file in triple_files.items()
    }
    return TripleDataset(ent2id, rel2id, ds2tr)


def rand_ent_except(num_ent, ent):
    rand_ent = random.randint(0, num_ent - 1)
    while rand_ent == ent:
        rand_ent = random.randint(0, num_ent - 1)
    return rand_ent


def build_batch(raw_batch: List[Triple], num_ents,neg_ratio = 1):
    #TODO(tilo): neg_ratio ??

    pos_batch = np.array([[tr.subject, tr.predicate, tr.object] for tr in raw_batch])

    neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
    for i in range(len(neg_batch)):
        if random.random() < 0.5:
            neg_batch[i][0] = rand_ent_except(
                num_ents, neg_batch[i][0]
            )  # flipping head
        else:
            neg_batch[i][2] = rand_ent_except(
                num_ents, neg_batch[i][2]
            )  # flipping tail

    input_batch = np.append(pos_batch, neg_batch, axis=0)
    neg_targets = np.zeros((neg_batch.shape[0], 1)) # zeros or -1 ??
    pos_targets = np.ones((pos_batch.shape[0], 1))
    target_batch = np.concatenate(
        [pos_targets, neg_targets]
    )

    input_tensor = torch.tensor(input_batch).float().to(device)
    target_tensor = torch.tensor(target_batch).float().to(device)
    return input_tensor, target_tensor


def build_batch_iter(triples: List[Triple], num_ents, batch_size=32):
    random.shuffle(triples)
    g = (
        build_batch(raw_batch, num_ents)
        for raw_batch in util_methods.iterable_to_batches(triples, batch_size)
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

    get_batch_fun = build_resetting_next_fun(
        lambda: build_batch_iter(
            data.dataset2triples["dev"], len(data.ent2id.keys()), 32
        )
    )

    dl = build_BatchingDataLoader(get_batch_fun=get_batch_fun)

    for epoch in range(3):
        for batch in tqdm(dl):
            pass
