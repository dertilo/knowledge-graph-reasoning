import math
import random

import torch
import numpy as np
from tqdm import tqdm

from pytorch_util import build_BatchingDataLoader


class TripleDataset:
    def __init__(self, triple_file_name):
        self.ent2id = {}
        self.rel2id = {}
        self.batch_index = 0
        self.data = self.read(triple_file_name)

    def read(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        triples = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            s, o, p = line.strip().split("\t")
            triples[i] = np.array(self.triple2ids((s, p, o)))
        return triples

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def triple2ids(self, triple):
        return [
            self.get_ent_id(triple[0]),
            self.get_rel_id(triple[1]),
            self.get_ent_id(triple[2]),
        ]

    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    def rand_ent_except(self, ent):
        rand_ent = random.randint(0, self.num_ent() - 1)
        while rand_ent == ent:
            rand_ent = random.randint(0, self.num_ent() - 1)
        return rand_ent

    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data):
            batch = self.data[self.batch_index : self.batch_index + batch_size]
            self.batch_index += batch_size
        elif self.batch_index < len(self.data):
            batch = self.data[self.batch_index :]
            self.batch_index = len(self.data)
        else:
            self.batch_index = 0
            raise StopIteration
        return np.append(batch, np.ones((len(batch), 1)), axis=1).astype(
            "int"
        )  # appending the +1 label

    def generate_neg(self, pos_batch, neg_ratio):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        for i in range(len(neg_batch)):
            if random.random() < 0.5:
                neg_batch[i][0] = self.rand_ent_except(neg_batch[i][0])  # flipping head
            else:
                neg_batch[i][2] = self.rand_ent_except(neg_batch[i][2])  # flipping tail
        neg_batch[:, -1] = -1
        return neg_batch

    def next_batch(self, batch_size, neg_ratio, device):
        pos_batch = self.next_pos_batch(batch_size)
        neg_batch = self.generate_neg(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        np.random.shuffle(batch)
        heads = torch.tensor(batch[:, 0]).long().to(device)
        rels = torch.tensor(batch[:, 1]).long().to(device)
        tails = torch.tensor(batch[:, 2]).long().to(device)
        labels = torch.tensor(batch[:, 3]).float().to(device)
        return heads, rels, tails, labels

    def was_last_batch(self):
        return self.batch_index == 0

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))


if __name__ == "__main__":

    triple_file = "../MultiHopKG/data/umls/dev.triples"
    triples = TripleDataset(triple_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dl = build_BatchingDataLoader(
        get_batch_fun=lambda _: triples.next_batch(32, 10, device)
    )

    for epoch in range(3):
        for batch in tqdm(dl):
            pass
