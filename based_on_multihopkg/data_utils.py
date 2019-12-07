"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Data processing utilities.
"""

import collections
from typing import NamedTuple, List, Tuple

import numpy as np
import os
import pickle

START_RELATION = "START_RELATION"
NO_OP_RELATION = "NO_OP_RELATION"
NO_OP_ENTITY = "NO_OP_ENTITY"
DUMMY_RELATION = "DUMMY_RELATION"
DUMMY_ENTITY = "DUMMY_ENTITY"

DUMMY_RELATION_ID = 0
START_RELATION_ID = 1
NO_OP_RELATION_ID = 2
DUMMY_ENTITY_ID = 0
NO_OP_ENTITY_ID = 1


def check_answer_ratio(examples):
    entity_dict = {}
    for e1, e2, r in examples:
        if not e1 in entity_dict:
            entity_dict[e1] = set()
        entity_dict[e1].add(e2)
    answer_ratio = 0
    for e1 in entity_dict:
        answer_ratio += len(entity_dict[e1])
    return answer_ratio / len(entity_dict)


def get_train_path(args):
    if "NELL" in args.data_dir:
        if not args.model.startswith("point"):
            if args.test:
                train_path = os.path.join(args.data_dir, "train.dev.large.triples")
            else:
                train_path = os.path.join(args.data_dir, "train.large.triples")
        else:
            if args.test:
                train_path = os.path.join(args.data_dir, "train.dev.triples")
            else:
                train_path = os.path.join(args.data_dir, "train.triples")
    else:
        train_path = os.path.join(args.data_dir, "train.triples")

    return train_path


class Fork(NamedTuple):
    subj: int
    objects: List[int]
    predi: int


def load_raw_triples(data_path):
    with open(data_path) as f:
        for line in f:
            yield line.strip().split()


def load_triples(
    train_path, dev_path, entity_index_path, relation_index_path,
) -> Tuple[List[Fork], List[Tuple]]:
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples_g = load_raw_triples(train_path)
    train_forks = build_forks(triples_g, triple2ids)

    dev_triples = [(triple2ids(*tr)) for tr in load_raw_triples(dev_path)]

    return train_forks, dev_triples


def build_forks(triples_g, triple2ids, add_reverse_relations=True) -> List[Fork]:
    triple_dict = {}

    for e1, e2, r in triples_g:
        e1_id, e2_id, r_id = triple2ids(e1, e2, r)

        add_triple(e1_id, r_id, e2_id, triple_dict)
        if add_reverse_relations:
            e2_id, e1_id, r_inv_id = triple2ids(e2, e1, r + "_inv")
            add_triple(e2_id, r_inv_id, e1_id, triple_dict)

    return [
        Fork(e1_id, list(triple_dict[e1_id][r_id]), r_id)
        for e1_id in triple_dict
        for r_id in triple_dict[e1_id]
    ]


def add_triple(subj, predi, obje, triple_dict):
    if subj not in triple_dict:
        triple_dict[subj] = {}
    if predi not in triple_dict[subj]:
        triple_dict[subj][predi] = set()
    triple_dict[subj][predi].add(obje)


def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index
