from typing import Dict

from tqdm import tqdm
from util import data_io
from skge import HolE, StochasticTrainer
from triple_data import get_id, build_trees_dataset, Trees


def label_triple(s,p,o,trees:Trees):
    if o in trees.get(s,{}).get(p,{}):
        label = 1
    else:
        label = 0

    return label

def build_triples(trees:Trees,ent2id:dict,rel2id:dict):
    triples = [(s, o, p) for s in ent2id.values() for p in rel2id.values() for o in
               ent2id.values()][:1000]
    print('enumerated %d triples'%len(triples))
    targets = [label_triple(s,p,o,trees) for s,p,o in tqdm(triples)]
    return triples,targets


if __name__ == '__main__':
    build_path = lambda ds: "../MultiHopKG/data/umls/%s.triples" % ds
    triple_files = {ds: build_path(ds) for ds in ["train", "dev", "test"]}
    data = build_trees_dataset(triple_files)
    triples = {name:build_triples(ds,data.ent2id,data.rel2id) for name,ds in data.dataset2trees.items()}
    N = len(data.ent2id.keys())
    M = len(data.rel2id.keys())
    xs,ys = triples['train']

    # instantiate HolE with an embedding space of size 100
    model = HolE((N, N, M), 100)
    trainer = StochasticTrainer(model,max_epochs=10)
    trainer.fit(xs, ys)