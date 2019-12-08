from typing import List

import torch
import torch.nn as nn
from torch import optim

from kgr.conv_e import Config, ConvE
from pytorch_util import build_BatchingDataLoader
from triple_data import Branch, build_triple_dataset, build_resetting_next_fun, \
    build_batch_iter


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

    return batch_e1, batch_r,batch_e2

if __name__ == "__main__":

    build_path = lambda ds: "../MultiHopKG/data/umls/%s.triples" % ds
    triple_files = {ds: build_path(ds) for ds in ["train", "dev", "test"]}
    data = build_triple_dataset(triple_files)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_entities = len(data.ent2id.keys())
    get_batch_fun = build_resetting_next_fun(
        lambda: build_batch_iter(data.dataset2trees["dev"], 32)
    )
    dl = build_BatchingDataLoader(get_batch_fun=get_batch_fun)

    num_relations = len(data.rel2id.keys())
    config = Config()
    model = ConvE(config, num_entities, num_relations)

    label_smoothing_epsilon = 0.1
    loss_fun = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.003,
    )


    def train_one_batch(model: nn.Module, optimizer, raw_batch):
        optimizer.zero_grad()

        e1, r, e2 = convert_to_tensors(raw_batch, num_entities)
        pred_scores = model.forward(e1, r)
        e2_label = ((1 - label_smoothing_epsilon) * e2) + (1.0 / e2.size(1))
        loss = loss_fun(pred_scores, e2_label)

        loss.backward()
        optimizer.step()
        return float(loss.data.cpu().numpy())


    for epoch in range(1):
        for raw_batch in dl:
            print(train_one_batch(model,optimizer,raw_batch))
