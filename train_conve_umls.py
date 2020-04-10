from typing import List

import numpy
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from util import data_io

from kgr.conv_e import Config, ConvE
from kgr.ranking_metrics import hits_and_ranks
from pytorch_util import build_BatchingDataLoader
from triple_data import (
    Branch,
    build_trees_dataset,
    build_resetting_next_fun,
    build_batch_iter,
)


def convert_branches_to_tensors(batch: List[Branch], num_entities):
    def convert_to_binary(batch):
        z = torch.zeros((len(batch), num_entities))
        for i, objs_idx in enumerate(batch):
            z[i][list(objs_idx)] = 1
        return z

    batch_e1, batch_r, batch_e2 = [list(l) for l in zip(*batch)]
    batch_e1 = torch.LongTensor(batch_e1)
    batch_r = torch.LongTensor(batch_r)
    batch_e2 = convert_to_binary(batch_e2)

    return batch_e1, batch_r, batch_e2


def convert_tuples_to_tensors(batch: List[Branch]):
    batch_e1, batch_r, _ = [list(l) for l in zip(*batch)]
    batch_e1 = torch.LongTensor(batch_e1)
    batch_r = torch.LongTensor(batch_r)
    return batch_e1, batch_r


if __name__ == "__main__":

    build_path = lambda ds: "../MultiHopKG/datasets/umls/%s.triples" % ds
    triple_files = {ds: build_path(ds) for ds in ["train", "dev", "test"]}
    data = build_trees_dataset(triple_files)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_entities = len(data.ent2id.keys())
    train_loader = build_BatchingDataLoader(
        get_batch_fun=build_resetting_next_fun(
            lambda: build_batch_iter(data.dataset2trees["train"], 32)
        )
    )
    eval_loader = build_BatchingDataLoader(
        get_batch_fun=build_resetting_next_fun(
            lambda: build_batch_iter(data.dataset2trees["dev"], 128)
        )
    )

    num_relations = len(data.rel2id.keys())
    config = Config()
    model: ConvE = ConvE(config, num_entities, num_relations)

    label_smoothing_epsilon = 0.1
    loss_fun = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.003,
    )

    def train_one_batch(model: nn.Module, optimizer, raw_batch):
        optimizer.zero_grad()

        e1, r, e2 = convert_branches_to_tensors(raw_batch, num_entities)
        pred_scores = model.forward(e1.to(device), r.to(device))
        e2_label = ((1 - label_smoothing_epsilon) * e2) + (1.0 / e2.size(1))
        loss = loss_fun(pred_scores, e2_label.to(device))

        loss.backward()
        optimizer.step()
        return float(loss.data.cpu().numpy())


    def run_evaluation(eval_loader, model):
        model.eval()
        pred_scores = []
        dev_data = []
        for mini_batch in eval_loader:
            dev_data.extend(mini_batch)
            e1, r = convert_tuples_to_tensors(mini_batch)
            scores = model.forward(e1.to(device), r.to(device)).cpu()
            pred_scores.append(scores)
        dev_scores = torch.cat(pred_scores)
        return hits_and_ranks(dev_data, dev_scores, data.dataset2trees)


    pbar = tqdm(range(100))
    model.to(device)
    for epoch in pbar:
        model.train()
        epoch_loss = numpy.mean(
            [train_one_batch(model, optimizer, raw_batch) for raw_batch in train_loader]
        )
        if epoch%10==0:
            mrr = run_evaluation(eval_loader, model)["mrr"]
            named_params = {n: v for n, v in model.named_parameters()}
            data_io.write_json('ent2id.json',data.ent2id)
            torch.save(named_params['entity_embeddings.weight'].data,"entity_embeddings.pt")
        pbar.set_description("Epoch: {}; mean-loss: {:.4f}; MRR: {:.3f}".format(epoch + 1,epoch_loss, mrr))


'''
Epoch: 100; mean-loss: 0.0891; MRR: 0.947: 100%|██████████| 100/100 [02:10<00:00,  1.30s/it]
'''