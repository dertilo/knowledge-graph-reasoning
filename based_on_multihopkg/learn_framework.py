"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Base learning framework.
"""
import os
import random
import shutil
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from kgr.conv_e import ConvE
from knowledge_graph import KnowledgeGraph
from ops import var_cuda, zeros_var_cuda
import ops as ops
from kgr.ranking_metrics import hits_and_ranks
from pytorch_util import build_BatchingDataLoader


def convert_tuples_to_tensors(batch_data, num_labels=-1):
    def convert_to_binary(batch):
        z = zeros_var_cuda([len(batch), num_labels])
        for i in range(len(batch)):
            z[i][batch[i]] = 1
        return z

    batch_e1, batch_e2, batch_r = [list(l) for l in zip(*batch_data)]
    batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
    batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)

    if type(batch_e2[0]) is list:
        assert num_labels != -1
        batch_e2 = convert_to_binary(batch_e2)
    else:
        batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)

    return batch_e1, batch_e2, batch_r


class LFramework(nn.Module):
    def __init__(
        self,
        args,
        kg: KnowledgeGraph,
        agent: ConvE,
        secondary_kg=None,
        tertiary_kg=None,
    ):
        super(LFramework, self).__init__()
        self.args = args
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.kg = kg
        self.agent = agent

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate,
        )

        print("{} module created".format(self.model))

        self.num_negative_samples = args.num_negative_samples
        self.label_smoothing_epsilon = args.label_smoothing_epsilon
        self.loss_fun = nn.BCELoss()
        self.theta = args.theta
        self.secondary_kg = secondary_kg
        self.tertiary_kg = tertiary_kg

    def run_train(self, train_data, dev_data):

        for epoch_id in tqdm(range(self.start_epoch, self.num_epochs)):
            self.train()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []

            batches_g = (
                train_data[batch_id : batch_id + self.batch_size]
                for batch_id in range(0, len(train_data), self.batch_size)
            )

            for mini_batch in batches_g:
                loss = self.train_one_batch(mini_batch)
                batch_losses.append(loss)

            if epoch_id > 0 and epoch_id % self.num_peek_epochs == 0:
                self.eval()
                dev_scores = self.calc_scores(dev_data, self.dev_batch_size)
                print("Dev set performance: (include test set labels)")
                hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)

    def train_one_batch(self, mini_batch):
        self.optim.zero_grad()
        loss = self.calc_loss(mini_batch)
        loss.backward()
        if self.grad_norm > 0:
            clip_grad_norm_(self.parameters(), self.grad_norm)
        self.optim.step()
        return float(loss.data.cpu().numpy())

    def calc_scores(self, examples, batch_size):
        pred_scores = []
        for example_id in range(0, len(examples), batch_size):
            mini_batch = examples[example_id : example_id + batch_size]
            e1, e2, r = convert_tuples_to_tensors(mini_batch)
            pred_scores_batch = self.agent.forward(e1, r, self.kg)
            pred_scores.append(pred_scores_batch)
        scores = torch.cat(pred_scores)
        return scores

    def calc_loss(self, mini_batch):
        # compute object training loss
        e1, e2, r = convert_tuples_to_tensors(
            mini_batch, num_labels=self.kg.num_entities
        )
        e2_label = ((1 - self.label_smoothing_epsilon) * e2) + (1.0 / e2.size(1))
        pred_scores = self.agent.forward(e1, r, self.kg)
        return self.loss_fun(pred_scores, e2_label)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict["state_dict"] = self.state_dict()
        checkpoint_dict["epoch_id"] = epoch_id

        out_tar = os.path.join(
            self.model_dir, "checkpoint-{}.tar".format(checkpoint_id)
        )
        if is_best:
            best_path = os.path.join(self.model_dir, "model_best.tar")
            shutil.copyfile(out_tar, best_path)
            print("=> best model updated '{}'".format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            print("=> saving checkpoint to '{}'".format(out_tar))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, "vector.tsv")
        meta_data_path = os.path.join(self.model_dir, "metadata.tsv")
        v_o_f = open(vector_path, "w")
        m_o_f = open(meta_data_path, "w")
        for r in self.kg.relation2id:
            if r.endswith("_inv"):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ""
            for i in range(len(R)):
                r_print += "{}\t".format(float(R[i]))
            v_o_f.write("{}\n".format(r_print.strip()))
            m_o_f.write("{}\n".format(r))
            print(r, "{}".format(float(R.norm())))
        v_o_f.close()
        m_o_f.close()
        print("KG embeddings exported to {}".format(vector_path))
        print("KG meta data exported to {}".format(meta_data_path))


# if __name__ == '__main__':
#     dl = build_BatchingDataLoader(
#         get_batch_fun=lambda _: triples.next_batch(32, 10, device)
#     )
#
#     for epoch in range(3):
#         for batch in tqdm(dl):
#             pass
