"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Embedding-based knowledge base completion baselines.
"""

import os
from tqdm import tqdm

import torch
import torch.nn as nn

from learn_framework import LFramework
from data_utils import NO_OP_ENTITY_ID, DUMMY_ENTITY_ID
from ops import var_cuda, int_var_cuda, int_fill_var_cuda


class EmbeddingBasedMethod(LFramework):
    def __init__(self, args, kg, agent, secondary_kg=None, tertiary_kg=None):
        super(EmbeddingBasedMethod, self).__init__(args, kg, agent)
        self.num_negative_samples = args.num_negative_samples
        self.label_smoothing_epsilon = args.label_smoothing_epsilon
        self.loss_fun = nn.BCELoss()

        self.theta = args.theta
        self.secondary_kg = secondary_kg
        self.tertiary_kg = tertiary_kg

    def forward_fact(self, examples):
        kg, mdl = self.kg, self.agent
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id : example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            e1, e2, r = self.convert_tuples_to_tensors(mini_batch)
            pred_score = mdl.forward_fact(e1, r, e2, kg)
            pred_scores.append(pred_score[:mini_batch_size])
        return torch.cat(pred_scores)

    def loss(self, mini_batch):
        kg, mdl = self.kg, self.agent
        # compute object training loss
        e1, e2, r = self.convert_tuples_to_tensors(
            mini_batch, num_labels=kg.num_entities
        )
        e2_label = ((1 - self.label_smoothing_epsilon) * e2) + (1.0 / e2.size(1))
        pred_scores = mdl.forward(e1, r, kg)
        loss = self.loss_fun(pred_scores, e2_label)
        loss_dict = {}
        loss_dict["model_loss"] = loss
        loss_dict["print_loss"] = float(loss)
        return loss_dict

    def predict(self, mini_batch, verbose=False):
        kg, mdl = self.kg, self.agent
        e1, e2, r = self.convert_tuples_to_tensors(mini_batch)
        if self.model == "hypere":
            pred_scores = mdl.forward(e1, r, kg, [self.secondary_kg])
        elif self.model == "triplee":
            pred_scores = mdl.forward(e1, r, kg, [self.secondary_kg, self.tertiary_kg])
        else:
            pred_scores = mdl.forward(e1, r, kg)
        return pred_scores

    def get_subject_mask(self, e1_space, e2, q):
        assert False  # TODO(tilo) !?
        kg = self.kg
        if kg.args.mask_test_false_negatives:
            answer_vectors = kg.all_subject_vectors
        else:
            answer_vectors = kg.train_subject_vectors
        subject_masks = []
        for i in range(len(e1_space)):
            _e2, _q = int(e2[i]), int(q[i])
            if not _e2 in answer_vectors or not _q in answer_vectors[_e2]:
                answer_vector = var_cuda(torch.LongTensor([[kg.num_entities]]))
            else:
                answer_vector = answer_vectors[_e2][_q]
            subject_mask = torch.sum(e1_space[i].unsqueeze(0) == answer_vector, dim=0)
            subject_masks.append(subject_mask)
        subject_mask = torch.cat(subject_masks).view(len(e1_space), -1)
        return subject_mask

    def get_object_mask(self, e2_space, e1, q):
        kg = self.kg
        if kg.args.mask_test_false_negatives:
            answer_vectors = kg.all_object_vectors
        else:
            answer_vectors = kg.train_object_vectors
        object_masks = []
        for i in range(len(e2_space)):
            _e1, _q = int(e1[i]), int(q[i])
            if not e1 in answer_vectors or not q in answer_vectors[_e1]:
                answer_vector = var_cuda(torch.LongTensor([[kg.num_entities]]))
            else:
                answer_vector = answer_vectors[_e1][_q]
            object_mask = torch.sum(e2_space[i].unsqueeze(0) == answer_vector, dim=0)
            object_masks.append(object_mask)
        object_mask = torch.cat(object_masks).view(len(e2_space), -1)
        return object_mask

    def export_reward_shaping_parameters(self):
        """
        Export knowledge graph embeddings and fact network parameters for reward shaping models.
        """
        fn_state_dict_path = os.path.join(self.model_dir, "fn_state_dict")
        fn_kg_state_dict_path = os.path.join(self.model_dir, "fn_kg_state_dict")
        torch.save(self.agent.state_dict(), fn_state_dict_path)
        print("Fact network parameters export to {}".format(fn_state_dict_path))
        torch.save(self.kg.state_dict(), fn_kg_state_dict_path)
        print("Knowledge graph embeddings export to {}".format(fn_kg_state_dict_path))
