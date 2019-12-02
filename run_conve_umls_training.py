import sys

from learn_framework import LFramework

sys.path.append(".")
from dataclasses import dataclass

import os
import torch
import data_utils as data_utils
from knowledge_graph import KnowledgeGraph
from fact_network import ConvE
from emb import EmbeddingBasedMethod
from ops import to_cuda

torch.manual_seed(1)


@dataclass
class GeneralArgs:
    data_dir = "../MultiHopKG/data/umls"
    model_root_dir = "../MultiHopKG/model"
    model_dir = "../MultiHopKG/model"
    model = "conve"


@dataclass
class Args:
    model = "conve"
    group_examples_by_query = True
    entity_dim = 200
    relation_dim = 200
    history_dim = 400
    history_num_layers = 3
    add_reverse_relations = (True,)
    add_reversed_training_edges = True
    train_entire_graph = False
    emb_dropout_rate = 0.3
    num_epochs = 4
    num_wait_epochs = 500
    num_peek_epochs = 2
    start_epoch = 0
    batch_size = 512
    train_batch_size = 32
    dev_batch_size = 64
    margin = 0.05
    learning_rate = 0.003
    learning_rate_decay = 1.0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    grad_norm = 0.0
    xavier_initialization = True
    label_smoothing_epsilon = 0.1
    hidden_dropout_rate = 0.3
    feat_dropout_rate = 0.2
    emb_2D_d1 = 10
    emb_2D_d2 = 20
    num_out_channels = 32
    kernel_size = 3
    conve_state_dict_path = ""
    ff_dropout_rate = 0.1
    num_negative_samples = 20
    bandwidth = 400
    num_graph_convolution_layers = 0
    relation_only = False
    run_analysis = False
    theta = 0.2
    checkpoint_path = None


def initialize_model_directory(args):
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    model_dir = os.path.join(model_root_dir, "{}-{}".format(dataset, args.model))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def construct_model(args: Args, gargs: GeneralArgs):
    kg = KnowledgeGraph(args, gargs.data_dir)
    fn = ConvE(args, kg.num_entities)
    lf = EmbeddingBasedMethod(args, kg, fn)
    return lf


if __name__ == "__main__":

    args = Args()
    gargs = GeneralArgs()

    with torch.enable_grad():
        args.model_dir = initialize_model_directory(gargs)
        lf = construct_model(args, gargs)
        to_cuda(lf)

        train_path = data_utils.get_train_path(gargs)
        data_dir = gargs.data_dir
        dev_path = os.path.join(data_dir, "dev.triples")
        entity_index_path = os.path.join(data_dir, "entity2id.txt")
        relation_index_path = os.path.join(data_dir, "relation2id.txt")
        train_data = data_utils.load_triples(
            train_path,
            entity_index_path,
            relation_index_path,
            group_examples_by_query=args.group_examples_by_query,
            add_reverse_relations=args.add_reversed_training_edges,
        )

        seen_entities = set()
        dev_data = data_utils.load_triples(
            dev_path,
            entity_index_path,
            relation_index_path,
            seen_entities=seen_entities,
        )
        if args.checkpoint_path is not None:
            lf.load_checkpoint(args.checkpoint_path)
        lf.run_train(train_data, dev_data)


"""

Epoch 18: average training loss = 0.09945450909435749
=> saving checkpoint to './model/umls-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/checkpoint-18.tar'
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 1381.90it/s]
Dev set performance: (correct evaluation)
Hits@1 = 0.514
Hits@3 = 0.848
Hits@5 = 0.933
Hits@10 = 0.975
MRR = 0.693
Dev set performance: (include test set labels)
Hits@1 = 0.836
Hits@3 = 0.956
Hits@5 = 0.963
Hits@10 = 0.982
MRR = 0.900

"""
