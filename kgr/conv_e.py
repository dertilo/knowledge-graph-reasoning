import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge_graph import KnowledgeGraph


class ConvE(nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        entity_dim = args.entity_dim
        emb_dropout_rate = args.emb_dropout_rate

        self.relation_dim = args.relation_dim
        assert args.emb_2D_d1 * args.emb_2D_d2 == entity_dim
        assert args.emb_2D_d1 * args.emb_2D_d2 == args.relation_dim
        self.emb_2D_d1 = args.emb_2D_d1
        self.emb_2D_d2 = args.emb_2D_d2
        self.num_out_channels = args.num_out_channels
        self.w_d = args.kernel_size
        self.HiddenDropout = nn.Dropout(args.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(args.feat_dropout_rate)

        # stride = 1, padding = 0, dilation = 1, groups = 1
        self.conv1 = nn.Conv2d(1, self.num_out_channels, (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(entity_dim)
        self.register_parameter("b", nn.Parameter(torch.zeros(num_entities)))
        h_out = 2 * self.emb_2D_d1 - self.w_d + 1
        w_out = self.emb_2D_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        self.fc = nn.Linear(self.feat_dim, entity_dim)

        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.EDropout = nn.Dropout(emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(emb_dropout_rate)

        self.initialize_modules()

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def initialize_modules(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    def forward(self, e1, r, kg: KnowledgeGraph):
        E1 = self.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = self.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = self.get_all_entity_embeddings()

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.mm(X, E2.transpose(1, 0))
        X += self.b.expand_as(X)

        S = F.sigmoid(X)
        return S