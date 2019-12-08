from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    entity_dim:int=200
    relation_dim:int=200
    emb_dropout_rate:float=0.3
    hidden_dropout_rate:float=0.3
    feat_dropout_rate:float=0.2
    emb_2D_d1:int=10
    emb_2D_d2:int=20
    num_out_channels:int=32
    kernel_size:int=3

class ConvE(nn.Module):
    def __init__(self, config:Config, num_entities, num_relations):
        super(ConvE, self).__init__()
        entity_dim = config.entity_dim
        emb_dropout_rate = config.emb_dropout_rate

        assert config.emb_2D_d1 * config.emb_2D_d2 == entity_dim
        assert config.emb_2D_d1 * config.emb_2D_d2 == config.relation_dim
        self.emb_2D_d1 = config.emb_2D_d1
        self.emb_2D_d2 = config.emb_2D_d2
        self.num_out_channels = config.num_out_channels
        self.w_d = config.kernel_size
        self.HiddenDropout = nn.Dropout(config.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(config.feat_dropout_rate)

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
        self.relation_embeddings = nn.Embedding(num_relations, config.relation_dim)
        self.RDropout = nn.Dropout(emb_dropout_rate)

        self.initialize_modules()

    def initialize_modules(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    def forward(self, e1, r):
        E1 = self.EDropout(self.entity_embeddings(e1)).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = self.RDropout(self.relation_embeddings(r)).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        all_embeddings = self.EDropout(self.entity_embeddings.weight)

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
        X = torch.mm(X, all_embeddings.transpose(1, 0))
        X += self.b.expand_as(X)

        S = F.sigmoid(X)
        return S