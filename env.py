# Standard Library
import math

# Third Party Library
import networkx as nx
import numpy as np
import scipy
import torch

# First Party Library
import config

device = config.select_device


class Env:
    def __init__(self, edges, feature, temper, alpha, beta,persona_ration) -> None:
        self.edges = edges
        self.feature = feature.to(device)
        self.temper = temper
        self.alpha = alpha
        self.beta = beta
        self.persona_ration = persona_ration
        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div_(norm)
        self.feature_t = self.feature.t()

        return

    def reset(self, edges, attributes):
        self.edges = edges
        self.feature = attributes
        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div(norm)
        self.feature_t = self.feature.t()

    #一つ進める
    def step(self, actions,persona):
        #actionsの確率に基づいて行動を決める
        #print(actions)
        next_mat = actions.bernoulli()   
        self.edges = next_mat

        dot_product = torch.mm(self.feature, self.feature_t).to(device)
        sim = torch.mul(self.edges,dot_product).sum(1)
        persona_alpha = torch.mm(self.persona_ration,self.alpha.view(self.persona_ration.size()[1],1))
        sim = torch.dot(sim, persona_alpha.view(32))
        sim = torch.add(sim, 0.001)

        persona_beta = torch.mm(self.persona_ration,self.beta.view(self.persona_ration.size()[1],1))
        costs = torch.dot(self.edges.sum(1), persona_beta.view(32))
        costs = torch.add(costs, 0.001)
        reward = torch.sub(sim, costs)
        
        return reward.sum()

    #特徴量の更新
    def update_attributes(self, attributes):
        self.feature = attributes
        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div(norm)

        self.feature_t = self.feature.t()

    #隣接行列を返す
    def state(self):
        #neighbor_mat = torch.mul(self.edges, self.edges)
        neighbor_mat = torch.mul(self.edges, self.edges)
        return neighbor_mat, self.feature
