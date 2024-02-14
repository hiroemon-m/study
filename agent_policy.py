# Standard Library
from typing import Tuple

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party Library
import config
import csv
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
device = config.select_device

class AgentPolicy(nn.Module):
    def __init__(self, T, e, r, w) -> None:
        super().__init__()
        self.T = nn.Parameter(
            torch.tensor(T).float().to(device), requires_grad=True
        )
        self.e = nn.Parameter(
            torch.tensor(e).float().to(device), requires_grad=True
        )
        self.r = nn.Parameter(
            torch.tensor(r).float().view(-1, 1).to(device), requires_grad=True
        )
        self.W = nn.Parameter(
            torch.tensor(w).float().view(-1, 1).to(device), requires_grad=True
        )
        

    def forward(
        self, attributes, edges, N
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        edges = (edges > 0).float().to(device)

        #隣接ノードと自分の特徴量を集約する
        #print(edges.size()) 32x32
        #print(attributes.size())32x2411
        tmp_tensor = self.W * torch.matmul(edges, attributes)
        r = self.r

        r = r + 1e-8
        feat = r * attributes + tmp_tensor * (1 - r)
        feat_prob = feat
        # Compute similarity
        x = torch.mm(feat, feat.t())
        #print(x)
        x = x.div(self.T).exp().mul(self.e)

        x = torch.tanh(x)

        return x, feat, feat_prob




    def predict(
        self, attributes, edges, N
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(attributes, edges, N)
