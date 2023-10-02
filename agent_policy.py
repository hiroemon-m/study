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
    def __init__(self, T, e, r, w, persona) -> None:
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
        
        self.persona = persona

    def forward(
        self, attributes, edges, N,persona
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        edges = (edges > 0).float().to(device)

        #隣接ノードと自分の特徴量を集約する
        #print(edges.size()) 32x32
        #print(attributes.size())32x2411
        tmp_tensor = self.W[self.persona] * torch.matmul(edges, attributes)
        #tmp_tensor = torch.matmul(edges, attributes)
        
        #feat =  0.5*attributes + 0.5*tmp_tensor
        r = self.r[self.persona]

        r = r + 1e-8
        feat = r * attributes + tmp_tensor * (1 - r)
        #print("feat",feat)
        feat_prob = torch.tanh(feat)

        #x = x.div((np.linalg.norm(feat.detach().clone()))*(np.linalg.norm(feat.detach().clone().t())))

        # Compute similarity
        x = torch.mm(feat, feat.t())
        #print(x)
        x = x.div(self.T[self.persona]).exp().mul(self.e[self.persona])

        min_values = torch.min(x, dim=0).values
        # # 各列の最大値 (dim=0 は列方向)
        max_values = torch.max(x, dim=0).values
        # Min-Max スケーリング
        x = (x - min_values) / ((max_values - min_values) + 1e-4)+1e-4
        #print(x[0])

        x = torch.tanh(x)

        #x = torch.sigmoid(x)
        #print(x[0])
        #new_x = x.clone()
        #new_x[x<0]=0
        #print(x)
        #y = torch.tanh((1-sim).div(self.T[self.persona]).exp().mul(self.e[self.persona]))
        #print("prob", x)
        #new_x = x.clone()
        #new_x[x<y] = 0
        #print("prob", new_x)


        return x, feat, feat_prob

        # エッジの存在関数
        # x = torch.sigmoid(x)
        # x = torch.tanh(x)
        # x = torch.relu(x)
        # print("prob", x)
        # return x, feat



    def predict(
        self, attributes, edges, N,persona
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(attributes, edges, N,persona)
