####ハード割り当て


# Standard Library
import random
from enum import IntEnum

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# First Party Library
import config
from init_real_data import init_real_data
import csv

device = config.select_device


class Interest(IntEnum):
    RED = 2
    BLUE = 1


class Model(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()

        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        return


class Optimizer:
    def __init__(self, edges, feats, model: Model, size: int,persona_ration,persona_num):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size
        self.persona_ration = persona_ration
        self.persona_num = persona_num
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        return

    def optimize(self, t: int):
        feat = self.feats[t].to(device)
        edge = self.edges[t].to(device)
        self.optimizer.zero_grad()
        dot_product = torch.matmul(feat, torch.t(feat)).to(device)#内積

        #alpha
        sim = torch.mul(edge, dot_product).sum(1) #行列積
        persona_alpha = torch.mm(self.persona_ration,self.model.alpha.view(persona_num,1))
        sim = torch.dot(sim, persona_alpha.view(32))
        sim = torch.add(sim, 0.001)

        #beta
        persona_beta = torch.mm(self.persona_ration,self.model.beta.view(persona_num,1))
        costs = torch.dot(edge.sum(1), persona_beta.view(32))
        costs = torch.add(costs, 0.001)

        reward = torch.sub(sim, costs)
        loss = -reward
        print("loss",loss)        
        loss.backward()
        del loss
        self.optimizer.step()

      


    def export_param(self):
        with open("model.param.data.fast", "w") as f:
            max_alpha = 1.0
            max_beta = 1.0
            #ペルソナの数
            for i in range(self.persona_num):
                f.write(
                    "{},{}\n".format(
                        self.model.alpha[i].item() / max_alpha,
                        self.model.beta[i].item() / max_beta,
                    )
                )


if __name__ == "__main__":
    # data = attr_graph_dynamic_spmat_NIPS(T=10)
    # data = attr_graph_dynamic_spmat_DBLP(T=10)
    # data = TwitterData(T=10)
    # data = attr_graph_dynamic_spmat_twitter(T=10)

    #データのロード
    data = init_real_data()
    #データのサイズ
    data_size = len(data.adj[0])


    #ペルソナの設定
    #ペルソナの数[3,4,5,6,8]
    persona_num = 8
    path = "data/NIPS/gamma{}.npy".format(int(persona_num))
    print(path)
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
    persona_ration = torch.from_numpy(persona_ration).to(device)

    alpha = torch.from_numpy(
        np.array(
            [0.2 for i in range(persona_num)],
            dtype=np.float32,
        ),
    ).to(device)

    beta = torch.from_numpy(
        np.array(
            [0.2 for i in range(persona_num)],
            dtype=np.float32,
        ),
    ).to(device)

    model = Model(alpha, beta)
    optimizer = Optimizer(data.adj, data.feature, model, data_size,persona_ration,persona_num)
    for t in range(5):
        optimizer.optimize(t)
    optimizer.export_param()

 
