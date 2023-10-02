#ソフト割り当て


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
    def __init__(self, edges, feats, model: Model, size: int, persona_ration):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size
        self.persona_ration = persona_ration
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        return

    def optimize(self, t: int):
        feat = self.feats[t].to(device)
        edge = self.edges[t].to(device)
        #print("feat",feat.shape)
        #print("edge",edge.shape)
        self.optimizer.zero_grad()
        #内積
        dot_product = torch.matmul(feat, torch.t(feat)).to(device)
        print("dot",dot_product.shape)
        #αの和　5x1
        #print(torch.tensor(self.model.alpha).view(5,1))
	#行列積
        #simはそれぞれのデータの和(32x32)->(32x1)
        sim = torch.mul(edge, dot_product).sum(1)
        #print(sim)
        #間違い　abx+cdy-> (a+c)(b+d)xとしてる
        #sim = torch.mul(sim, self.model.alpha.sum(1))
        #ペルソナとパラメターの積
        print(self.persona_ration)
        persona_alpha = torch.mm(self.persona_ration,self.model.alpha.view(self.size,1))
        #print("persona_alpha",persona_alpha.size)
        #print(persona_alpha)
        #32x1
        #print(sim.size())
        sim = torch.dot(sim, persona_alpha.view(32))
        sim = torch.add(sim, 0.001)
        #print("sim",sim)

        persona_beta = torch.mm(self.persona_ration,self.model.beta.view(self.size,1))
        #print("persona_beta",persona_beta)
        #print("edge",edge)
        costs = torch.dot(edge.sum(1), persona_beta.view(32))
        costs = torch.add(costs, 0.001)
        #print(costs)
        reward = torch.sub(sim, costs)
        #ここから下に行追加
        #reward = torch.sum(reward,1)
        loss = -reward
        #ここまで
        print("reward",reward.size())
        #loss = -reward.sum()
        print("loss",loss)        

        loss.backward()
        del loss
        self.optimizer.step()

    def export_param(self):
        with open("model.param.data.fast", "w") as f:
            max_alpha = 1.0
            max_beta = 1.0

            for i in range(self.size):
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
    data = init_real_data()
    path = "/Users/matsumoto-hirotomo/Downloads/netevolve-main/data/NIPS/gamma5.npy"
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
  
    #ペルソナ５個の場合のアルファベータ
    persona_num = 5

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
    persona_ration = torch.from_numpy(persona_ration).to(device)
    model = Model(alpha, beta)
    optimizer = Optimizer(data.adj, data.feature, model,persona_num,persona_ration)
    for t in range(5):
        optimizer.optimize(t)
    optimizer.export_param()