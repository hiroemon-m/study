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
    def __init__(self, edges, feats, model: Model, size: int,persona,persona_num):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size
        self.persona = persona
        self.persona_num = persona_num
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        return

    def optimize(self, t: int):
        feat = self.feats[t].to(device)
        edge = self.edges[t].to(device)
        self.optimizer.zero_grad()
        dot_product = torch.matmul(feat, torch.t(feat)).to(device)#内積

        #dot_product = dot_product.div((np.linalg.norm(feat))*(np.linalg.norm(feat.t())))
        #print("dot",dot_product.shape)
        #simはそれぞれのデータの和(32x32)->(32x1)
        sim = torch.mul(edge, dot_product).sum(1) #行列積

        #ハード
        #print(self.persona)
        persona_alpha = self.model.alpha[self.persona]
        #print("persona_alpha",self.model.alpha[self.persona])
        #print(persona_alpha.size())
        #print("persona_alpha",persona_alpha.size())
        #print(persona_alpha)
        #32x1
        #print(sim.size())
        sim = torch.dot(sim, persona_alpha)
        sim = torch.add(sim, 0.001)
        #print(sim.size())
        #print("sim",sim)

       
        persona_beta = self.model.beta[self.persona]


        #print("persona_beta",persona_beta.size())
        costs = torch.dot(edge.sum(1), persona_beta)
        costs = torch.add(costs, 0.001)
        #print("costs",costs.shape)
        reward = torch.sub(sim, costs)
        #ここから下に行追加
        #reward = torch.sum(reward,1)
        loss = -reward
        #ここまで
        #print("reward",reward.shape)
        #loss = -reward.sum()
        print("loss",loss)        

        loss.backward()
        del loss
        print("alpha",alpha)
        print("beta",beta)
        self.optimizer.step()


    def export_param(self):
        with open("/Users/matsumoto-hirotomo/study/data/DBLP/model.param.data.fast", "w") as f:
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

    data = init_real_data()
    #print(data.adj)
    #print(data.feature)
    data_size = len(data.adj[0])
    #ペルソナの設定[3,4,6,8,12]
    persona_num = 12
    data_persona = []
    path = "data/DBLP/data_norm{}.csv".format(int(persona_num))
    print(path)
    csvfile = open(path, 'r')
    gotdata = csv.reader(csvfile)
    for row in gotdata:
        data_persona.append(int(row[2]))
    csvfile.close()

    

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
    optimizer = Optimizer(data.adj, data.feature, model, data_size,data_persona,persona_num)
    for t in range(5):
        optimizer.optimize(t)
    optimizer.export_param()

 
