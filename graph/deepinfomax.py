import os
import os.path as osp
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
import torch_geometric.transforms as T 
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self,  data):
        deg = degree(data.edge_index[0], dtype = torch.float)
        deg = (deg - self.mean ) / self.std
        data.x = deg.view (-1 , 1)
        return data

 
class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers

    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs, beta):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'

    #set measure='JSD_hard' to use hard negative sampling
    #and measure='JSD' otherwise
    measure='JSD_hard'

    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure, beta)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR

if __name__ == '__main__':
    
    args = arg_parse()
    epochs = 200
    log_interval = 100
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    repeats = args.repeats
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)

    #loop over several values of beta
    betas=[0.0, 1.0, 2.0, 10.0]

    results={beta: [0]*repeats for beta in betas}

    for r in range(repeats):
        for beta in betas:
            accuracies = []
            dataset = TUDataset(path, name=DS).shuffle()

            if dataset.data.x is None:
            	max_degree = 0
            	degs = []
            	for data in dataset:
            		degs += [degree(data.edge_index[0], dtype=torch.long)]
            		max_degree = max(max_degree, degs[-1].max().item())
            	if max_degree < 1000:
            		dataset.transform = T.OneHotDegree(max_degree)
            	else:
            		deg = torch.cat(degs, dim=0).to(torch.float)
            		mean, std = deg.mean().item(), deg.std().item()
            		dataset.transform = NormalizedDegree(mean, std)
            try:
                dataset_num_features = dataset.num_features
            except:
                dataset_num_features = 1

            dataloader = DataLoader(dataset, batch_size=batch_size)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = GcnInfomax(args.hidden_dim, args.num_gc_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            print('================')
            print('lr: {}'.format(lr))
            print('num_features: {}'.format(dataset_num_features))
            print('hidden_dim: {}'.format(args.hidden_dim))
            print('num_gc_layers: {}'.format(args.num_gc_layers))
            print('================')
            
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader)
            res = evaluate_embedding(emb, y)
            accuracies.append(res)

            for epoch in range(1, epochs+1):
                loss_all = 0
                model.train()
                for data in dataloader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    loss = model(data.x, data.edge_index, data.batch, data.num_graphs, beta)
                    loss_all += loss.item() * data.num_graphs
                    loss.backward()
                    optimizer.step()
                print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

                if epoch % log_interval == 0:
                    model.eval()
                    emb, y = model.encoder.get_embeddings(dataloader)
                    res = evaluate_embedding(emb, y)
                    accuracies.append(res)                 

            with open('unsupervised.log', 'a+') as f:
                s = json.dumps(accuracies)
                f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))

            results[beta][r]=res 

            if not osp.exists('../results'):
                os.mkdir('../results')
            if not osp.exists(f'../results/{DS}'):
                os.mkdir(f'../results/{DS}')
            df = pd.DataFrame(data=results) 
            df.to_csv(f"../results/{DS}/results.csv", sep=',',index=False)


