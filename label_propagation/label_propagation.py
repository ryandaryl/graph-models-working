#!/usr/bin/env python
# coding: utf-8


import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.nn import LabelPropagation
from torch_sparse import SparseTensor
import torch


model = LabelPropagation(num_layers=3, alpha=0.9)


root = osp.join('data', 'OGB')
dataset = PygNodePropPredDataset(
    'ogbn-arxiv', root, transform=T.Compose([
        T.ToUndirected(),
        T.ToSparseTensor(),
    ]))
split_idx = dataset.get_idx_split()
data = dataset[0]
y = data.y
adj_t = data.adj_t


n_nodes = 1200
edge_probability = 0.5
split_idx = {k: v for k, v in zip(['train', 'valid', 'test'], torch.arange(
    n_nodes).reshape(n_nodes // 3, 3).permute(1, 0))}
adjacency_dense = torch.rand([n_nodes] * 2) > edge_probability
adjacency_dense[:n_nodes // 2, n_nodes // 2:] = 0
adjacency_dense[n_nodes // 2:, :n_nodes // 2] = 0
y = torch.zeros([n_nodes, 1], dtype=torch.long)
y[n_nodes // 2:] = 1
row, col = adjacency_dense.nonzero().permute(1, 0)
adj_t = SparseTensor(row=row, col=col)


out = model(y, adj_t, mask=split_idx['train'])
y_pred = out.argmax(dim=-1, keepdim=True)
for k in ['valid', 'test']:
    print(k, (y[split_idx[k]] == y_pred[split_idx[k]]).sum() /
          split_idx[k].shape[0])
