#!/usr/bin/env python
# coding: utf-8


import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.nn import LabelPropagation


root = osp.join('data', 'OGB')
dataset = PygNodePropPredDataset(
    'ogbn-arxiv', root, transform=T.Compose([
        T.ToUndirected(),
        T.ToSparseTensor(),
    ]))
split_idx = dataset.get_idx_split()
data = dataset[0]

model = LabelPropagation(num_layers=3, alpha=0.9)


out = model(data.y, data.adj_t, mask=split_idx['train'])
y_pred = out.argmax(dim=-1, keepdim=True)
for k in ['valid', 'test']:
    print(k, (data.y[split_idx[k]] == y_pred[split_idx[k]]).sum() /
          split_idx[k].shape[0])
