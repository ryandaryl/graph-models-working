import torch
from ogb.nodeproppred import DglNodePropPredDataset
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, ogbn_name, mask_rate=0.5, use_labels=False):
        super().__init__()
        self.mask_rate = mask_rate
        self.ogbn_name = ogbn_name
        self.use_labels = use_labels
        self.graph = None
        self.labels = None
        data = DglNodePropPredDataset(name=self.ogbn_name)
        split_idx = data.get_idx_split()
        self.train_idx = split_idx["train"]
        self.val_idx = split_idx["valid"]
        self.test_idx = split_idx["test"]
        self.n_classes = int(data.num_classes)

    def setup(self, stage):
        data = DglNodePropPredDataset(name=self.ogbn_name)
        graph, labels = data[0]
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        self.graph = graph
        self.labels = labels

    def train_dataloader(self):
        feat = self.graph.ndata["feat"]
        mask = torch.rand(self.train_idx.shape) < self.mask_rate
        train_labels_idx = self.train_idx[~mask]
        train_pred_idx = self.train_idx[mask]
        if self.use_labels:
            feat = add_labels(feat, self.labels, train_labels_idx, self.n_classes)
        return torch.utils.data.DataLoader(
            [(self.graph, feat, self.labels, train_pred_idx)],
            collate_fn=lambda data: data,
        )

    def val_dataloader(self):
        feat = self.graph.ndata["feat"]
        if self.use_labels:
            feat = add_labels(feat, self.labels, self.train_idx, self.n_classes)
        return torch.utils.data.DataLoader(
            [(self.graph, feat, self.labels, self.val_idx)],
            collate_fn=lambda data: data,
        )


def add_labels(feat, labels, idx, n_classes):
    onehot = torch.zeros([feat.shape[0], n_classes])
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)
