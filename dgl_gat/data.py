import torch
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        graph,
        labels,
        split_idx=None,
        n_classes=None,
        mask_rate=0.5,
        use_labels=False,
    ):
        super().__init__()
        self.graph = graph
        self.labels = labels
        self.train_idx = split_idx["train"]
        self.valid_idx = split_idx["valid"]
        self.test_idx = split_idx["test"]
        self.n_classes = n_classes
        self.mask_rate = mask_rate
        self.use_labels = use_labels

    def setup(self, stage):
        srcs, dsts = self.graph.all_edges()
        self.graph.add_edges(dsts, srcs)
        self.graph = self.graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {self.graph.number_of_edges()}")

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
            [(self.graph, feat, self.labels, self.valid_idx)],
            collate_fn=lambda data: data,
        )


def add_labels(feat, labels, idx, n_classes):
    onehot = torch.zeros([feat.shape[0], n_classes])
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)
