import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from models import GCN, GAT
from data import DataModule


def compute_acc(pred, labels, evaluator):
    return evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]


n_epochs = 100
n_layers = 3
n_hidden = 256
n_heads = 3
dropout = 0.75
attn_drop = 0.05
norm = "none"  # "both"

datamodule = DataModule("ogbn-arxiv", use_labels=True)
data = DglNodePropPredDataset("ogbn-arxiv")
graph, labels = data[0]
in_feats = graph.ndata["feat"].shape[1]
n_classes = (labels.max() + 1).item()

evaluator = Evaluator(name="ogbn-arxiv")
accuracy = lambda pred, labels, idx: {
    f"{stage}_acc": compute_acc(
        pred[getattr(datamodule, f"{stage}_idx")],
        labels[getattr(datamodule, f"{stage}_idx")],
        evaluator,
    )
    for stage in ["train", "val", "test"]
}

model = GCN(
    in_feats=in_feats + n_classes,
    n_classes=n_classes,
    n_hidden=n_hidden,
    n_layers=n_layers,
    activation=F.relu,
    dropout=dropout,
    use_linear=False,
    val_metric=accuracy,
)
"""model = GAT(
    in_feats + n_classes,
    n_classes,
    n_hidden=n_hidden,
    n_layers=n_layers,
    n_heads=n_heads,
    activation=F.relu,
    dropout=dropout,
    attn_drop=attn_drop,
    norm=norm,
    val_metric=accuracy,
)"""
print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
print(sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=n_epochs,
    log_every_n_steps=1,
)
trainer.fit(model, datamodule=datamodule)
