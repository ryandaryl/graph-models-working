import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from models import GCN, GAT
from data import DataModule
import warnings

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore")


def compute_acc(pred, labels, evaluator):
    return evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]


def compute_accuracy_train_val_test(pred, labels, idx, split_idx):
    return {
        f"{stage}_acc": compute_acc(
            pred[split_idx[stage]],
            labels[split_idx[stage]],
            evaluator=Evaluator(name="ogbn-arxiv"),
        )
        for stage in ["train", "valid", "test"]
    }


from plotly import graph_objects as go
from plotly import express as px
import numpy as np

fig = go.FigureWidget(px.line(y=[0], x=[0], range_x=[0, 100]))


def thing(trainer, _):
    fig.data[0].y = np.append(
        fig.data[0].y, trainer.callback_metrics["train_acc"].cpu()
    )
    fig.data[0].x = np.append(fig.data[0].x, trainer.current_epoch)


loss_callback = pl.callbacks.LambdaCallback(on_train_epoch_end=thing)
fig


data = DglNodePropPredDataset("ogbn-arxiv")
graph, labels = data[0]
split_idx = data.get_idx_split()
n_classes = data.num_classes
datamodule = DataModule(
    use_labels=True,
    split_idx=split_idx,
    labels=labels,
    graph=graph,
    n_classes=n_classes,
)
accuracy = lambda pred, labels, idx: compute_accuracy_train_val_test(
    pred, labels, idx, split_idx
)

model = GCN(
    in_feats=graph.ndata["feat"].shape[1] + n_classes,
    n_classes=n_classes,
    n_hidden=256,
    n_layers=3,
    activation=F.relu,
    dropout=0.75,
    use_linear=False,
    val_metric=accuracy,
)

trainer = pl.Trainer(accelerator="auto", max_epochs=100, callbacks=[loss_callback])
trainer.fit(model, datamodule=datamodule)


n_epochs = 100
n_layers = 3
n_hidden = 256
n_heads = 3
dropout = 0.75
attn_drop = 0.05
norm = "none"  # "both"

model = GAT(
    graph.ndata["feat"].shape[1] + n_classes,
    n_classes,
    n_hidden=n_hidden,
    n_layers=n_layers,
    n_heads=n_heads,
    activation=F.relu,
    dropout=dropout,
    attn_drop=attn_drop,
    norm=norm,
    val_metric=accuracy,
)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=100,
    log_every_n_steps=1,
)
trainer.fit(model, datamodule=datamodule)
