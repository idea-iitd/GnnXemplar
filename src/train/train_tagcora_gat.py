from load_dataset import load_dataset
from utils import seed_everything
from gnns import CORA_GAT
import warnings
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import torch_geometric as pyg
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from types import SimpleNamespace
import pdb
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

"""Get concept vectors for the CORA dataset."""

# * >>> Imports and seeds


warnings.filterwarnings("ignore")


sns.set()
sns.set_style("white")
seed = 0
seed_everything(seed)

# * >>> Data
dataset = None
device = 'cpu'
# dataset_name = 'ogbn-products'
dataset_name = 'TAGCora'  # 'TAGCora', 'CORA', 'bashapes', 'ogbn-proteins'
print(f'Reading {dataset_name} dataset')

if dataset_name == 'CORA':
    dataset = pyg.datasets.Planetoid(
        root="data", name="CORA", split="public", force_reload=True,)
    data = dataset[0]
elif dataset_name == 'TAGCora':
    dataset, data = load_dataset(
        dataset_name=dataset_name, trained_model=False)

# data = dataset[0]

# * >>> Training loop
model = CORA_GAT(in_channels=dataset.num_features,
                 num_classes=dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=model.lr, weight_decay=model.weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()


def train():
    mask = data.train_mask
    # put the model in training mode
    model.train()
    # zero the gradients
    optimizer.zero_grad()
    # call forward
    # print(f"{data.y.shape = }")
    out, __ = model(data.x, data.edge_index)
    # compute the loss
    if len(data.y.shape) == 2:
        #     out = out.argmax(dim = 1)
        loss = loss_fn(out[mask], data.y[mask].squeeze())
    else:
        loss = loss_fn(out[mask], data.y[mask])

    # gradient descent
    loss.backward()
    # optimizer step
    optimizer.step()
    return loss.item()


def test(mask):
    model.eval()
    with torch.no_grad():
        out, __ = model(data.x, data.edge_index)
        # if len(data.y.shape) == 2:
        #     loss = loss_fn(out[mask], data.y[mask].squeeze())
        # else:
        #     loss = loss_fn(out[mask], data.y[mask])
        loss = loss_fn(out[mask], data.y[mask])
        acc = 1.0*(out[mask].argmax(dim=-1) == data.y[mask]).sum() / mask.sum()
    return round(loss.item(), 4), round(acc.item(), 4)


def plot_curves(losses_train, losses_val, accs_train, accs_val):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.lineplot(ax=axes[0], x=range(len(losses_train)),
                 y=losses_train, color="C0")
    sns.lineplot(ax=axes[0], x=range(len(losses_val)),
                 y=losses_val, color="C1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    sns.lineplot(ax=axes[1], x=range(len(accs_train)),
                 y=accs_train, color="C0", label="Train")
    sns.lineplot(ax=axes[1], x=range(len(accs_val)),
                 y=accs_val, color="C1", label="Valid")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    axes[1].legend()
    fig.savefig(f"data/{dataset_name}/training.png",
                dpi=150, bbox_inches="tight")


loss_val_best = float("inf")
acc_val_best = 0.0
epoch_best = 0
weights_best = None
patience = 5
# warmup = 50
max_epochs = 100
trunc = 4
losses_train = []
losses_val = []
accs_train = []
accs_val = []
for epoch in range(max_epochs):
    train()
    loss_train, acc_train = test(data.train_mask)
    loss_val, acc_val = test(data.val_mask)

    losses_train.append(loss_train)
    losses_val.append(loss_val)
    accs_train.append(acc_train)
    accs_val.append(acc_val)

    if epoch % 1 == 0:
        print(f"Epoch {epoch:03d} | Loss train: {loss_train:.4f}, Loss val: {loss_val:.4f}"
              f" | Acc train: {acc_train:.4f}, Acc val: {acc_val:.4f}")
        plot_curves(losses_train, losses_val, accs_train, accs_val)

    # store best results
    if (loss_val <= loss_val_best) or (acc_val >= acc_val_best):
        epoch_best = epoch
        loss_val_best = loss_val
        acc_val_best = acc_val
        weights_best = deepcopy(model.state_dict())
        torch.save(weights_best, f"data/{dataset_name}/state_dict.pt")

        # check stopping criteria
    if (epoch - epoch_best) > patience:
        print(f"\nStopping early at {epoch}")
        break

# * >>> Save to disk
print(
    f"Best epoch: {epoch_best} | Loss val: {loss_val_best:.4f} | Acc val: {acc_val_best:.4f}")
torch.save(weights_best, f"data/{dataset_name}/state_dict.pt")

model.load_state_dict(torch.load(f"data/{dataset_name}/state_dict.pt"))
model.eval()
loss_test, acc_test = test(data.test_mask)
print(f"GAT test accuracy: {acc_test:.4f}")

with torch.no_grad():
    out, __ = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    pred = pred.numpy().astype(int)
np.save(f"data/{dataset_name}/gnn_pred.npy", pred)
