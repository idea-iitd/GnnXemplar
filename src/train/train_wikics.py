# File: train_wikics.py
from gnns import WikiCS_GCN  # Adjust if your file/module names differ
from torch_geometric.datasets import WikiCS
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)

    # Only compute loss on train_mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, data, criterion):
    model.eval()
    out, _ = model(data.x, data.edge_index)
    loss = criterion(out[data.val_mask], data.y[data.val_mask])

    # Accuracy
    preds = out.argmax(dim=-1)
    correct = (preds[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return float(loss), acc


def main():
    # 1) Load WikiCS dataset
    dataset = WikiCS(root="data/WikiCS")
    data = dataset[0]

    # We pick the 0th split for train/val
    data.train_mask = data.train_mask[:, 0]
    data.val_mask = data.val_mask[:, 0]
    data.test_mask = data.test_mask  # same mask for test across splits

    # 2) Initialize your GCN
    model = WikiCS_GCN(
        in_channels=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=64
    )

    # 3) Set up optimizer & loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 4) Training loop
    best_val_acc = 0.0
    best_state_dict = None
    epochs = 200
    for epoch in range(1, epochs+1):
        loss = train(model, data, optimizer, criterion)
        val_loss, val_acc = evaluate(model, data, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Train Loss: {loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 5) Save best model checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), "data/WikiCS/state_dict.pt")

    # 6) Evaluate on test set
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        preds = out.argmax(dim=-1)
        correct = (preds[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(correct) / int(data.test_mask.sum())
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy (final): {test_acc:.4f}")


if __name__ == "__main__":
    main()
