from gnns import Citeseer_GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion, mask):
    model.eval()
    out, _ = model(data.x, data.edge_index)
    loss = F.nll_loss(out[mask], data.y[mask])
    preds = out.argmax(dim=-1)
    acc = (preds[mask] == data.y[mask]).sum().item() / mask.sum().item()
    return loss.item(), acc


def main():
    # Load Citeseer dataset with a simple normalization transform:
    dataset = Planetoid(root="data/Citeseer", name="Citeseer",
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    # Data from Planetoid already has train_mask, val_mask, and test_mask.
    print(
        f"Dataset: {dataset.name}, Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    model = Citeseer_GCN(in_channels=dataset.num_features,
                         # hidden_channels=64,
                         hidden_channels=16,
                         out_channels=dataset.num_classes,
                         # num_layers=3,
                         num_layers=1,
                         dropout=0.5)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs = 200

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, criterion)
        val_loss, val_acc = evaluate(model, data, criterion, data.val_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), "data/Citeseer/model.pt")
    _, test_acc = evaluate(model, data, criterion, data.test_mask)
    print(f"Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
