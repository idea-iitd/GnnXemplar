from gnns import GCN
from load_dataset import load_dataset
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


def ensure_1d_mask(mask):
    if mask.dim() > 1:
        return mask[:, 0]
    return mask


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    data.train_mask = ensure_1d_mask(data.train_mask)
    target = data.y if data.y.dim() == 1 else data.y.squeeze(1)
    loss = F.nll_loss(out[data.train_mask], target[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion, mask):
    model.eval()
    out, _ = model(data.x, data.edge_index)
    mask = ensure_1d_mask(mask)
    target = data.y if data.y.dim() == 1 else data.y.squeeze(1)
    loss = F.nll_loss(out[mask], target[mask])
    preds = out.argmax(dim=-1)
    acc = (preds[mask] == target[mask]).sum().item() / mask.sum().item()
    return loss.item(), acc


def main():
    parser = argparse.ArgumentParser(
        description='Amazon-Ratings Node Classification')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.005,
                        help="Learning rate, choose from {0.001, 0.005, 0.01}")
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help="Hidden dimension, choose from {64, 256, 512}")
    parser.add_argument('--num_layers', type=int, default=3,
                        help="Number of GNN layers (1 to 10)")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help="Dropout rate, choose from {0.2, 0.3, 0.5, 0.7}")
    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # Load the Amazon-Ratings dataset via our load_dataset branch.
    data, _, _, _ = load_dataset("amazonratings")
    data = data.to(device)

    num_nodes = data.num_nodes
    # If masks are not provided, create random splits (e.g., 60/20/20)
    if not hasattr(data, 'train_mask'):
        perm = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
        data.train_mask[perm[:train_size]] = True
        data.val_mask[perm[train_size:train_size+val_size]] = True
        data.test_mask[perm[train_size+val_size:]] = True

    data.y = data.y.squeeze() if data.y.dim() > 1 else data.y

    model = GCN(data.num_features, args.hidden_channels,
                int(data.y.max().item() + 1),
                num_layers=args.num_layers,
                dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs = args.epochs

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, criterion)
        val_loss, val_acc = evaluate(model, data, criterion, data.val_mask)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: Loss {loss:.4f}, Val Loss {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), "data/amazonratings/model.pt")
    test_loss, test_acc = evaluate(model, data, criterion, data.test_mask)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
