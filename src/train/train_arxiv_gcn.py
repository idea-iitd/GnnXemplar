from gnns import Arxiv_GCN, SAGE
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch
from copy import deepcopy
import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


# from logger import Logger  # Assumes you have a Logger implementation


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    out, _ = model(data.x, data.edge_index)

    # evaluator expects y_true and y_pred to be 2D tensors of shape (n, 1)
    y_pred = out.argmax(dim=-1, keepdims=True)  # (n, 1)
    y_true = data.y.reshape(-1, 1)  # (n, 1)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', transform=T.ToSparseTensor())
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    # Load dataset via our load_dataset function
    from load_dataset import load_dataset
    data, _, _, _ = load_dataset("arxiv")
    data = data.to(device)
    num_classes = int(data.y.max().item() + 1)

    if args.use_sage:
        model = SAGE(
            in_channels=data.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    else:
        model = Arxiv_GCN(
            in_channels=data.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=int(data.y.max().item() + 1),
            num_xlayers=args.num_layers,
            dropout=args.dropout
        ).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    best_valid_acc = 0
    best_weights = None

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(
                    f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}%, '
                    f'Test: {100 * test_acc:.2f}%'
                )
                if valid_acc >= best_valid_acc:
                    best_valid_acc = valid_acc
                    best_weights = deepcopy(model.state_dict())

    # save the best model
    torch.save(best_weights, "data/ogbn-arxiv/model.pt")
    print("Best model saved")
    print("Best model val accuracies: ", best_valid_acc)


if __name__ == "__main__":
    # todo: If the acc doesn't improve, try a different seed.
    main()
