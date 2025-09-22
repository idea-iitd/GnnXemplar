"""Get concept vectors for the CORA dataset using random concept selection for ablation study."""

# * >>> Imports and seeds
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch

from load_dataset import load_dataset
from utils import seed_everything
from sklearn.metrics.pairwise import cosine_similarity
import gnns
from torch_geometric import datasets
import pickle

# * >>> Compute reverse KNN (this function is still used to compute knn for the concept vectors)


def find_reverse_knn(distance_mat, k, farthest=False):
    """This function takes a distance matrix as input and computes the k-nearest neighbors (KNN) and
    reverse k-nearest neighbors (RKNN) for each node.

    Args:
        distance_mat (np.ndarray): Distance matrix containing pairwise distances between the nodes.
        k (int): The "k" in reverse k-nearest neighbors

    Returns:
        dict: A dictionary where keys are nodes and values are sets of nodes that have the
        given node as one of their k-nearest neighbors.
        list[list]: Indices of the k-nearest neighbors of each node.
    """
    num_nodes = distance_mat.shape[0]
    knn = []
    for node in range(num_nodes):
        if farthest:
            topk = np.argpartition(distance_mat[node], k)[:k].tolist()
        else:
            if k > num_nodes:
                topk = np.argsort(distance_mat[node]).tolist()
            else:
                topk = np.argpartition(distance_mat[node], -k)[-k:].tolist()
        knn.append(topk)
    rknn = defaultdict(set)
    for node, knn_nodes in enumerate(knn):
        for i in knn_nodes:
            rknn[i].add(node)
    return rknn, knn


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-k", "--knn", type=int, default=5)
    parser.add_argument("-t", "--target_size", type=int, default=10)
    args = parser.parse_args()

    seed_everything(args.seed)

    # * >>> Load the data
    data, _, gnn_pred, node_embeddings = load_dataset(args.dataset)
    num_classes = torch.unique(data.y).size(0)

    # Display class distributions for reference
    class_distribution = torch.unique(data.y, return_counts=True)
    gnn_pred_distribution = np.unique(gnn_pred, return_counts=True)
    print("Number of nodes per class:", class_distribution[1])
    print("Number of nodes per class in GNN Pred:", gnn_pred_distribution[1])

    # For each class, we now choose the concept nodes randomly instead of using a set-cover procedure.
    # We will still compute knn (and reverse knn) per class for later use in concept vector construction.
    # List of lists; each element is the list of node indices for that label.
    label_mapping = []
    # Will store the knn (as computed within each class) needed for global knn computation.
    label_knn = []
    # Computed but not used for concept selection in the ablation.
    label_rknn = []
    # Stores indices (within the label mapping) of the randomly chosen concept nodes.
    label_chosen_nodes = []

    for label in range(num_classes):
        # Select the nodes for which the GNN predicted the current label.
        mask = (label == gnn_pred).astype(bool)
        mapping = np.where(mask)[0].tolist()
        label_mapping.append(mapping)

        # Compute the kNN information on the label–wise embeddings (this is kept to allow comparable creation of concept vectors)
        label_embeddings = node_embeddings[mask]
        distance_mat = label_embeddings @ label_embeddings.T
        rknn, knn = find_reverse_knn(distance_mat, args.knn)
        label_knn.append(knn)
        label_rknn.append(rknn)

        # ***** Ablation Study change: Random selection of concept nodes *****
        # Instead of choosing nodes via a set cover algorithm based on reverse kNN,
        # randomly choose `args.target_size` indices from the available nodes in this label.
        if len(mapping) <= args.target_size:
            # Use all available nodes if fewer than target_size.
            chosen_nodes = list(range(len(mapping)))
        else:
            chosen_nodes = np.random.choice(
                range(len(mapping)), size=args.target_size, replace=False).tolist()
        label_chosen_nodes.append(chosen_nodes)

    # Convert per-label chosen node indices (within each mapping) to global indices.
    chosen_nodes = []
    for i, nodes in enumerate(label_chosen_nodes):
        chosen_nodes.extend([label_mapping[i][node] for node in nodes])
    print("Chosen Nodes (global indices):", sorted(chosen_nodes))

    print("Number of chosen nodes per class:",
          [len(nodes) for nodes in label_chosen_nodes])
    np.save(f"data/{args.dataset}/chosen_nodes.npy", chosen_nodes)

    # Build global knn based on the per-label knn information.
    knn_global = [[] for _ in range(data.num_nodes)]
    for i, node_knn in enumerate(label_knn):
        for j, neighbors in enumerate(node_knn):
            # Convert the local indices within the label mapping to global node indices.
            knn_global[label_mapping[i][j]] = [label_mapping[i][n]
                                               for n in neighbors]

    # Save the knn and rknn structures.
    with open(f"data/{args.dataset}/knn.pkl", "wb") as f:
        pickle.dump(knn_global, f)
    # Build global rknn from label-wise rknn.
    rknn_global = {}
    for i, nodes in enumerate(label_rknn):
        for node, neighbors in nodes.items():
            rknn_global[label_mapping[i][node]] = set(
                label_mapping[i][neighbor] for neighbor in neighbors)
    with open(f"data/{args.dataset}/rknn.pkl", "wb") as f:
        pickle.dump(rknn_global, f)

    # print the sizes of rknn of chosen nodes
    print("Sizes of rknn for chosen nodes:")
    for node in chosen_nodes:
        print(len(rknn_global.get(node, set())), end=", ")

    # Analyze the coverage provided by the selected concepts.
    nodes_covered = set()
    for node in chosen_nodes:
        nodes_covered = nodes_covered.union(rknn_global.get(node, set()))
    fraction_of_nodes_covered = round(len(nodes_covered) / data.num_nodes, 2)
    print(
        f"Percentage of nodes covered by the selected concepts: {fraction_of_nodes_covered}")

    # Count nodes that have empty reverse knn (not covered by any concept) per class.
    classwise_empty_rknn = {i: 0 for i in range(num_classes)}
    for node in range(data.num_nodes):
        if node not in rknn_global.keys():
            classwise_empty_rknn[data.y[node].item()] += 1
    print("Number of nodes with empty rknn per class:", classwise_empty_rknn)
    print("Total number of nodes with empty rknn:",
          sum(classwise_empty_rknn.values()))

    # Calculate per class coverage.
    class_coverage = {}
    for class_label in range(num_classes):
        mask = (data.y == class_label).numpy().astype(bool)
        mapping = np.where(mask)[0].tolist()
        class_coverage[class_label] = len(
            set(mapping).intersection(nodes_covered)) / len(mapping)
    print("Class coverage:", class_coverage)

    # Evaluate accuracy on the chosen concept nodes.
    correct = (data.y[chosen_nodes].numpy() == gnn_pred[chosen_nodes]).sum()
    accuracy = correct / len(chosen_nodes)
    print(f"Accuracy on chosen nodes: {accuracy}")

    # * >>> Create concept vectors based on knn:
    # For each node, if a randomly selected concept (from chosen_nodes) is among its knn neighbors, set the
    # corresponding entry in the concept vector to 1.
    concept_vectors = np.zeros(
        shape=(data.num_nodes, len(chosen_nodes)), dtype=int)
    for node in range(data.num_nodes):
        for i, c_node in enumerate(chosen_nodes):
            if c_node in knn_global[node]:
                concept_vectors[node, i] = 1
    np.save(f"data/{args.dataset}/knn_concept_vectors.npy", concept_vectors)

    # Analysis of concept vectors
    num_zeros = np.sum(np.sum(concept_vectors, axis=1) == 0)
    print("Number of nodes with all zeros in their concept vector:", num_zeros)
    print("Total number of nodes:", data.num_nodes)
