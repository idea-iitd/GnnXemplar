# * >>> Imports and seeds
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import time
from load_dataset import load_dataset
from utils import seed_everything
from sklearn.metrics.pairwise import cosine_similarity
import gnns
from torch_geometric import datasets

import pickle

# * >>> Compute reverse KNN


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
        # better than sorting the array.
        if farthest:
            topk = np.argpartition(distance_mat[node], k)[:k].tolist()
        # topk = np.argpartition(distance_mat[node], k)[:k].tolist()
        else:
            # handle the case when less than k neighbors are present
            if k > num_nodes:
                topk = np.argsort(distance_mat[node]).tolist()
            else:
                topk = np.argpartition(distance_mat[node], -k)[-k:].tolist()
            # topk = np.argpartition(distance_mat[node], -k)[-k:].tolist()
        knn.append(topk)
    rknn = defaultdict(set)

    for node, knn_nodes in enumerate(knn):
        for i in knn_nodes:
            rknn[i].add(node)
    return rknn, knn

# * >>> Max coverage


def find_set_cover(knn, rknn, target_size):
    """This function selects a subset of nodes whose neighbors cover as much of the graph as
    possible while having a relatively low overlap in their reverse k-nearest neighbor sets.

    Args:
        rknn (dict): A dictionary where keys are nodes and values are sets of nodes that have the
        given node as one of their k-nearest neighbors.
        target_size (int): The desired number of nodes to select.

    Returns:
        tuple: The selected subset of nodes.
    """
    sorted_rknn = sorted(
        rknn.items(), key=lambda x: len(x[1]), reverse=True)
    print("Size of sorted rknn:", len(sorted_rknn))
    if len(sorted_rknn) == 0:
        return tuple([])
    chosen_nodes = [sorted_rknn[0][0]]
    not_chosen_nodes = list(rknn.keys())
    not_chosen_nodes.remove(chosen_nodes[0])
    nodes_covered = set(rknn[chosen_nodes[0]])
    del sorted_rknn

    print("Percentage of nodes covered in this class: ",
          round(len(nodes_covered) / len(knn), 2))
    # check if 95% of the nodes are covered
    if len(nodes_covered) / len(knn) >= 0.95:
        return tuple(chosen_nodes)
    if len(not_chosen_nodes) == 0:
        return tuple(chosen_nodes)

    while len(chosen_nodes) < target_size:
        max_coverage = -float("inf")
        max_node = None
        for node in not_chosen_nodes:
            coverage = len(set(rknn[node]).difference(nodes_covered))
            if coverage > max_coverage:
                max_coverage = coverage
                max_node = node
        if max_node is None:
            break
        chosen_nodes.append(max_node)
        nodes_covered = nodes_covered.union(rknn[max_node])
        not_chosen_nodes.remove(max_node)

        # calculate the percentage of nodes covered in this class
        print("Percentage of nodes covered in this class: ",
              round(len(nodes_covered) / len(knn), 2))

        # check if 95% of the nodes are covered
        if len(nodes_covered) / len(knn) >= 0.95:
            # print("90% of the nodes are covered")
            break
        if len(not_chosen_nodes) == 0:
            break

    return tuple(chosen_nodes)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-k", "--knn", type=int, default=5, help="k in KNN")
    parser.add_argument("-t", "--target_size", type=int, default=5,
                        help="Maximum number of exemplars to be selected per class during set cover.")
    parser.add_argument("-r", "--sample_rate", type=float, default=1.0,
                        help="Fraction of nodes to consider for KNN. 1.0 means all nodes.")
    args = parser.parse_args()

    seed_everything(args.seed)

    # * >>> Load the data
    data, _, gnn_pred, node_embeddings = load_dataset(args.dataset)
    num_classes = torch.unique(data.y).size(0)

    # # --- Sample a subset of nodes for concept discovery
    # num_total  = node_embeddings.shape[0]
    # num_sample = max(1, int(num_total * args.sample_rate))
    # sampled_idx = np.random.choice(num_total, num_sample, replace=False)
    # node_embeddings = node_embeddings[sampled_idx]
    # gnn_pred        = gnn_pred[sampled_idx]

    # --- Measure runtime of concept‐discovery loop
    start_time = time.time()

    # calculate the accuracy of gnn on test_mask
    # check if data.test_mask is 2d
    if data.test_mask.ndim == 2:
        test_mask = data.test_mask[:, 0]
    else:
        test_mask = data.test_mask
    correct = (data.y[test_mask].numpy() == gnn_pred[test_mask]).sum()

    accuracy = correct / len(data.y[test_mask])
    print(f"Accuracy of GNN on test set: {accuracy}")

    # import classification_report
    from sklearn.metrics import classification_report
    # print classification report
    print(classification_report(
        data.y[test_mask].numpy(), gnn_pred[test_mask], zero_division=0))
    # exit()

    # find the distribution of the training data among classes
    class_distribution = torch.unique(data.y, return_counts=True)
    gnn_pred_distribution = np.unique(gnn_pred, return_counts=True)

    # print the number of nodes per class
    print("Number of nodes per class:", class_distribution[1])
    print("Number of nodes per class in GNN Pred:", gnn_pred_distribution[1])

    label_mapping = []
    label_knn = []
    label_rknn = []
    label_chosen_nodes = []
    for label in range(num_classes):
        # mask = (data.y == label).numpy().astype(bool)
        # if label == 0:
        #     continue
        mask = (label == gnn_pred).astype(bool)
        mapping = np.where(mask)[0].tolist()

        # 2) stratified sampling per class
        if args.sample_rate < 1.0 and len(mapping) > 0:
            n_keep = max(1, int(len(mapping) * args.sample_rate))
            mapping = list(np.random.choice(mapping, n_keep, replace=False))

        # mask the node embeddings
        # label_embeddings = node_embeddings[mask]
        label_embeddings = node_embeddings[mapping]

        # Compute the pairwise distances between the nodes.
        distance_mat = label_embeddings @ label_embeddings.T

        # Compute reverse KNN
        rknn, knn = find_reverse_knn(distance_mat, args.knn)
        print("Size of knn:", len(knn))
        # print("Size of rknn:", len(rknn))
        # chosen_nodes = find_set_cover(rknn=rknn, target_size=target_sizes[label])
        chosen_nodes = find_set_cover(
            knn=knn, rknn=rknn, target_size=args.target_size)

        label_mapping.append(mapping)
        label_knn.append(knn)
        label_rknn.append(rknn)
        label_chosen_nodes.append(chosen_nodes)

    elapsed = time.time() - start_time
    print(
        f"[Sampling {args.sample_rate*100:.1f}%] Concept discovery runtime: {elapsed:.4f}s")

    # use the label mapping to get the original node indices
    chosen_nodes = []
    for i, nodes in enumerate(label_chosen_nodes):
        chosen_nodes.extend(label_mapping[i][node] for node in nodes)
    print("Chosen Nodes:", sorted(chosen_nodes))
    # print number of chosen nodes per class
    print("Number of chosen nodes per class:", [
          len(nodes) for nodes in label_chosen_nodes])
    np.save(f"data/{args.dataset}/chosen_nodes.npy", chosen_nodes)

    # define knn as a list of lists, and intialize it with empty lists
    knn = [[] for _ in range(data.num_nodes)]
    # label_knn is a list of list of arrays
    # iterate over the list and store the knn data into knn list
    for i, node_data in enumerate(label_knn):
        for j, neighbors in enumerate(node_data):
            knn[label_mapping[i][j]] = [label_mapping[i][node]
                                        for node in neighbors]

    # define rknn as a dictionary of sets, and initialize it with empty sets
    rknn = {}
    for i, nodes in enumerate(label_rknn):
        for node, neighbors in nodes.items():
            rknn[label_mapping[i][node]] = set(
                label_mapping[i][neighbor] for neighbor in neighbors)

    # save knn and rknn
    with open(f"data/{args.dataset}/knn.pkl", "wb") as f:
        pickle.dump(knn, f)
    with open(f"data/{args.dataset}/rknn.pkl", "wb") as f:
        pickle.dump(rknn, f)

    # set chosen_nodes (a list) as all nodes in rknn.keys()
    # chosen_nodes = list(rknn.keys())
    # print("Size of chosen nodes:", len(chosen_nodes))
    nodes_covered = set()
    for node in chosen_nodes:
        nodes_covered = nodes_covered.union(rknn[node])
    fraction_of_nodes_covered = round(len(nodes_covered) / data.num_nodes, 2)
    print(f"Percentage of nodes covered: {fraction_of_nodes_covered}")

    # print number of nodes which have rknn empty from each class
    classwise_empty_rknn = {i: 0 for i in range(num_classes)}
    for node in range(data.num_nodes):
        if node not in rknn.keys():
            # add to the dictionary
            classwise_empty_rknn[data.y[node].item()] += 1

    print("Number of nodes with empty rknn:", classwise_empty_rknn)
    print("Total number of nodes with empty rknn:",
          sum(classwise_empty_rknn.values()))

    # calculate per class coverage
    class_coverage = {}
    for class_label in range(num_classes):
        mask = (data.y == class_label).numpy().astype(bool)
        mapping = np.where(mask)[0].tolist()
        class_coverage[class_label] = len(
            set(mapping).intersection(nodes_covered)) / len(mapping)
    print("Class coverage:", class_coverage)

    # print(chosen_nodes)
    # print("Labels of chosen nodes: ", data.y[chosen_nodes])
    # print("GNN Preds of chosen nodes: ", gnn_pred[chosen_nodes])
    # calc accuracy on the chosen nodes
    correct = (data.y[chosen_nodes].numpy() == gnn_pred[chosen_nodes]).sum()
    accuracy = correct / len(chosen_nodes)
    print(f"Accuracy on chosen nodes: {accuracy}")

    # * >>> Create concept vectors
    # Create a matrix of zeros of size (#nodes, #chosen_nodes)
    concept_vectors = np.zeros(
        shape=(data.num_nodes, len(chosen_nodes)), dtype=int)
    # Iterate over the nodes
    for node in range(data.num_nodes):
        # Iterate over the chosen nodes
        for i, c_node in enumerate(chosen_nodes):
            # if c_node is in the knn of the node
            if c_node in knn[node]:
                # set corresponding entry to 1
                concept_vectors[node, i] = 1
    # np.save(f"data/{args.dataset}/concept_vectors.npy", concept_vectors)
    np.save(f"data/{args.dataset}/knn_concept_vectors.npy", concept_vectors)

    # Analysis of concepts
    # calculate the number of nodes which have all zeros in their concept vector
    num_zeros = np.sum(concept_vectors == 0, axis=1)
    num_zeros = np.sum(num_zeros == len(chosen_nodes))
    print("Number of nodes with all zeros in their concept vector:", num_zeros)
    print("Total number of nodes:", data.num_nodes)


#     pairwise_distances = node_embeddings @ node_embeddings.T
#     rknn, knn = find_reverse_knn(pairwise_distances, 5)
# # for each concept node print the size of its rknn
#     for node in chosen_nodes:
#         print(f"Size of rknn of node {node}: {len(rknn[node])}")

    # # #* CLUSTERING BASED ON CHOSEN NODES TO VISUALISE THE DISTRIBUTION

    # # form clusters based on the chosen nodes, that is each chosen node has a cluster around it
    # clusters = []
    # for node in chosen_nodes:
    #     cluster = list(rknn[node])
    #     clusters.append(cluster)

    # distance_mat = node_embeddings @ node_embeddings.T
    # # plot a graph of the distribution of the intracluster distances (distances from chosen node for that cluster) for each cluster
    # intracluster_distances = []
    # for i, cluster in enumerate(clusters):
    #     # intracluster_distances = []
    #     for node in cluster:
    #         distance = distance_mat[node, chosen_nodes[i]]
    #         intracluster_distances.append(distance)

    # # print("Size of intracluster distances:", len(intracluster_distances))
    # # plot the distribution of the intracluster distances
    # import matplotlib.pyplot as plt
    # plt.hist(intracluster_distances, bins=100)
    # plt.title("Distribution of intracluster distances")
    # plt.xlabel("Distance")
    # plt.ylabel("Frequency")
    # plt.show()
    # plt.savefig(f"data/{args.dataset}/intracluster_distances.png")

    # # inter cluster is the distance from chosen node of one cluster to all nodes of other clusters, and do this for all clusters
    # intercluster_distances = []
    # for i, cluster in enumerate(clusters):
    #     # the other cluster can be all clusters except the current cluster
    #     other_clusters = clusters[:i] + clusters[i+1:]
    #     # now calculate the distance from the chosen node of the current cluster to all nodes of the other clusters
    #     for other_cluster in other_clusters:
    #         for other_node in other_cluster:
    #             distance = distance_mat[other_node, chosen_nodes[i]]
    #             intercluster_distances.append(distance)

    # # print("Size of intercluster distances:", len(intercluster_distances))

    # # now plot the intercluster distances (distances between the chosen nodes)
    # # clear the old plot
    # plt.clf()
    # plt.hist(intercluster_distances, bins=100)
    # plt.title("Distribution of intercluster distances")
    # plt.xlabel("Distance")
    # plt.ylabel("Frequency")
    # plt.show()
    # plt.savefig(f"data/{args.dataset}/intercluster_distances.png")

    # # get the global knn for each node
    # distance_mat = node_embeddings @ node_embeddings.T
    # # Compute reverse KNN
    # rknn, knn = find_reverse_knn(distance_mat, 5)
    # # chosen_nodes = find_set_cover(rknn=rknn, target_size=100)
    # # form the concept vectors
    # concept_vectors = np.zeros(
    #     shape=(data.num_nodes, len(chosen_nodes)), dtype=int)
    # # Iterate over the nodes
    # for node in range(data.num_nodes):
    #     # Iterate over the chosen nodes
    #     for i, c_node in enumerate(chosen_nodes):
    #         # if c_node is in the knn of the node
    #         if c_node in knn[node]:
    #             # set corresponding entry to 1
    #             concept_vectors[node, i] = 1

    # # for each node, find the number of exemplars from its own class and from other classes in its knn
    # num_exemplars = []
    # for node in range(data.num_nodes):
    #     # find the concept nodes in the knn of the current node
    #     concept_vector = concept_vectors[node]
    #     concepts = np.where(concept_vector == 1)[0]
    #     concepts = [chosen_nodes[concept] for concept in concepts]
    #     # get the labels of the concept nodes in the knn of the current node
    #     concept_labels = data.y[concepts]
    #     # get the label of the current node
    #     node_label = data.y[node]
    #     # find the number of exemplars from the same class and from other classes
    #     num_same_class = (concept_labels == node_label).sum()
    #     num_other_class = len(concepts) - num_same_class
    #     num_exemplars.append((num_same_class.item(), num_other_class.item()))
    # # print(num_exemplars)
    # # print number of exemplars from the same class and from other classes
    # same_class = 0
    # other_class = 0
    # for num in num_exemplars:
    #     same_class += num[0]
    #     other_class += num[1]
    # print("Number of exemplars from the same class:", same_class)
    # print("Number of exemplars from other classes:", other_class)

    # # plot the dsitribution of the number of exemplars from the same class and from other classes
    # num_same_class = [num[0] for num in num_exemplars]
    # num_other_class = [num[1] for num in num_exemplars]
    # plt.clf()
    # # replace the frequency with the percentage
    # plt.hist(num_same_class, bins=10, label='Same Class', weights=np.ones(len(num_same_class)) / len(num_same_class) * 100, )

    # # plt.hist(num_other_class, label='Other Class')
    # plt.legend(loc='upper right')
    # plt.title("Distribution of number of exemplars from same class")
    # plt.xlabel("Number of exemplars")
    # plt.ylabel("Percentage of nodes")
    # plt.show()
    # plt.savefig(f"data/{args.dataset}/num_exemplars.png")

    # plt.clf()
    # plt.hist(num_other_class,bins=10, label='Other Class', weights=np.ones(len(num_other_class)) / len(num_other_class) * 100)
    # plt.legend(loc='upper right')
    # plt.title("Distribution of number of exemplars from other classes")
    # plt.xlabel("Number of exemplars")
    # plt.ylabel("Percentage of nodes")
    # plt.show()
    # plt.savefig(f"data/{args.dataset}/num_exemplars_other.png")

    # exit()

    # # # USE A GRADIENT EXPLAINER ON THE NODE EMBEDDINGS
    # # from captum.attr import IntegratedGradients

    # # dataset = datasets.Planetoid(
    # #             root="data", name="CORA", split="public", force_reload=True,)
    # # data = dataset[0]
    # # model = gnns.CORA_GAT(in_channels=dataset.num_features,
    # #                               num_classes=dataset.num_classes)
    # # model.load_state_dict(torch.load("data/CORA/state_dict.pt"))

    # # # # 4. Define a helper function to return logits for Integrated Gradients
    # # # def model_forward_logits(x, edge_index):
    # # #     model.explaining = True  # Return logits without softmax
    # # #     logits = model(x, edge_index)
    # # #     model.explaining = False
    # # #     return logits.unsqueeze(0)
    # # # # 4. Define a helper function to return the scalar logit for a specific node and class
    # # # def get_node_class_logit(node_idx, target_class):
    # # #     def forward_fn(x, edge_index):
    # # #         model.explaining = True  # Return logits without softmax
    # # #         logits = model(x, edge_index)  # Shape: [num_nodes, num_classes]
    # # #         model.explaining = False
    # # #         # Return the logit for the specific node and class as a 1-element tensor
    # # #         return logits[node_idx, target_class].unsqueeze(0)
    # # #     return forward_fn

    # # # # 4. Define a helper function to return flattened logits
    # # # def model_forward_flat_logits(x, edge_index):
    # # #     model.explaining = True  # Return logits without softmax
    # # #     logits = model(x, edge_index)  # Shape: [num_nodes, num_classes]
    # # #     model.explaining = False
    # # #     return logits.view(-1)  # Shape: [num_nodes * num_classes]

    # # # model.eval()

    # # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # # # 6. Instantiate the Integrated Gradients explainer
    # # # ig = IntegratedGradients(model_forward_flat_logits)

    # # # # 7. Compute Attributions for Individual Nodes
    # # # num_nodes = data.num_nodes
    # # # num_features = data.num_node_features
    # # # num_classes = dataset.num_classes
    # # # all_attributions = np.zeros((num_nodes, num_features))
    # # # baseline = torch.zeros_like(data.x).to(device)

    # # # print("Starting attribution computation for each node...")

    # # # for node_idx in range(num_nodes):
    # # #     # Get the target class for the current node
    # # #     target_class = data.y[node_idx].item()

    # # #     # Calculate the target index in the flattened logits
    # # #     target = node_idx * num_classes + target_class

    # # #     # Clone and require gradients for input features
    # # #     input_features = data.x.clone().detach().requires_grad_(True).to(device)

    # # #     # Compute attributions using Integrated Gradients
    # # #     attributions = ig.attribute(
    # # #         inputs=input_features,
    # # #         baselines=baseline,
    # # #         additional_forward_args=(data.edge_index),
    # # #         target=target,
    # # #         n_steps=50
    # # #     )

    # # #     # Move attributions to CPU and convert to numpy
    # # #     attributions = attributions.detach().cpu().numpy()

    # # #     # Store the absolute attributions for the current node
    # # #     all_attributions[node_idx] = np.abs(attributions)

    # # #     # Progress update
    # # #     if (node_idx + 1) % 500 == 0 or (node_idx + 1) == num_nodes:
    # # #         print(f'Computed attributions for {node_idx + 1}/{num_nodes} nodes.')

    # # # print("Attribution computation completed.")

    # # # # 8. Aggregate Attributions for Global Feature Importance
    # # # global_attributions = np.mean(all_attributions, axis=0)
    # # # global_attributions_normalized = global_attributions / np.linalg.norm(global_attributions)

    # # # # 9. Print the top 10 most important features
    # # # top_k = 10
    # # # top_features = np.argsort(global_attributions_normalized)[-top_k:][::-1]
    # # # print("\nTop 10 Global Feature Importances:")
    # # # for idx in top_features:
    # # #     print(f"Feature {idx}: {global_attributions_normalized[idx]:.6f}")

    # # # # 10. Visualize the Feature Importances
    # # # plt.figure(figsize=(12, 6))
    # # # plt.bar(range(len(global_attributions_normalized)), global_attributions_normalized, color='skyblue')
    # # # plt.xlabel('Input Feature Index')
    # # # plt.ylabel('Normalized Average Attribution')
    # # # plt.title('Global Feature Importance in GAT (Integrated Gradients)')
    # # # plt.xticks(range(0, len(global_attributions_normalized), 100))  # Adjust ticks for readability
    # # # plt.show()
