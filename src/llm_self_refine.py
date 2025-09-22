from load_dataset import load_dataset
import os
import sys
import json
import pickle
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, ThresholdConfig
from torch_geometric.utils import k_hop_subgraph
import google.generativeai as genai

from argparse import ArgumentParser
from utils import numpy_encoder

######################################################
# Load Dataset & Setup
######################################################
parser = ArgumentParser()
parser.add_argument(
    "-d", "--dataset",
    type=str,
    default='TAGCora',
    help="Dataset to use. Options: bashapes, KarateClub, CORA, TAGCora, WikiCS, Twitch, arxiv, Citeseer, questions, roman, amazonratings"
)
parser.add_argument(
    "-p", "--phase",
    type=int,
    default=1,
    choices=[1, 2, 3],
    help=("Phase to run."
          "1) Discover signatures and run phases 2 and 3."
          "2) If phase 1 is already run, run phase 2 to combine the signatures to get the global explanation of each class."
          "3): If phase 2 is already run, run phase 3 to get a human-interpretable version of the global explanations.")
)
parser.add_argument(
    "-s", "--seed",
    type=int,
    default=2001,
    help="Seed for random number generator"
)
parser.add_argument(
    "-i", "--in_context",
    type=bool,
    default=True,
    help="Use in-context examples. Options: True, False"
)

# add arguments for temperate, batch_size, epochs, max_cluster_population
parser.add_argument(
    "-t", "--temperature",
    type=float,
    default=0.1,
    help="Temperature for LLM generation"
)

parser.add_argument(
    "-b", "--batch_size",
    type=int,
    default=20,
    help="Sample size for training."
)

parser.add_argument(
    "-e", "--epochs",
    type=int,
    default=5,
    help="Number of epochs for LLM generation"
)

args = parser.parse_args()
# test the generated explanation on this number of nodes from the dataset.
npr = np.random.RandomState(seed=args.seed)
max_cluster_population = 500

data, model, gnn_pred, node_embeddings = load_dataset(args.dataset)
num_classes = len(np.unique(data.y))
test_mask = np.arange(data.num_nodes)

# Load chosen nodes and kNN data
chosen_nodes = np.load(f'data/{args.dataset}/chosen_nodes.npy')
with open(f'data/{args.dataset}/rknn.pkl', 'rb') as f:
    rknn = pickle.load(f)
with open(f'data/{args.dataset}/knn.pkl', 'rb') as f:
    knn = pickle.load(f)

# LLM model parameters
GOOGLE_API_KEY = 'YOUR_API_KEY_HERE'
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


def get_response(prompt, temperature=0.1):
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=temperature
        )
    )
    return response.text


# read from json file
with open(f"data/{args.dataset}/dataset_config.json", "r", encoding="utf-8") as f:
    dataset_config = json.load(f)
in_context_examples = dataset_config["in_context_examples"]
class_description = dataset_config["class_description"]
dataset_description = dataset_config["dataset_description"]


# modes = ["no-features", "dense", "one-hot", "tag"]

if args.dataset == 'bashapes':
    mode = 'no-features'
    max_cluster_population = 20

elif args.dataset == 'TAGCora':
    data.edge_index = pyg.utils.to_undirected(data.edge_index)
    max_cluster_population = 200
    mode = 'tag'

elif args.dataset == 'WikiCS':
    mode = 'dense'
    max_cluster_population = 500
    test_mask = []
    test_nodes_sample_size = 50
    for i in range(num_classes):
        nodes = np.where(gnn_pred == i)[0]
        test_mask.extend(npr.choice(nodes, min(
            test_nodes_sample_size, len(nodes)), replace=False))
    print("Class distribution of test_mask: ",
          np.bincount(gnn_pred[test_mask]))

elif args.dataset == 'arxiv':
    mode = 'dense'
    max_cluster_population = 200
    test_mask = []
    for i in range(num_classes):
        nodes = np.where(gnn_pred == i)[0]
        test_mask.extend(npr.choice(
            nodes, min(20, len(nodes)), replace=False))
    print("Class distribution of test_mask: ",
          np.bincount(gnn_pred[test_mask]))


elif args.dataset == 'Citeseer':
    mode = 'dense'
    max_cluster_population = 10
    test_mask = npr.choice(data.num_nodes, 1000, replace=False)

elif args.dataset == 'questions':
    mode = 'dense'
    max_cluster_population = 500

    test_mask = npr.choice(data.num_nodes, 500, replace=False)
    test_mask = np.union1d(test_mask, np.where(gnn_pred == 1)[0])


elif args.dataset == 'roman':
    mode = 'dense'
    max_cluster_population = 500
    # test_mask = npr.choice(data.num_nodes, 1000, replace=False)
    # select 50 nodes from each class
    test_mask = []
    for i in range(num_classes):
        nodes = np.where(gnn_pred == i)[0]
        test_mask.extend(npr.choice(
            nodes, min(20, len(nodes)), replace=False))
    print("Class distribution of test_mask: ",
          np.bincount(gnn_pred[test_mask]))


elif args.dataset == 'amazonratings':
    mode = 'dense'
    max_cluster_population = 500
    test_mask = npr.choice(data.num_nodes, 1000, replace=False)

elif args.dataset == 'minesweeper':
    mode = 'one-hot'
    max_cluster_population = 500
    test_mask = npr.choice(data.num_nodes, 10000, replace=False)

# Prepare Concepts dict
concepts = {}
for i in range(num_classes):
    concepts[i] = [node for node in chosen_nodes if gnn_pred[node] == i]
print("Concepts: ", concepts)


def generate_class_cluster_profiles(data):
    """
    Generate cluster profiles for each class based on concept nodes.

    For each class, define a 'cluster' for each concept node.
    Each cluster consists of all nodes in the reverse k-nearest neighbors (rknn) of that concept node.

    Args:
        data: PyTorch Geometric data object containing the graph structure

    Returns:
        dict: A dictionary where keys are class labels and values are lists of cluster profiles.
                Each cluster profile is a dictionary containing:
                - class_label: The class of the concept node
                - cluster_id: Same as the concept node ID
                - concept_node: The ID of the concept node
                - cluster_size: Number of nodes in the cluster
                - population_ids: List of node IDs in the cluster
    """
    class_cluster_profiles = {}
    for i in range(num_classes):
        class_cluster_profiles[i] = []
        for concept_node in concepts[i]:
            # check if the concept node is in the rknn
            if concept_node not in rknn:
                cluster_nodes = []
            else:
                cluster_nodes = rknn[concept_node]
            cluster_profile = {
                "class_label": i,
                "cluster_id": int(concept_node),
                "concept_node": int(concept_node),
                "cluster_size": int(len(cluster_nodes)),
                "population_ids": list(cluster_nodes),
            }
            class_cluster_profiles[i].append(cluster_profile)
    return class_cluster_profiles


def generate_node_description(node, data):
    """
    Generate a description of a node capturing its features and neighborhood information.

    This function creates a dictionary capturing the input available to the GNN:
    - Node features (depending on the mode: dense, one-hot, meaningful, or tag)
    - Frequency distribution of GNN predicted classes among 1-hop and 2-hop neighbors.

    Args:
        node (int): The ID of the node to describe
        data: PyTorch Geometric data object containing the graph structure

    Returns:
        dict: A dictionary containing node description with:
            - node_id: ID of the node
            - features: Node features (format depends on the mode)
            - 1-hop: Dictionary with neighbor_class_freq showing class distribution of 1-hop neighbors
            - 2-hop: Dictionary with neighbor_class_freq showing class distribution of 2-hop neighbors
    """
    node_description = {}
    node_description['node_id'] = int(node)
    # if mode == 'feature-vector' and features_desc:
    if mode == 'dense':
        # node_description['features'] = data.x[node].tolist()
        # calculate the indices where the node features are non-zero
        # nonzero_indices = torch.nonzero(data.x[node]).squeeze().tolist()
        # node_description['features'] = nonzero_indices

        # calculate the dot product of the concept node's feature vector with all chosen nodes's features
        node_features = data.x[node].tolist()
        chosen_nodes_features = [data.x[node].tolist()
                                 for node in chosen_nodes]
        # calculate the dot product of the concept node's feature vector with all chosen nodes's features
        dot_products = np.dot(chosen_nodes_features, node_features)
        node_description['features'] = dot_products
    elif mode == 'one-hot':
        # calculate the indices where the node features are non-zero
        nonzero_indices = torch.nonzero(data.x[node]).squeeze().tolist()
        node_description['features'] = nonzero_indices
    elif mode == 'meaningful':
        node_description['features'] = data.x[node].tolist()
    elif mode == 'tag':
        node_description['features'] = data.raw_text[node]

    # Compute 1-hop neighborhood using k_hop_subgraph
    node_tensor = torch.tensor([node])
    sub_nodes_1, _, _, _ = k_hop_subgraph(
        node_tensor, 1, data.edge_index, relabel_nodes=False
    )

    # Count frequencies of GNN predicted classes among neighbors
    # Also compute the distribution of feature values among neighbors for one hot

    neighbor_classes = []
    neighbor_features = []
    for sub_n in sub_nodes_1:
        if sub_n.item() != node:
            neighbor_classes.append(int(gnn_pred[sub_n]))
            # if mode == 'one-hot':
            #     # calculate the indices where the node features are non-zero
            #     nonzero_indices = torch.nonzero(data.x[sub_n]).squeeze().tolist()
            #     neighbor_features.append(nonzero_indices)

    freq_dict = {}
    for cls in neighbor_classes:
        freq_dict[cls] = freq_dict.get(cls, 0) + 1

    features_dict = {}
    for feature_value in neighbor_features:
        features_dict[feature_value] = features_dict.get(
            feature_value, 0) + 1

    total_neighbors = len(neighbor_classes)
    # Convert counts to frequencies (percentages) if neighbors exist
    if total_neighbors > 0:
        freq_percentage = {
            cls: count/total_neighbors for cls, count in freq_dict.items()}
    else:
        freq_percentage = {}
    freq_percentage = dict(sorted(freq_percentage.items()))

    # Store the frequency distribution in the node description under the 1-hop key
    node_description['1-hop'] = {}
    node_description['1-hop']['neighbor_class_freq'] = freq_percentage
    # node_description['1-hop']['neighbor_class_freq'] = freq_dict

    #  add description of 2-hop neighbors
    node_description['2-hop'] = {}

    sub_nodes_2, _, _, _ = k_hop_subgraph(
        node_tensor, 2, data.edge_index, relabel_nodes=False
    )
    neighbor_classes = []
    for sub_n2 in sub_nodes_2:
        if sub_n2.item() != node:
            neighbor_classes.append(int(gnn_pred[sub_n2]))
    freq_dict = {}
    for cls in neighbor_classes:
        freq_dict[cls] = freq_dict.get(cls, 0) + 1
    total_neighbors = len(neighbor_classes)
    # Convert counts to frequencies (percentages) if neighbors exist
    if total_neighbors > 0:
        freq_percentage = {
            cls: count/total_neighbors for cls, count in freq_dict.items()}
    else:
        freq_percentage = {}
    # sort the dictionary by key
    freq_percentage = dict(sorted(freq_percentage.items()))
    node_description['2-hop']['neighbor_class_freq'] = freq_percentage
    # node_description['2-hop']['neighbor_class_freq'] = freq_dict
    return node_description


def get_misclassifications(formula_code, cluster_nodes, non_cluster_nodes, data):
    """
    Identifies nodes misclassified by a given classification formula.

    Args:
        formula_code (str): String containing Python code defining a classify_node function
        cluster_nodes (list): List of node IDs that should be classified as 1 (positive)
        non_cluster_nodes (list): List of node IDs that should be classified as 0 (negative)
        data: PyTorch Geometric data object containing the graph structure

    Returns:
        tuple: Contains:
            - false_negatives (list): Cluster nodes misclassified as non-cluster (should be 1, got 0)
            - false_positives (list): Non-cluster nodes misclassified as cluster (should be 0, got 1)
            - error (Exception or None): Any error that occurred during execution, None if successful
    """

    if not formula_code:
        return cluster_nodes, non_cluster_nodes
    local_namespace = {}
    try:
        exec(formula_code, globals(), local_namespace)
    except Exception as e:
        print("Error in executing formula code.", e)
        return cluster_nodes, non_cluster_nodes, e
    classify_node = local_namespace['classify_node']
    false_negatives = []
    false_positives = []
    try:
        for node in cluster_nodes:
            pred = classify_node(generate_node_description(node, data))
            if pred == 0:
                false_negatives.append(node)
        for node in non_cluster_nodes:
            pred = classify_node(generate_node_description(node, data))
            if pred == 1:
                false_positives.append(node)
    except Exception as e:
        print("Error in executing formula code.", e)
        return cluster_nodes, non_cluster_nodes, e

    return false_negatives, false_positives, None


def calculate_global_accuracy(formula_code, cluster_nodes, non_cluster_nodes, data):
    """
    Calculates accuracy metrics for a classification formula against a node cluster.

    Given a candidate Python formula for classify_node(), a list of cluster_nodes (positives),
    and a list of non_cluster_nodes (negatives), this function computes:
    - Positive accuracy: fraction of cluster nodes correctly classified (should output 1)
    - Negative accuracy: fraction of non-cluster nodes correctly classified (should output 0)
    - Global accuracy: average of positive and negative accuracies.

    Args:
        formula_code (str): String containing Python code defining a classify_node function
        cluster_nodes (list): List of node IDs that should be classified as 1 (positive)
        non_cluster_nodes (list): List of node IDs that should be classified as 0 (negative)
        data: PyTorch Geometric data object containing the graph structure

    Returns:
        tuple: Contains:
            - pos_acc (float): Positive accuracy (true positive rate)
            - neg_acc (float): Negative accuracy (true negative rate)
            - global_acc (float): Global accuracy = (pos_acc + neg_acc) / 2
    """
    # Get misclassifications using the provided utility function.
    false_negatives, false_positives, error = get_misclassifications(
        formula_code, cluster_nodes, non_cluster_nodes, data)

    if len(cluster_nodes) > 0:
        pos_acc = 1.0 - (len(false_negatives) / len(cluster_nodes))
    else:
        pos_acc = 0.0

    if len(non_cluster_nodes) > 0:
        neg_acc = 1.0 - (len(false_positives) / len(non_cluster_nodes))
    else:
        neg_acc = 0.0

    global_acc = (pos_acc + neg_acc) / 2.0
    return pos_acc, neg_acc, global_acc


# todo: take global variables as input and update them after the function call.
init_prompt = ()


def generate_initial_formula(cluster, cluster_nodes, non_cluster_nodes, data, dataset_description, in_context=False):
    """
    Generates an initial rule-based formula to classify nodes in a cluster.

    Uses an LLM to create a symbolic rule-based formula that outputs 1 for nodes that belong 
    to the specified cluster and 0 for nodes that don't. The LLM is given the concept node, 
    a sample of cluster nodes, and a sample of non-cluster nodes along with their descriptions.

    Args:
        cluster (dict): Dictionary containing cluster information
        cluster_nodes (list): List of node IDs belonging to the cluster
        non_cluster_nodes (list): List of node IDs not belonging to the cluster
        data: PyTorch Geometric data object containing the graph structure
        dataset_description (str): Description of the dataset
        in_context (bool, optional): Whether to include in-context learning examples. Defaults to False.

    Returns:
        str: The LLM response containing rules and Python code
    """

    prompt = (
        # dataset_description + "\n" +
        "You are given a concept node and a set of nodes that belong to its cluster (from a GNN embedding). "
        "Also provided are some nodes that do NOT belong to this cluster. "
        "Please propose an INITIAL symbolic rule-based formula that outputs 1 for cluster nodes and 0 for non-cluster nodes. Remember that your rules should not be generic and should be specific to the concept node and its cluster. "
        "Rely ONLY on node features and local adjacency information. Focus on finding what features and structural neighborhood information makes the cluster nodes similar and the non cluter nodes dissimilar to the concept node. Use them to form the rules.\n\n"
        "Output your explanation in JSON with a top-level 'rules' list and an 'interpretation' key. Also implement a Python function called classify_node() that takes a node_description as input and returns 1 or 0. Ensure the code can run independently without external file dependencies and without any errors. Avoid using regex, or importing any more external libraries. Also ensure that all the code is inside the classify_node function only, and nothing should be outside it. Also do NOT use eval() function. \n\n"
    )
    concept_node = cluster['concept_node']
    prompt += f"Concept node: {concept_node}\n"
    prompt += f"Concept node description: {generate_node_description(concept_node, data)}\n"
    prompt += f"Cluster nodes: {cluster_nodes}\n"
    prompt += f"Non-cluster nodes: {non_cluster_nodes}\n\n"
    node_descriptions = {}
    node_descriptions[concept_node] = generate_node_description(
        concept_node, data)
    for n in cluster_nodes:
        node_descriptions[n] = generate_node_description(n, data)
    for n in non_cluster_nodes:
        node_descriptions[n] = generate_node_description(n, data)
    prompt += "Descriptions of cluster nodes:\n"
    for n in cluster_nodes:
        prompt += f"Node {n}: {node_descriptions[n]}\n"
    prompt += "\nDescriptions of non-cluster nodes:\n"
    for n in non_cluster_nodes:
        prompt += f"Node {n}: {node_descriptions[n]}\n"

    # add in context examples to the prompt
    if in_context:
        prompt += "Here are some toy examples for you to understand how you can use the node descriptions to form rules:\n"

        global in_context_examples

        prompt += in_context_examples + "\n\n"

    with open(f'data/{args.dataset}/query_prompt.txt', 'w') as f_q:
        f_q.write(prompt)
    global init_prompt
    init_prompt = prompt
    response = get_response(prompt, temperature=args.temperature)
    with open(f'data/{args.dataset}/llm_init_output.txt', 'w') as f_init:
        f_init.write(response)
    return response


def generate_feedback(current_formula, false_negatives, false_positives, data, history, error):
    """
    Generates actionable feedback for improving a rule-based formula based on its misclassifications.

    This function analyzes the current formula and its performance on cluster nodes and non-cluster
    nodes, focusing on false negatives and false positives. It prepares a prompt for the LLM to
    provide specific, actionable feedback on how to modify the formula to reduce misclassifications.

    Parameters:
        current_formula (str): Python code for the current formula being evaluated
        false_negatives (list): Node IDs that are in the cluster but classified as non-cluster
        false_positives (list): Node IDs that are not in the cluster but classified as cluster
        data (torch_geometric.data.Data): The graph data object containing node features and structure
        history (list): Previous iterations of formula refinement with feedback
        error (Exception or None): Any error encountered during formula execution

    Returns:
        str: LLM-generated feedback suggesting modifications to improve the formula's accuracy
    """

    prompt = f"Here is the initial prompt: ({init_prompt})\n"
    # prompt += "Below is the history of previous iterations:\n"
    # for idx, (prev_formula, prev_feedback, prev_pos_acc, prev_neg_acc) in enumerate(history):
    #     prompt += f"Iteration {idx}:\nFormula:\n{prev_formula}\n Positives Accuracy: {prev_pos_acc}, Negatives Accuracy: {prev_neg_acc}\nFeedback:\n{prev_feedback}\n\n"
    if error is not None:
        prompt += f"Here is current formula: {current_formula} \n and this threw an error: {error} \n"
    else:
        prompt += (
            "Below is the CURRENT formula:\n" +
            f"{current_formula}\n\n" +
            "It was evaluated on cluster nodes (should be 1) and non-cluster nodes (should be 0). "
            f"It yielded {len(false_negatives)} false negatives and {len(false_positives)} false positives.\n\n"
            "False Negatives (Cluster nodes misclassified as non-cluster):\n"
        )
    for node in false_negatives:
        prompt += f"Node {node}: {generate_node_description(node, data)}\n"
    prompt += "\nFalse Positives(Non-cluster nodes misclassified as cluster):\n"
    for node in false_positives:
        prompt += f"Node {node}: {generate_node_description(node, data)}\n"
    prompt += (
        "\n Your goal is to improve the formula so that it does not make these false predictions (misclassifications) while preserving the performance on the rest of the correct predictions. Please provide detailed feedback on how to adjust the formula. Focus on which conditions "
        "need to be added, removed, or modified and specifically mention these actionable points in your output. Give a single most important change to make in the current formula as your actionable feedback. Do not provide the final revised code yet. \n"
    )
    with open(f'data/{args.dataset}/feedback_prompt.txt', 'w') as f:
        f.write(prompt)
    feedback = get_response(prompt, temperature=args.temperature)
    return feedback


def refine_formula(current_formula, feedback_text, false_negatives, false_positives, history, error):
    """
    Refines a formula based on feedback and historical performance data.

    This function uses the provided feedback, current formula, and the history of previous
    iterations to generate an improved version of the formula. It leverages an LLM to understand
    the history and feedback and produce a revised formula that aims to achieve at least 95%
    positive and negative accuracy.

    Parameters:
        current_formula (str): Python code for the current formula being refined
        feedback_text (str): LLM-generated feedback on how to improve the formula
        false_negatives (list): Node IDs that are in the cluster but classified as non-cluster
        false_positives (list): Node IDs that are not in the cluster but classified as cluster
        history (list): Previous iterations of formula refinement with feedback and accuracy results
        error (Exception or None): Any error encountered during formula execution

    Returns:
        str: LLM-generated response containing the revised formula with rules and interpretation
    """

    prompt = f"Here is the initial prompt: ({init_prompt})\n"
    prompt += "Below is the history of previous iterations:\n"
    for idx, (prev_formula, prev_feedback, prev_pos_acc, prev_neg_acc, prev_error) in enumerate(history):
        if prev_error is None:
            prompt += f"Iteration {idx}:\nFormula:\n{prev_formula}\n Positives Accuracy: {prev_pos_acc}, Negatives Accuracy: {prev_neg_acc}, Total Accuracy: {0.5 * (prev_pos_acc + prev_neg_acc)}\nFeedback:\n{prev_feedback}\n\n"
        else:
            prompt += f"Iteration {idx}:\nFormula:\n{prev_formula}\n \nError: {prev_error}\n\n"
    prompt += (
        # "Now, the CURRENT formula is:\n" +
        # current_formula + "\n\n" +
        # "It produced the following misclassifications:\n" +
        # f"False Negatives: {false_negatives}\nFalse Positives: {false_positives}\n\n" +
        "Using the history above, and the last feedback, please produce a REVISED formula in JSON format with a top-level 'rules' list and an 'interpretation' key. Your goal is to improve the accuracy metrics by addressing the feedback provided. You must achieve atleast 95% positive and negative accuracy. \n"
        "Also implement a Python function called classify_node() that takes a node_description as input and returns 1 or 0. Ensure the code can run independently without external file dependencies and without any errors. Avoid using regex, or importing any more external libraries. Also ensure that all the code is inside the classify_node function only, and nothing should be outside it. Refer to the in context examples to generate the rules. \n"
    )
    with open(f'data/{args.dataset}/refine_prompt.txt', 'w') as f_ref:
        f_ref.write(prompt)
    response = get_response(prompt, temperature=args.temperature)
    with open(f'data/{args.dataset}/llm_refine_output.txt', 'w') as f_ref_out:
        f_ref_out.write(response)
    return response


def train_cluster_LLM_self_refine(cluster, data, class_cluster_profiles):
    """
    Trains an LLM through self-refinement to generate symbolic rules for a cluster.

    This function implements an iterative process where an LLM generates, tests, and refines
    symbolic rules to explain a cluster of nodes. It selects positive nodes (from the cluster)
    and negative nodes (from other classes) for training. Over multiple epochs, it refines
    the formula based on its performance on these nodes, aiming to achieve high accuracy.

    Parameters:
        cluster (dict): Information about the cluster, including its class and population
        data (torch_geometric.data.Data): The graph data object containing node features and structure
        class_cluster_profiles (dict): Profiles of all clusters across all classes

    Returns:
        tuple: (best_accuracy, best_formula, best_response)
            - best_accuracy (float): The highest accuracy achieved by any formula
            - best_formula (str): Python code for the best-performing formula
            - best_response (str): Full LLM response containing the best formula with rules and interpretation
    """

    # Gather positive nodes (cluster) and negative nodes (from other classes)
    total_pos_nodes = cluster['population_ids']
    pos_nodes = cluster['population_ids'][:max_cluster_population]
    negative_test_nodes = []
    for c in range(num_classes):
        if c != cluster['class_label']:
            for neg_cluster in class_cluster_profiles[c]:
                negative_test_nodes.extend(neg_cluster['population_ids'])

    total_neg_nodes = negative_test_nodes
    if len(negative_test_nodes) > max_cluster_population:
        negative_test_nodes = npr.choice(
            negative_test_nodes, max_cluster_population, replace=False)

    # runs = 1
    # Maintain the total accuracies over epochs over runs for plotting
    total_accuracies = []

    best_accuracy = 0.0
    best_formula = None
    best_response = None

    # History of iterations: list of tuples (formula, feedback, results)
    history = []

    accuracies = []

    # select batch_size number of nodes
    init_pos_nodes = npr.choice(pos_nodes, min(
        args.batch_size, len(pos_nodes)), replace=False)
    init_neg_nodes = npr.choice(negative_test_nodes, min(
        args.batch_size, len(negative_test_nodes), len(pos_nodes)), replace=False)
    # Step 1: Generate initial formula (y0)
    initial_response = generate_initial_formula(
        cluster, init_pos_nodes, init_neg_nodes, data, None, in_context=args.in_context)
    try:
        current_formula_code = initial_response.split('```python')[
            1].split('```')[0]
    except:
        print("No python code found in the initial formula.")
        current_formula_code = None

    # Evaluate misclassifications
    fnodes, fpnodes, error = get_misclassifications(
        current_formula_code, pos_nodes, negative_test_nodes, data)
    curr_pos_acc = 1.0 - (len(fnodes) / len(pos_nodes)
                          ) if len(pos_nodes) > 0 else 0
    curr_neg_acc = 1.0 - \
        (len(fpnodes) / len(negative_test_nodes)
         ) if len(negative_test_nodes) > 0 else 0
    curr_accuracy = 0.5 * (curr_pos_acc + curr_neg_acc)
    accuracies.append(curr_accuracy)

    best_accuracy = curr_accuracy
    best_formula = current_formula_code
    best_response = initial_response

    print(
        f"Initial formula => pos_acc={curr_pos_acc:.3f}, neg_acc={curr_neg_acc:.3f}, total={curr_accuracy:.3f}")

    pos_acc = curr_pos_acc
    neg_acc = curr_neg_acc
    # Iterative refinement loop with history
    for ep in range(1, args.epochs + 1):
        if best_accuracy >= 0.95:
            print("Stopping early: reached >= 0.95 accuracy.")
            break

        # Generate feedback on current formula
        feedback_text = generate_feedback(
            current_formula_code, fnodes, fpnodes, data, history, error)
        # Append current iteration to history
        history.append(
            (current_formula_code, feedback_text, pos_acc, neg_acc, error))

        # Generate refined formula using history
        refine_resp = refine_formula(
            current_formula_code, feedback_text, fnodes, fpnodes, history, error)
        try:
            new_formula_code = refine_resp.split(
                '```python')[1].split('```')[0]
        except:
            print("No python code found in refined formula.")
            new_formula_code = current_formula_code

        # Evaluate the new formula
        fnodes, fpnodes, error = get_misclassifications(
            new_formula_code, pos_nodes, negative_test_nodes, data)
        pos_acc = 1.0 - (len(fnodes) / len(pos_nodes)
                         ) if len(pos_nodes) > 0 else 0
        neg_acc = 1.0 - (len(fpnodes) / len(negative_test_nodes)
                         ) if len(negative_test_nodes) > 0 else 0
        total_acc = 0.5 * (pos_acc + neg_acc)
        accuracies.append(total_acc)
        # global_acc = calculate_global_accuracy(new_formula_code, total_pos_nodes, total_neg_nodes, data)[2]

        print(
            f"[Epoch {ep}] pos_acc={pos_acc:.3f}, neg_acc={neg_acc:.3f}, total={total_acc:.3f}")
        # print(f"[Epoch {ep}] pos_acc={pos_acc:.3f}, neg_acc={neg_acc:.3f}, total={total_acc:.3f}", f"global_acc={global_acc:.3f}")

        if total_acc > best_accuracy:
            best_accuracy = total_acc
            best_formula = new_formula_code
            best_response = refine_resp

        current_formula_code = new_formula_code

    cluster['accuracy'] = best_accuracy
    cluster['formula'] = best_formula
    cluster['summary'] = best_response
    # if accuracies size is less than epochs+1, fill the rest with the last accuracy
    if len(accuracies) < args.epochs+1:
        accuracies.extend([accuracies[-1]] *
                          (args.epochs+1 - len(accuracies)))
    total_accuracies.append(accuracies)

    return best_accuracy, best_formula, best_response


def summarise_clusters(class_cluster_profiles, data):
    """
    Trains and summarizes all clusters across all classes.

    This function iterates through all classes and clusters, training an LLM to generate
    symbolic rules for each cluster. It updates the class_cluster_profiles with the
    generated formulas, accuracy scores, and summaries.

    Parameters:
        class_cluster_profiles (dict): Profiles of all clusters for all classes
        data (torch_geometric.data.Data): The graph data object containing node features and structure

    Returns:
        dict: Updated class_cluster_profiles with trained formulas and summary information
    """

    for i in range(num_classes):
        for j, cluster in enumerate(class_cluster_profiles[i]):
            print(
                f"=== Class {i}, concept_node={cluster['cluster_id']} ===")
            best_accuracy, best_formula, best_response = train_cluster_LLM_self_refine(
                cluster, data, class_cluster_profiles)
            class_cluster_profiles[i][j]['accuracy'] = best_accuracy
            class_cluster_profiles[i][j]['formula'] = best_formula
            class_cluster_profiles[i][j]['summary'] = best_response
    return class_cluster_profiles


def train_llm():
    """
    Main function for training the LLM to generate explanations for all clusters.

    This function orchestrates the complete training pipeline for generating symbolic rule-based
    explanations for all clusters across all classes. It:
    1. Generates class/cluster profiles
    2. Trains the LLM for each cluster to generate symbolic rules
    3. Saves the results to disk (JSON for profiles, text file for summaries)

    The LLM training process involves generating initial rules and then iteratively refining
    them based on their performance on nodes from the cluster and outside the cluster.

    Returns:
        None
    """
    class_cluster_profiles = generate_class_cluster_profiles(data)
    os.makedirs(f'data/{args.dataset}', exist_ok=True)
    # with open(f'data/{dataset}/class_cluster_profiles.json', 'w') as f:
    #     json.dump(class_cluster_profiles, f)
    class_cluster_profiles = summarise_clusters(
        class_cluster_profiles, data)
    print("Summarised clusters")

    with open(f'data/{args.dataset}/class_cluster_profiles.json', 'w') as f:
        json.dump(class_cluster_profiles, f, default=numpy_encoder)

    with open(f'data/{args.dataset}/cluster_summaries.txt', 'w') as f:
        for i in range(num_classes):
            for j, cluster in enumerate(class_cluster_profiles[i]):
                # for j, cluster in enumerate(class_cluster_profiles[str(i)]):
                f.write(
                    f"Class {i} Cluster {cluster['cluster_id']} Summary:\n")
                summary = cluster.get('summary', None)
                if summary:
                    f.write(summary)
                else:
                    f.write("No summary available.\n")
                f.write("\nAccuracy: " +
                        str(cluster.get('accuracy', 0.0)) + "\n\n")


def gen_concept_vectors():
    """
    Generates concept vectors for all test nodes using the trained formulas.

    This function loads the trained formulas for each concept node and applies them to the
    test nodes to generate binary concept vectors. A concept vector represents whether each
    test node belongs to the concept cluster (1) or not (0) according to the symbolic rule.

    The concept vectors are saved to disk as a numpy array for later use in explanation and
    evaluation.

    Returns:
        None
    """

    with open(f'data/{args.dataset}/class_cluster_profiles.json', 'r') as f:
        class_cluster_profiles = json.load(f)

    for i in range(num_classes):
        # for j, cluster in enumerate(class_cluster_profiles[i]):
        for j, cluster in enumerate(class_cluster_profiles[str(i)]):
            print(
                f"Class {i} Cluster {cluster['cluster_id']} => Accuracy: {cluster.get('accuracy', 0.0)}")

    cluster_formulae = {}
    for i in range(num_classes):
        for j, cluster in enumerate(class_cluster_profiles[str(i)]):
            cid = cluster['cluster_id']
            cluster_formulae[cid] = cluster['formula']

    concept_vectors = np.zeros((len(data.x), len(chosen_nodes)))
    # for i_node in range(min(max_test_nodes, len(data.x))):
    for i_node in test_mask:
        # print(f"Processing node {i_node}")
        for j_c, concept_node in enumerate(chosen_nodes):
            python_function = cluster_formulae.get(concept_node, None)
            if python_function is None:
                continue
            local_namespace = {}
            try:
                exec(python_function, globals(), local_namespace)
            except:
                # print(f"Error executing python formula for concept_node={concept_node}")
                continue
            if 'classify_node' not in local_namespace:
                print(
                    f"Function classify_node not found in the formula for concept_node={concept_node}")
                continue
            classify_node_fn = local_namespace['classify_node']
            try:
                concept_vectors[i_node][j_c] = classify_node_fn(
                    generate_node_description(i_node, data))
            except:
                # print(f"Error executing classify_node for node {i_node} with concept_node={concept_node}")
                concept_vectors[i_node][j_c] = 0

    np.save(f'data/{args.dataset}/concept_vectors.npy', concept_vectors)
    print("Saved concept_vectors.npy")
    # save the test_mask
    np.save(f'data/{args.dataset}/test_mask.npy', test_mask)
    print("Saved test_mask.npy")


# remove the results file if it exists
if os.path.exists(f'data/{args.dataset}/results.txt'):
    os.remove(f'data/{args.dataset}/results.txt')

runs_results = []

if args.phase == 1:
    train_llm()
    gen_concept_vectors()

if args.phase in (1, 2):

    def get_pos_neg_points(test_indices, gnn_pred, target_class, sample_size=50, seed=1234):
        """
        Samples positive and negative nodes for a target class.

        For a given target class, this function samples a balanced set of positive nodes 
        (belonging to the target class) and negative nodes (belonging to other classes).
        It uses stratified sampling for negative nodes to ensure class balance.

        Parameters:
            test_indices (array-like): Indices of test nodes to sample from
            gnn_pred (array-like): GNN predicted classes for all nodes
            target_class (int): The target class for which to sample nodes
            sample_size (int): Maximum number of positive and negative samples to return
            seed (int): Random seed for reproducibility

        Returns:
            tuple: (pos_sample, neg_sample)
                - pos_sample (ndarray): Indices of positive nodes
                - neg_sample (ndarray): Indices of negative nodes
                Returns (None, None) if either positive or negative samples cannot be found
        """
        from sklearn.model_selection import train_test_split

        if type(gnn_pred) == torch.tensor:
            gnn_pred = gnn_pred.cpu().numpy()
        elif type(gnn_pred) != np.ndarray:
            gnn_pred = np.array(gnn_pred)
        # Ensure gnn_pred is a numpy array

        pos_indices = np.array(
            [i for i in test_indices if int(gnn_pred[i]) == target_class])
        neg_indices = np.array(
            [i for i in test_indices if int(gnn_pred[i]) != target_class])

        num_pos = len(pos_indices)
        num_neg = len(neg_indices)

        if num_pos == 0 or num_neg == 0:
            print(
                f"Not enough samples for class {target_class}: pos={num_pos}, neg={num_neg}")
            return None, None

        k = min([sample_size, num_pos, num_neg])
        # check if k is less than number of classes, then select randomly from the positive and negative indices
        if k < num_classes:
            pos_sample = np.random.choice(
                pos_indices, k, replace=False)
            neg_sample = np.random.choice(
                neg_indices, k, replace=False)
        else:

            # Sample k positive nodes
            if num_pos == k:
                pos_sample = pos_indices
            else:
                _, pos_sample = train_test_split(
                    pos_indices, test_size=k, random_state=seed)

            # Sample k negative nodes, stratified sampling based on gnn_pred values
            if num_neg == k:
                neg_sample = neg_indices
            else:
                _, neg_sample = train_test_split(
                    neg_indices, test_size=k, random_state=seed, stratify=gnn_pred[neg_indices])

        return pos_sample, neg_sample

    # JUST TAKES OR of ALL FORMULAE IN THAT CLASS

    def calculate_explainer_accuracy_for_class(concept_vectors, chosen_nodes, gnn_pred, target_class, num_samples=100, test_mask=None, seed=1234):
        """
        Calculates the accuracy of concept-based explanations for a target class.

        This function evaluates how well the concept vectors can explain membership in a target class.
        For nodes predicted to be in the target class, it checks if any concept from the target class
        includes the node (a concept vector value of 1). It then computes accuracy metrics comparing
        these concept-based predictions to the GNN's predictions.

        Parameters:
            concept_vectors (np.ndarray): Binary matrix [num_nodes, num_chosen_nodes] indicating 
                                        whether each node belongs to each concept
            chosen_nodes (list): List of node IDs used as concept prototypes
            gnn_pred (np.ndarray): GNN predicted classes for all nodes
            target_class (int): The class for which to evaluate explanation accuracy
            num_samples (int): Number of positive and negative test nodes to sample
            test_mask (array-like, optional): Indices of test nodes to consider
            seed (int): Random seed for reproducibility

        Returns:
            tuple: (classification_report_dict, accuracy)
                - classification_report_dict (dict): Precision, recall, F1 metrics for the target class
                - accuracy (float): Overall accuracy of the concept-based classifier
                Returns (None, None) if samples cannot be obtained
        """
        import numpy as np
        from sklearn.metrics import classification_report

        print(
            f"Calculating explainer accuracy for class {target_class}")

        # Determine test indices.
        if test_mask is not None:
            if hasattr(test_mask, "cpu"):
                test_mask = test_mask.cpu().numpy()
            # test_indices = np.where(test_mask)[0]
            test_indices = test_mask
        else:
            test_indices = np.arange(gnn_pred.shape[0])
        # print("Number of test nodes: ", len(test_indices))

        # # Identify positive test indices (where the true label equals target_class)
        # pos_indices = np.array([i for i in test_indices if int(gnn_pred[i]) == target_class])
        # # Identify negative test indices (where the true label is not target_class)
        # neg_indices = np.array([i for i in test_indices if int(gnn_pred[i]) != target_class])

        # # Sample num_samples from each group
        # num_pos = min(num_samples, len(pos_indices))
        # num_neg = min(num_samples, len(neg_indices))
        # min_num_samples = min(num_pos, num_neg)
        # num_pos = min_num_samples
        # num_neg = min_num_samples
        # print(f"Sampling {num_pos} positive and {num_neg} negative test nodes for class {target_class}")
        # if num_pos == 0 or num_neg == 0:
        #     print(f"Not enough samples for class {target_class}: pos={len(pos_indices)}, neg={len(neg_indices)}")
        #     return None, None

        pos_sample, neg_sample = get_pos_neg_points(
            test_indices, gnn_pred, target_class, sample_size=num_samples, seed=seed)
        if pos_sample is None or neg_sample is None:
            return None, None

        # pos_sample = np.random.choice(pos_indices, num_pos, replace=False)
        # neg_sample = np.random.choice(neg_indices, num_neg, replace=False)
        sample_indices = np.concatenate([pos_sample, neg_sample])

        # Ground truth: for each sampled node, 1 if its true class equals target_class, else 0.
        y_true = np.array(
            [1 if int(gnn_pred[i]) == target_class else 0 for i in sample_indices])

        predictions = []
        # For each sampled node i, apply the global explainer rule for target_class:
        # 1. Among the chosen_nodes, find those indices j where gnn_pred[chosen_nodes[j]] == target_class.
        # 2. If any such j has concept_vectors[i][j] == 1, predict 1; otherwise predict 0.
        valid_indices = [j for j, cn in enumerate(
            chosen_nodes) if int(gnn_pred[cn]) == target_class]
        for ind in sample_indices:

            if valid_indices and any(int(concept_vectors[ind][j]) == 1 for j in valid_indices):
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.array(predictions)
        # print("Predictions: ", predictions)
        # accuracy = np.mean(predictions == y_true)

        accuracy = classification_report(y_true, predictions, target_names=[
            f"Not {target_class}", f"{target_class}"], output_dict=True)['accuracy']
        # return classification_report(y_true, predictions, target_names=[f"Not {target_class}", f"{target_class}"], output_dict=True)['weighted avg'], accuracy
        return classification_report(y_true, predictions, target_names=[f"Not {target_class}", f"{target_class}"], output_dict=True), accuracy
        # return accuracy

    # concept_vectors = np.load(f'data/{dataset}/knn_concept_vectors.npy')
    concept_vectors = np.load(f'data/{args.dataset}/concept_vectors.npy')

    if os.path.exists(f'data/{args.dataset}/test_mask.npy'):
        test_mask = np.load(f'data/{args.dataset}/test_mask.npy')

    class_accuracies = []
    fidelities = []
    for i in range(num_classes):
        class_report, fidelity = calculate_explainer_accuracy_for_class(
            concept_vectors, chosen_nodes, gnn_pred, i, num_samples=200, test_mask=test_mask)

        # class_report is a map containing keys like 'i' and 'Not i'
        # example {'Not 0': {'precision': 0.6014492753623188, 'recall': 0.83, 'f1-score': 0.6974789915966386, 'support': 200.0}, '0': {'precision': 0.7258064516129032, 'recall': 0.45, 'f1-score': 0.5555555555555556, 'support': 200.0}, 'accuracy': 0.64, 'macro avg': {'precision': 0.663627863487611, 'recall': 0.64, 'f1-score': 0.6265172735760971, 'support': 400.0}, 'weighted avg': {'precision': 0.663627863487611, 'recall': 0.64, 'f1-score': 0.6265172735760971, 'support': 400.0}}
        # extract the precision, recall, f1-score for the positive class
        if class_report is not None:
            class_report = class_report.get(str(i), None)
            class_accuracies.append(class_report)
            fidelities.append(fidelity)
    class_accuracies = np.array(class_accuracies)
    print("Class Accuracies: ", class_accuracies)
    print("Mean of the class precision, recall, f1-score:")
    # take the mean of the precision, recall, f1-score of the positive class
    print("Precision: ", np.mean(
        [d['precision'] for d in class_accuracies]))
    print("Recall: ", np.mean([d['recall'] for d in class_accuracies]))
    print("F1-Score: ", np.mean([d['f1-score']
                                 for d in class_accuracies]))
    print("Fidelity: ", np.mean(fidelities))
    # save the results to a file
    with open(f'data/{args.dataset}/results.txt', 'a') as f:
        f.write("Mean of the class precision, recall, f1-score:\n")
        f.write(
            "Precision: " + str(np.mean([d['precision'] for d in class_accuracies])) + "\n")
        f.write(
            "Recall: " + str(np.mean([d['recall'] for d in class_accuracies])) + "\n")
        f.write(
            "F1-Score: " + str(np.mean([d['f1-score'] for d in class_accuracies])) + "\n")
        f.write("Fidelity: " + str(np.mean(fidelities)) + "\n\n")

    run_results = {
        'precision_mean': np.mean([d['precision'] for d in class_accuracies]),
        'recall_mean': np.mean([d['recall'] for d in class_accuracies]),
        'f1_score_mean': np.mean([d['f1-score'] for d in class_accuracies]),
        'fidelity_mean': np.mean(fidelities),
    }
    runs_results.append(run_results)


# * Step C: Generate Global Explanations

if args.phase in (1, 2, 3):

    # Combine all the explanations from different clusters of a class to form the global explanation
    def combine_explanations(class_cluster_profiles):
        """
        Combines cluster explanations to generate global explanations for each class.

        This function generates human-interpretable textual explanations for each class
        by combining the symbolic rules and interpretations from all clusters within that class.
        It uses an LLM to synthesize these explanations into coherent class-level descriptions.

        Parameters:
            class_cluster_profiles (dict): Profiles of all clusters for all classes, including
                                        their symbolic rules and interpretations

        Returns:
            dict: Dictionary mapping class indices to their global explanations
        """
        global_explanations = {}
        for i in range(num_classes):
            global_explanations[i] = []
            # generate the prompt for the LLM to combine the formulae for all clusters within a class to produce human interpretable textual explanations for each class
            prompt = (
                "You are given a set of clusters for a class, each with a symbolic rule-based formula that outputs 1 for cluster nodes and 0 for non-cluster nodes. "
                "Please combine these formulas to generate a human-interpretable textual explanation for the class."
                "The explanation should tell a human what belonging to this class means, basically what characteristics of a node lead it to be classified as belonging to the class. "
                # "The explanation should be a single sentence or a short paragraph that combines the information from all the clusters. The explanation should not include any graph based terminology, the key idea is that a lay man should be able to understand the explanations, so it should consist of simple words. "
                "The explanation should be a single sentence or a short paragraph that combines the information from all the clusters. The key idea is that a lay man should be able to understand the explanations, so it should consist of simple words. "
                "You can use the cluster formulae, the class description, and the dataset description to form the explanation. "
            )

            # add the dataset description
            prompt += "The dataset description is as follows:\n" + dataset_description + "\n"
            # add class info to the prompt
            prompt += f"Class {i}: {class_description[str(i)]}\n"
            # add cluster info to the prompt
            for j, cluster in enumerate(class_cluster_profiles[str(i)]):
                prompt += f"Cluster {j}: {cluster['summary']}\n"
            if mode == 'dense':
                # add the meaning of features for the dense case
                prompt += "Since the node features are dense embeddings for this dataset, we have replaced the features vector by their dot product with the feature vector of each of the concept node. Basically a higher value of the feature means higher similarity to the exemplar at that index. So in your explanations you must use terms like - having features highly similar to exemplar node x of class y (instead of saying feature value > 3). \n"
                # add the list of concepts along with their classes
                prompt += "The exemplar nodes (in order) are as follows:\n"
                for j, concept_node in enumerate(chosen_nodes):
                    prompt += f"Exemplar node {j}: {concept_node} (Class {gnn_pred[concept_node]}), "

            response = get_response(prompt, temperature=args.temperature)
            global_explanations[i] = response

        return global_explanations

    class_cluster_profiles = json.load(
        open(f'data/{args.dataset}/class_cluster_profiles.json'))
    combined_explanations = combine_explanations(
        class_cluster_profiles)
    # save the combined explanations into a txt file
    with open(f'data/{args.dataset}/combined_explanations.txt', 'w') as f:
        for i in range(num_classes):
            f.write(f"Class {i}: {class_description[str(i)]}\n")
            f.write(f"Class {i}:\n")
            f.write(combined_explanations[i])
            f.write("\n\n")

# compute the mean and std of precision, recall, f1 score and fidelity for all runs
with open(f'data/{args.dataset}/metrics.txt', 'a') as f:
    f.write("Mean and Std of the class precision, recall, f1-score:\n")
    f.write("Precision: " + "Mean: " + str(np.mean([d['precision_mean'] for d in runs_results])) + "\n")
    f.write("Recall: " + "Mean: " + str(np.mean([d['recall_mean'] for d in runs_results])) + "\n")
    f.write("F1-Score: " + "Mean: " + str(np.mean([d['f1_score_mean'] for d in runs_results])) + "\n")
    f.write("Fidelity: " + "Mean: " + str(np.mean([d['fidelity_mean'] for d in runs_results])) + "\n")
    f.write("\n\n")
    f.write("=== Run Results ===\n")
