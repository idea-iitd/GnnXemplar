from torch_geometric.nn import GCNConv, SAGEConv
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as pyg
from torch.nn import init
from torch_geometric.nn import GCNConv


class CORA_GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) implementation for the CORA dataset.
    
    Implementation based on the architecture described in the GAT paper:
    https://github.com/PetarV-/GAT/tree/master
    
    Args:
        in_channels (int): Number of input features.
        num_classes (int): Number of output classes.
    
    Attributes:
        explaining (bool): Flag for explanation mode.
        lr (float): Learning rate for optimization.
        weight_decay (float): L2 regularization parameter.
        gat1 (pyg.nn.GATConv): First GAT layer with 8 attention heads.
        gat2 (pyg.nn.GATConv): Second GAT layer with 1 attention head.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.explaining = False
        self.lr = 0.005
        self.weight_decay = 0.0005  # L2 regularization
        self.gat1 = pyg.nn.GATConv(
            in_channels=in_channels,
            out_channels=8,
            heads=8,
            dropout=0.6,
        )
        self.gat2 = pyg.nn.GATConv(
            in_channels=64,  # out_channels * heads = 8 * 8
            out_channels=num_classes,
            heads=1,
            dropout=0.6,
        )

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass for the GAT model.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            **kwargs: Additional arguments.
            
        Returns:
            tuple or torch.Tensor: When explaining=False, returns (out, node_embeddings) where:
                - out (torch.Tensor): Softmax probabilities for node classification.
                - node_embeddings (torch.Tensor): Node embeddings from the final layer.
            When explaining=True, returns only the output probabilities.
        """
        x = self.gat1(x, edge_index)
        x = torch.nn.functional.elu(x)
        node_embeddings = self.gat2(x, edge_index)
        out = torch.nn.functional.softmax(node_embeddings, dim=-1)
        if self.explaining:
            return out
        else:
            return out, node_embeddings


# * >>> GNNExplainer's GCN
"""Replicated from: https://github.com/RexYing/gnn-model-explainer"""


class GraphConv(torch.nn.Module):
    """
    Graph Convolution layer implementation for GNNExplainer.
    
    This is a customized graph convolution layer that supports additional features
    like self-loops, attention, and normalization.
    
    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output features.
        add_self (bool, optional): Whether to add self-loops. Defaults to False.
        normalize_embedding (bool, optional): Whether to normalize embeddings. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        bias (bool, optional): Whether to use bias. Defaults to True.
        gpu (bool, optional): Whether to use GPU. Defaults to True.
        att (bool, optional): Whether to use attention. Defaults to False.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        add_self=False,
        normalize_embedding=False,
        dropout=0.0,
        bias=True,
        gpu=True,
        att=False,
    ):
        super(GraphConv, self).__init__()
        self.att = att
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = torch.nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not gpu:
            self.weight = torch.nn.Parameter(
                torch.FloatTensor(input_dim, output_dim))
            if add_self:
                self.self_weight = torch.nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim)
                )
            if att:
                self.att_weight = torch.nn.Parameter(
                    torch.FloatTensor(input_dim, input_dim))
        else:
            self.weight = torch.nn.Parameter(
                torch.FloatTensor(input_dim, output_dim).cuda())
            if add_self:
                self.self_weight = torch.nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim).cuda()
                )
            if att:
                self.att_weight = torch.nn.Parameter(
                    torch.FloatTensor(input_dim, input_dim).cuda()
                )
        if bias:
            if not gpu:
                self.bias = torch.nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.bias = torch.nn.Parameter(
                    torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        """
        Forward pass of the graph convolution layer.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [batch_size, num_nodes, input_dim].
            adj (torch.Tensor): Adjacency matrix with shape [batch_size, num_nodes, num_nodes].
            
        Returns:
            tuple: Contains:
                - y (torch.Tensor): Updated node features with shape [batch_size, num_nodes, output_dim].
                - adj (torch.Tensor): Adjacency matrix, potentially modified with attention weights.
        """
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        # deg = torch.sum(adj, -1, keepdim=True)
        if self.att:
            x_att = torch.matmul(x, self.att_weight)
            # import pdb
            # pdb.set_trace()
            att = x_att @ x_att.permute(0, 2, 1)
            # att = self.softmax(att)
            adj = adj * att

        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.add_self:
            self_emb = torch.matmul(x, self.self_weight)
            y += self_emb
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y, adj


class GcnEncoderGraph(nn.Module):
    """
    Graph Convolutional Network (GCN) encoder for graph-level tasks.
    
    This model implements a multi-layer GCN with options for batch normalization,
    concatenation of layer outputs, and dropout.
    
    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        embedding_dim (int): Output embedding dimension.
        label_dim (int): Number of output classes.
        num_layers (int): Number of GCN layers.
        pred_hidden_dims (list, optional): Dimensions of MLP prediction layers. Defaults to [].
        concat (bool, optional): Whether to concatenate outputs from all layers. Defaults to True.
        bn (bool, optional): Whether to use batch normalization. Defaults to True.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        add_self (bool, optional): Whether to add self-loops. Defaults to False.
        args (object, optional): Additional arguments. Defaults to None.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers,
        pred_hidden_dims=[],
        concat=True,
        bn=True,
        dropout=0.0,
        add_self=False,
        args=None,
    ):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = add_self
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        # self.gpu = args.gpu
        self.gpu = False
        self.att = False
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            add_self,
            normalize=True,
            dropout=dropout,
        )
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs
        )

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.att:
                    init.xavier_uniform_(
                        m.att_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.add_self:
                    init.xavier_uniform_(
                        m.self_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_layers,
        add_self,
        normalize=False,
        dropout=0.0,
    ):
        """
        Builds the graph convolutional layers of the network.
        
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            embedding_dim (int): Output embedding dimension.
            num_layers (int): Number of GCN layers.
            add_self (bool): Whether to add self-loops.
            normalize (bool, optional): Whether to normalize embeddings. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            
        Returns:
            tuple: Contains:
                - conv_first (GraphConv): First graph convolution layer.
                - conv_block (nn.ModuleList): Middle graph convolution layers.
                - conv_last (GraphConv): Last graph convolution layer.
        """
        conv_first = GraphConv(
            input_dim=input_dim,
            output_dim=hidden_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        conv_block = nn.ModuleList(
            [
                GraphConv(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    add_self=add_self,
                    normalize_embedding=normalize,
                    dropout=dropout,
                    bias=self.bias,
                    gpu=self.gpu,
                    att=self.att,
                )
                for i in range(num_layers - 2)
            ]
        )
        conv_last = GraphConv(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        return conv_first, conv_block, conv_last

    def build_pred_layers(
        self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1
    ):
        """
        Builds the prediction layers (MLP) of the network.
        
        Args:
            pred_input_dim (int): Input dimension for prediction layer.
            pred_hidden_dims (list): List of hidden dimensions for prediction layers.
            label_dim (int): Number of output classes.
            num_aggs (int, optional): Number of aggregations. Defaults to 1.
            
        Returns:
            nn.Module: Prediction model (Linear layer or MLP).
        """
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        """ 
        Constructs a mask for batched graph operations.
        
        For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        
        Args:
            max_nodes (int): Maximum number of nodes across all graphs in the batch.
            batch_num_nodes (list): List of node counts for each graph in the batch.
            
        Returns:
            torch.Tensor: Mask tensor with dimension [batch_size x max_nodes x 1]
        """
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, : batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        """ 
        Applies batch normalization to a 3D tensor.
        
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_nodes, feature_dim].
            
        Returns:
            torch.Tensor: Batch normalized tensor.
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

    def gcn_forward(
        self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None
    ):
        """ 
        Performs forward propagation with graph convolution.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [batch_size, num_nodes, input_dim].
            adj (torch.Tensor): Adjacency matrix with shape [batch_size, num_nodes, num_nodes].
            conv_first (GraphConv): First graph convolution layer.
            conv_block (nn.ModuleList): Middle graph convolution layers.
            conv_last (GraphConv): Last graph convolution layer.
            embedding_mask (torch.Tensor, optional): Mask for embeddings. Defaults to None.
            
        Returns:
            tuple: Contains:
                - x_tensor (torch.Tensor): Embedding matrix with dimension [batch_size, num_nodes, embedding_dim].
                - adj_att_tensor (torch.Tensor): Attention tensor with dimension [batch_size, num_nodes, num_nodes, num_gc_layers].
        """
        x, adj_att = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        adj_att_all = [adj_att]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x, _ = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
            adj_att_all.append(adj_att)
        x, adj_att = conv_last(x, adj)
        x_all.append(x)
        adj_att_all.append(adj_att)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)
        return x_tensor, adj_att_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        """
        Forward pass for the GCN encoder.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [batch_size, num_nodes, input_dim].
            adj (torch.Tensor): Adjacency matrix with shape [batch_size, num_nodes, num_nodes].
            batch_num_nodes (list, optional): List of node counts for each graph in batch. Defaults to None.
            **kwargs: Additional arguments.
            
        Returns:
            tuple: Contains:
                - ypred (torch.Tensor): Predicted class scores.
                - adj_att_tensor (torch.Tensor): Attention tensor.
        """
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(
                max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x, adj_att = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        adj_att_all = [adj_att]
        for i in range(self.num_layers - 2):
            x, adj_att = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
            adj_att_all.append(adj_att)
        x, adj_att = self.conv_last(x, adj)
        adj_att_all.append(adj_att)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)

        self.embedding_tensor = output
        ypred = self.pred_model(output)
        # print(output.size())
        return ypred, adj_att_tensor

    def loss(self, pred, label, type="softmax"):
        """
        Computes loss for the model prediction.
        
        Args:
            pred (torch.Tensor): Predicted class scores.
            label (torch.Tensor): Ground truth labels.
            type (str, optional): Loss type, either "softmax" or "margin". Defaults to "softmax".
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        # softmax + CE
        if type == "softmax":
            return F.cross_entropy(pred, label, size_average=True)
        elif type == "margin":
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(
                batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class Gnnexplainer_GCN(GcnEncoderGraph):
    """
    GCN implementation compatible with GNNExplainer.
    
    This extends the GcnEncoderGraph with additional functionality to support
    the GNNExplainer framework.
    
    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        embedding_dim (int): Output embedding dimension.
        label_dim (int): Number of output classes.
        num_layers (int): Number of GCN layers.
        pred_hidden_dims (list, optional): Dimensions of MLP prediction layers. Defaults to [].
        concat (bool, optional): Whether to concatenate outputs from all layers. Defaults to True.
        bn (bool, optional): Whether to use batch normalization. Defaults to True.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        args (object, optional): Additional arguments. Defaults to None.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers,
        pred_hidden_dims=[],
        concat=True,
        bn=True,
        dropout=0.0,
        args=None
    ):
        super(Gnnexplainer_GCN, self).__init__(
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims,
            concat,
            bn,
            dropout,
            args=args,
        )
        self.explaining = False
        if hasattr(args, "loss_weight"):
            print("Loss weight: ", args.loss_weight)
            self.celoss = torch.nn.CrossEntropyLoss(weight=args.loss_weight)
        else:
            self.celoss = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch_num_nodes=None, **kwargs):
        """
        Forward pass for the Gnnexplainer_GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            batch_num_nodes (list, optional): List of node counts for each graph in batch. Defaults to None.
            **kwargs: Additional arguments.
            
        Returns:
            tuple or torch.Tensor: When explaining=False, returns (out, embedding_tensor).
            When explaining=True, returns only the output scores.
        """
        # mask
        adj = pyg.utils.to_dense_adj(edge_index)
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(
                max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.adj_atts = []
        self.embedding_tensor, adj_att = self.gcn_forward(
            x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask
        )

        out = self.pred_model(self.embedding_tensor)
        # get kwargs
        if self.explaining:
            return out.squeeze(0)
        else:
            return out, self.embedding_tensor

    def loss(self, pred, label):
        """
        Computes the cross-entropy loss for the model prediction.
        
        Args:
            pred (torch.Tensor): Predicted class scores.
            label (torch.Tensor): Ground truth labels.
            
        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
        pred = torch.transpose(pred, 1, 2)
        return self.celoss(pred, label)


class KarateClub_GCN(torch.nn.Module):
    """
    GCN implementation for the Zachary's Karate Club dataset.
    
    This model implements a 3-layer GCN with tanh activation, followed by an MLP.
    
    Attributes:
        explaining (bool): Flag for explanation mode.
        lr (float): Learning rate for optimization.
        gcn1, gcn2, gcn3 (pyg.nn.GCNConv): GCN layers.
        mlp (torch.nn.Linear): Final classification layer.
    """
    def __init__(self):
        super().__init__()
        self.explaining = False
        self.lr = 0.01

        self.gcn1 = pyg.nn.GCNConv(34, 4)
        self.gcn2 = pyg.nn.GCNConv(4, 4)
        self.gcn3 = pyg.nn.GCNConv(4, 2)

        self.mlp = torch.nn.Linear(2, 4)

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass for the KarateClub_GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            **kwargs: Additional arguments.
            
        Returns:
            tuple or torch.Tensor: When explaining=False, returns (out, node_embeddings).
            When explaining=True, returns only the output probabilities.
        """
        x = self.gcn1(x, edge_index)
        x = torch.tanh(x)
        x = self.gcn2(x, edge_index)
        x = torch.tanh(x)
        x = self.gcn3(x, edge_index)
        node_embeddings = torch.tanh(x)

        out = self.mlp(node_embeddings)
        out = torch.nn.functional.softmax(out, dim=-1)

        if self.explaining:
            return out
        else:
            return out, node_embeddings


class KarateClub_GCN(torch.nn.Module):
    """
    Revised GCN implementation for the Zachary's Karate Club dataset.
    
    This model implements a 3-layer GCN with tanh activation, followed by an MLP
    with output dimension matching the number of classes (2).
    
    Attributes:
        explaining (bool): Flag for explanation mode.
        lr (float): Learning rate for optimization.
        gcn1, gcn2, gcn3 (pyg.nn.GCNConv): GCN layers.
        mlp (torch.nn.Linear): Final classification layer.
    """
    def __init__(self):
        super().__init__()
        self.explaining = False
        self.lr = 0.01

        self.gcn1 = pyg.nn.GCNConv(34, 4)
        self.gcn2 = pyg.nn.GCNConv(4, 4)
        self.gcn3 = pyg.nn.GCNConv(4, 2)

        self.mlp = torch.nn.Linear(2, 2)

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass for the KarateClub_GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            **kwargs: Additional arguments.
            
        Returns:
            tuple or torch.Tensor: When explaining=False, returns (out, node_embeddings).
            When explaining=True, returns only the output probabilities.
        """
        x = self.gcn1(x, edge_index)
        x = torch.tanh(x)
        x = self.gcn2(x, edge_index)
        x = torch.tanh(x)
        x = self.gcn3(x, edge_index)
        node_embeddings = torch.tanh(x)

        out = self.mlp(node_embeddings)
        out = torch.nn.functional.softmax(out, dim=-1)

        if self.explaining:
            return out
        else:
            return out, node_embeddings

# File: gnns.py


class WikiCS_GCN(nn.Module):
    """
    A simple 2-layer GCN for WikiCS that returns (out, node_embeddings).
    
    This model implements a 2-layer GCN for node classification on the WikiCS dataset.
    
    Args:
        in_channels (int): Number of input features.
        num_classes (int): Number of output classes.
        hidden_dim (int, optional): Dimension of hidden layers. Defaults to 64.
        
    Returns:
        tuple: Contains:
            - out (torch.Tensor): Logits for each class.
            - node_embeddings (torch.Tensor): The final layer's node representations.
    """

    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super(WikiCS_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        """
        Forward pass for the WikiCS_GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            
        Returns:
            tuple: Contains:
                - out (torch.Tensor): Logits for node classification.
                - node_embeddings (torch.Tensor): Node embeddings from the final GCN layer.
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # The node_embeddings you want to extract
        node_embeddings = x

        # Classification layer
        out = self.linear(x)
        return out, node_embeddings


class Twitch_GCN(nn.Module):
    """
    A simple 2-layer GCN for the Twitch Gamers dataset that returns (out, node_embeddings).
    
    This model implements a 2-layer GCN for node classification on the Twitch Gamers dataset.
    
    Args:
        in_channels (int): Number of input features.
        num_classes (int): Number of output classes.
        hidden_dim (int, optional): Dimension of hidden layers. Defaults to 64.
        
    Returns:
        tuple: Contains:
            - out (torch.Tensor): Logits for each class.
            - node_embeddings (torch.Tensor): The final layer's node representations.
    """

    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super(Twitch_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        """
        Forward pass for the Twitch_GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            
        Returns:
            tuple: Contains:
                - out (torch.Tensor): Logits for node classification.
                - node_embeddings (torch.Tensor): Node embeddings from the final GCN layer.
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Save embeddings for later use
        node_embeddings = x

        # Classification layer
        out = self.linear(x)
        return out, node_embeddings


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

# class Arxiv_GCN(nn.Module):
#     """
#     A simple 2-layer GCN for ogbn-arxiv that returns (out, node_embeddings).
#     - out: Logits for each class.
#     - node_embeddings: The final hidden layer representations.
#     """
#     def __init__(self, in_channels, num_classes, hidden_dim=128):
#         super(Arxiv_GCN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.linear = nn.Linear(hidden_dim, num_classes)

#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()
#         self.linear.reset_parameters()

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         node_embeddings = x
#         out = self.linear(x)
#         return out, node_embeddings


class Arxiv_GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) for the ogbn-arxiv dataset.
    
    This model implements a multi-layer GCN with batch normalization and dropout
    for node classification on the ogbn-arxiv dataset.
    
    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units.
        out_channels (int): Number of output classes.
        num_layers (int, optional): Number of GCN layers. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(Arxiv_GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def reset_parameters(self):
        """
        Resets all learnable parameters of the model.
        This includes convolutional layers and batch normalization layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        """
        Forward pass of the Arxiv_GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels]
            adj_t (torch.Tensor): Graph connectivity in CSR format
            
        Returns:
            tuple: Contains:
                - out (torch.Tensor): Log softmax probabilities for node classification
                - node_embeddings (torch.Tensor): Node embeddings from the penultimate layer
        """
        # Pass through all layers except the final one:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Save node embeddings from the penultimate layer
        node_embeddings = x.clone()
        # Final classification layer
        out = self.convs[-1](x, adj_t)
        out = out.log_softmax(dim=-1)
        return out, node_embeddings


class SAGE(nn.Module):
    """
    GraphSAGE implementation for node classification tasks.
    
    This model implements a multi-layer GraphSAGE with batch normalization and dropout.
    
    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units.
        out_channels (int): Number of output classes.
        num_layers (int): Number of GraphSAGE layers.
        dropout (float): Dropout probability.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        """
        Resets all learnable parameters of the model.
        This includes convolutional layers and batch normalization layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        """
        Forward pass of the GraphSAGE model.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels]
            adj_t (torch.Tensor): Graph connectivity in CSR format
            
        Returns:
            torch.Tensor: Log softmax probabilities for node classification
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class Citeseer_GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) for the Citeseer dataset.
    
    This model implements a multi-layer GCN with batch normalization and dropout
    for node classification on the Citeseer dataset.
    
    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units.
        out_channels (int): Number of output classes.
        num_layers (int, optional): Number of GCN layers. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(Citeseer_GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def reset_parameters(self):
        """
        Resets all learnable parameters of the model.
        This includes convolutional layers and batch normalization layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        """
        Forward pass of the Citeseer_GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels]
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges]
            
        Returns:
            tuple: Contains:
                - out (torch.Tensor): Log softmax probabilities for node classification
                - node_embeddings (torch.Tensor): Node embeddings from the penultimate layer
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Save intermediate representations as node_embeddings
        node_embeddings = x.clone()
        # Final layer for classification
        x = self.convs[-1](x, edge_index)
        out = x.log_softmax(dim=-1)
        return out, node_embeddings


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) implementation for node classification tasks.
    
    This model implements a multi-layer GCN with batch normalization and dropout.
    The model returns both class predictions and node embeddings from the penultimate layer.
    
    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units.
        out_channels (int): Number of output classes.
        num_layers (int, optional): Number of GCN layers. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def reset_parameters(self):
        """
        Resets all learnable parameters of the model.
        This includes convolutional layers and batch normalization layers.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels]
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges]
            
        Returns:
            tuple: Contains:
                - out (torch.Tensor): Log softmax probabilities for node classification with shape [num_nodes, out_channels]
                - node_embeddings (torch.Tensor): Node embeddings from the penultimate layer with shape [num_nodes, hidden_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_embeddings = x.clone()
        x = self.convs[-1](x, edge_index)
        out = x.log_softmax(dim=-1)
        return out, node_embeddings