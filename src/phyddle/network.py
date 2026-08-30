#!/usr/bin/env python
"""
network
========
Defines classes for neural networks, loss functions, and datasets using
PyTorch.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2025, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
#   none

# external imports
import numpy as np
import torch
import torch.nn.functional as func
from torch import nn
from torch_geometric.nn import GraphConv, GCNConv, GATConv, SAGEConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Dataset as Geoset, Data as GeoData, Batch as GeoBatch
from torch_geometric.utils import to_torch_coo_tensor, is_undirected
import torch.nn.functional as F
from torch_scatter import scatter_add
import random

# phyddle imports
# from phyddle import utilities as util

##################################################

# https://github.com/ameliemelo/Phylo_Inference/blob/main/BiSSE/GNN_PhyloPool.py

def get_valid_node_indices(initial_num_nodes):
    num_conv_layers=3
    pooling_factor = 2
    ker_size = 5
    valid_node_count = initial_num_nodes
    for _ in range(num_conv_layers):
        valid_node_count = (valid_node_count - (ker_size//2)-(ker_size//2)) // pooling_factor
    return valid_node_count

def to_dense_batch(x, batch=None, fill_value = 0, max_num_nodes=2000):
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    batch_size = batch[-1].item() + 1
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = torch.as_tensor(fill_value, device = x.device)
    out = out.to(x.dtype).repeat(size)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])
    return out, num_nodes

class GCN_avg(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, use_cuda, dropout=0.01, activation_function = 'relu', extra_layers=False, n_layers = 4):
        super(GCN_avg, self).__init__()
        torch.manual_seed(12345)
        self.extra_layers = extra_layers
        self.n_layers = n_layers
        self.gconv1 = GCNConv(num_node_features, hidden_channels) 
        self.gconv2 = GCNConv(hidden_channels, hidden_channels) 
        self.gconv3 = GCNConv(hidden_channels, hidden_channels) 
        self.gconv4 = GCNConv(hidden_channels, 2*hidden_channels) 
        self.fc1  = torch.nn.Linear(2*hidden_channels, hidden_channels)
        self.fc2  = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_func = activation_function  
        self.TORCH_DEVICE_STR = (
            "cuda"
            if torch.cuda.is_available() and use_cuda
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)

    def get_embedding(self, x, edge_index, batch, pooling_method="mean"):
        x = x.to(self.TORCH_DEVICE)
        edge_index = edge_index.to(self.TORCH_DEVICE)
        batch = batch.to(self.TORCH_DEVICE)
        x = self.gconv1(x, edge_index)
        if self.activation_func == 'relu':
            x = F.relu(x)
        elif self.activation_func == 'leaky_relu':
            x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.gconv2(x, edge_index)
        if self.activation_func == 'relu':
            x = F.relu(x)
        elif self.activation_func == 'leaky_relu':
            x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.gconv3(x, edge_index)
        if self.activation_func == 'relu':
            x = F.relu(x)
        elif self.activation_func == 'leaky_relu':
            x = F.leaky_relu(x)
        x = self.dropout(x)
            #print("extra layers")
        for graph_layer in range(self.n_layers - 2):
            x = self.gconv3(x, edge_index)
            x = F.relu(x) 
            x = self.dropout(x)
        x = self.gconv4(x, edge_index)
        if self.activation_func == 'relu':
            x = F.relu(x)
        elif self.activation_func == 'leaky_relu':
            x = F.leaky_relu(x)
        x = self.dropout(x)
        if pooling_method == "var":
            mean = global_mean_pool(x, batch)
            mean_sq = global_mean_pool(x ** 2, batch)
            var = (mean_sq - mean**2).clamp_min(0)
            x = torch.cat([var], dim=1)#global_mean_pool
        else:
            x = global_mean_pool(x, batch)
        return x
    
    def forward(self, x, edge_index, batch):
        x = self.get_embedding(x, edge_index, batch)
        x = self.fc1(x)
        x = self.fc2(x)
        return  x



class GCN_PhyloPool(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, use_cuda, ker_size=5, n_parts=10, dropout=0.01, activation_function='relu', extra_layers=False, n_layers=3):
        super(GCN_PhyloPool, self).__init__()
        torch.manual_seed(12345)
        self.n_parts = n_parts
        ker_size = ker_size
        self.gconv1 = GCNConv(num_node_features, hidden_channels) 
        self.gconv2 = GCNConv(hidden_channels, hidden_channels) 
        self.gconv3 = GCNConv(hidden_channels, hidden_channels) 
        self.conv1 = nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size=ker_size) 
        self.conv2= nn.Conv1d(2*hidden_channels, 4*hidden_channels, kernel_size=ker_size) 
        self.conv3= nn.Conv1d(4*hidden_channels, 8*hidden_channels, kernel_size=ker_size) 

        self.fc1  = torch.nn.Linear(8*hidden_channels*self.n_parts, out_features=100)
        self.fc2  = torch.nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_func = activation_function 
        self.n_layers = n_layers
        self.TORCH_DEVICE_STR = (
            "cuda"
            if torch.cuda.is_available() and use_cuda
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)
    
    def forward(self, x, edge_index, batch):
        x = x.to(self.TORCH_DEVICE)
        edge_index = edge_index.to(self.TORCH_DEVICE)
        batch = batch.to(self.TORCH_DEVICE)
        batch_size = batch.max().item()+1
        x = self.gconv1(x, edge_index)
        if self.activation_func == 'relu':
            x = F.relu(x)
        elif self.activation_func == 'leaky_relu':
            x = F.leaky_relu(x)
        x = self.dropout(x)
        for i in range(self.n_layers -1): 
            x = self.gconv3(x, edge_index)
            if self.activation_func == 'relu':
                x = F.relu(x)
            elif self.activation_func == 'leaky_relu':
                x = F.leaky_relu(x)        
            x = self.dropout(x)

        x, num_nodes = to_dense_batch(x, batch)
        x_padded = x.permute(0,2,1)
        x_padded = self.conv1(x_padded)
        if self.activation_func == 'relu':
            x_padded = F.relu(x_padded)
        elif self.activation_func == 'leaky_relu':
             x_padded = F.leaky_relu(x_padded)
        x_padded = self.dropout(x_padded) 
        x_padded = F.avg_pool1d(x_padded, kernel_size = 2)
       
        x_padded = self.conv2(x_padded)
        if self.activation_func == 'relu':
            x_padded = F.relu(x_padded)
        elif self.activation_func == 'leaky_relu':
             x_padded = F.leaky_relu(x_padded)        
        x_padded = self.dropout(x_padded) 
        x_padded = F.avg_pool1d(x_padded, kernel_size = 2)


        x_padded = self.conv3(x_padded)
        if self.activation_func == 'relu':
            x_padded = F.relu(x_padded)
        elif self.activation_func == 'leaky_relu':
             x_padded = F.leaky_relu(x_padded)        
        x_padded = self.dropout(x_padded) 
        x_padded = F.avg_pool1d(x_padded, kernel_size = 2)

        valid_nodes = [get_valid_node_indices(n.item()) for n in num_nodes]
        selected_nodes_list = []
        for i, valid in enumerate (valid_nodes):
            valid_indices = torch.arange(valid) # Generate valid node indices
            base_size = valid // self.n_parts # Base size of each part
            remainder = valid % self.n_parts # Number of remaining indices to distribute

            # Calculate the sizes of the parts
            part_sizes = [base_size + 1 if j < remainder else base_size for j in range(self.n_parts)]
            part_means = []
            start_idx = 0

            for part_size in part_sizes:
                end_idx = start_idx + part_size
                part_indices = valid_indices[start_idx:end_idx]
                start_idx = end_idx
                part_mean = x_padded[i, :, part_indices].mean(dim=1)
                part_means.append(part_mean)
            selected_nodes_list.append(torch.stack(part_means))
        
        selected_nodes = torch.stack(selected_nodes_list)
        selected_nodes = selected_nodes.permute(0,2,1)
        selected_nodes_flattened = selected_nodes.reshape(batch_size, -1)


        if self.activation_func == 'relu':
            out = F.relu(self.fc1(selected_nodes_flattened)) 
        elif self.activation_func == 'leaky_relu':
            out = F.leaky_relu(self.fc1(selected_nodes_flattened)) 
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=HvhgQoO8Svw4




class ParameterEstimationNetwork(nn.Module):
    """
    Parameter estimation neural network. This class defines the network
    structure, activation functions, and forward pass behavior for of input
    to predict labels.
    
    Args:
            args (dict): Contains phyddle settings.
    """
    def __init__(self, num_node_features, hidden_size, num_classes, param_cat, phylo_pool, extra_layers, n_layers, graph_conv, dropout, activation_func, args):


        
        # initialize base class
        super(ParameterEstimationNetwork, self).__init__()
        # width for key input/output

        self.param_cat      = param_cat
        self.param_cat_size = dict()
        self.model_type_categ_ffnn = []
        
        self.phylo_pool = phylo_pool
        self.extra_layers = extra_layers
        # collect args
        self.activation_func        = activation_func
        self.use_cuda               = args['use_cuda']
        self.dropout                = dropout
        self.point_ffnn = []
        self.hidden_size = hidden_size

        self.TORCH_DEVICE_STR = (
            "cuda"
            if torch.cuda.is_available() and self.use_cuda
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)

        # build network
        self.phy_std = nn.ModuleList([])
        if (self.phylo_pool):
            self.phy_std = GCN_PhyloPool(num_node_features= num_node_features, hidden_channels = self.hidden_size, num_classes = num_classes, use_cuda = self.use_cuda, dropout=self.dropout, activation_function=self.activation_func, extra_layers = extra_layers, n_layers = n_layers)
        else:
            self.phy_std = GCN_avg(num_node_features = num_node_features, hidden_channels = self.hidden_size, num_classes = num_classes, use_cuda = self.use_cuda, dropout=self.dropout,  activation_function=self.activation_func, extra_layers = extra_layers, n_layers = n_layers)

        # initialize weights for layers
        self._initialize_weights()

        return

    def _initialize_weights(self):
        """Initializes weights for network."""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, GraphConv):
                torch.nn.init.xavier_uniform_(m.lin_rel.weight)
                torch.nn.init.xavier_uniform_(m.lin_root.weight)
                torch.nn.init.constant_(m.lin_rel.bias, 0)
            if isinstance(m, GCNConv):
                torch.nn.init.xavier_uniform_(m.lin.weight)
                torch.nn.init.constant_(m.bias, 0)
        return

    def forward(self, graph_dat): # forward(self, phy_dat, graph_dat, aux_dat):
        """Forward-pass function of input through network to output labels."""
        num_sample = 500 # phy_dat.shape[0]
        graph_dat = graph_dat.to(self.TORCH_DEVICE)

        x_concat = self.phy_std(graph_dat.x, graph_dat.edge_index, graph_dat.batch)
        x_point = torch.empty((num_sample,0), device=self.TORCH_DEVICE)
        x_lower = torch.empty((num_sample,0), device=self.TORCH_DEVICE)
        x_upper = torch.empty((num_sample,0), device=self.TORCH_DEVICE)

        x_point = torch.empty((num_sample,0), device=self.TORCH_DEVICE)
        x_lower = torch.empty((num_sample,0), device=self.TORCH_DEVICE)
        x_upper = torch.empty((num_sample,0), device=self.TORCH_DEVICE)

        x_categ = dict()
        x_categ = x_concat#.to(self.TORCH_DEVICE)
            
        # return loss
        return x_point, x_lower, x_upper, x_categ


class QuantileLoss(nn.Module):
    """
    Quantile loss function. This function uses an asymmetric quantile
    (or "pinball") loss function.
    
    https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
    """
    def __init__(self, alpha):
        """Defines quantile loss function to predict interval that captures
        alpha-% of output predictions."""
        super(QuantileLoss, self).__init__()
        self.alpha = alpha
        return

    def forward(self, predictions, targets):
        """Simple quantile loss function for prediction intervals."""
        err = targets - predictions
        return torch.mean(torch.max(self.alpha*err, (self.alpha-1)*err))


##################################################

class CrossEntropyLoss(nn.Module):
    """
    Quantile loss function. This function uses an asymmetric quantile
    (or "pinball") loss function.
    
    https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
    """
    def __init__(self):
        """Defines quantile loss function to predict interval that captures
        alpha-% of output predictions."""
        super(CrossEntropyLoss, self).__init__()
        return

    def forward(self, predictions, targets):
        """Simple quantile loss function for prediction intervals."""
        
        loss_list = []
        targets = targets.flatten().long() #.long()
        predictions = predictions.float()
        # weights = 1 / torch.bincount(targets).float()
        # weight_tensor = targets.clone().float()
        
        # if len(weight_tensor) < 2 or min(weight_tensor) == 0: #< 4
        loss_func = torch.nn.CrossEntropyLoss(reduction = 'mean')
        # else:
        #     weight_tensor[weight_tensor == 0] = weights[0]
        #     weight_tensor[weight_tensor == 1] = weights[1]            
        #     weight_tensor = weight_tensor.flatten().unsqueeze(1)   
        #     loss_func = torch.nn.CrossEntropyLoss(reduction = 'mean', weight=weights)
        targets  = targets.unsqueeze(1)
        loss = loss_func(predictions, targets.flatten())
        loss_list = [loss]  
        return torch.sum(torch.stack(loss_list))

