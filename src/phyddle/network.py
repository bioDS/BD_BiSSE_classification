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
from torch_geometric.nn import GraphConv, GCNConv, SAGEConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Dataset as Geoset, Data as GeoData, Batch as GeoBatch
from torch_geometric.utils import to_torch_coo_tensor
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

class GCN_PhyloPool(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, use_cuda):
        super(GCN_PhyloPool, self).__init__()
        torch.manual_seed(12345)
        self.n_parts = 10
        ker_size = 5
        self.gconv1 = GraphConv(num_node_features, hidden_channels) #GCNConv
        self.gconv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv1 = nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size=ker_size)
        self.conv2= nn.Conv1d(2*hidden_channels, 4*hidden_channels, kernel_size=ker_size)
        self.conv3= nn.Conv1d(4*hidden_channels, 8*hidden_channels, kernel_size=ker_size)
        # self.message_passing_layers = nn.ModuleList()
        # self.message_passing_layers.append(self.gconv1)
        # self.message_passing_layers.append(self.gconv2)
        # self.message_passing_layers.append(self.gconv3)

        # self.bn1 = BatchNorm(hidden_channels)
        # self.bn2 = BatchNorm(hidden_channels)
        # self.bn3 = BatchNorm(hidden_channels)
        # self.bn4 = BatchNorm(2*hidden_channels)

        self.lin1 = nn.Linear(8*hidden_channels*self.n_parts, out_features = 100)
        self.lin2 = nn.Linear(100, num_classes)

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
        batch_size = batch.max().item() + 1
        x = self.gconv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.01, training=self.training) # p = 0.01
        x = self.gconv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.01, training=self.training) # p = 0.01

        x, num_nodes = to_dense_batch(x, batch)
        x_padded = x.permute(0,2,1)
        x_padded = self.conv1(x_padded)
        x_padded = F.relu(x_padded)
        x_padded = F.dropout(x_padded, p=0.01, training=self.training) # p = 0.01
        x_padded = F.avg_pool1d(x_padded, kernel_size = 2)
        
        x_padded = self.conv2(x_padded)
        x_padded = F.relu(x_padded)
        x_padded = F.dropout(x_padded, p=0.01, training=self.training) # p = 0.01
        x_padded = F.avg_pool1d(x_padded, kernel_size = 2)

        x_padded = self.conv3(x_padded)
        x_padded = F.relu(x_padded) 
        x_padded = F.dropout(x_padded, p=0.01, training=self.training)   # p = 0.01
        x_padded = F.avg_pool1d(x_padded, kernel_size = 2)
     


        x = F.dropout(x, p=0.01, training=self.training) # p = 0.01
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
        out = F.relu(self.lin1(selected_nodes_flattened))
        out = F.dropout(out, p=0.01, training=self.training)
        out = self.lin2(out)


        return out

# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=HvhgQoO8Svw4
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, graph_conv, use_cuda):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        print("hidden channels:", str(hidden_channels))
        if (graph_conv):
            self.conv1 = GraphConv(num_node_features, hidden_channels) #GCNConv # GraphConv
            self.conv2 = GraphConv(hidden_channels, hidden_channels)
            self.conv3 = GraphConv(hidden_channels, hidden_channels)
            #self.conv4 = GraphConv(hidden_channels, hidden_channels)
            #self.conv5 = GraphConv(hidden_channels, 2*hidden_channels)
        else:
        #     self.conv1 = GCNConv(num_node_features, hidden_channels) #GCNConv # GraphConv
        #     self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #     self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #     self.conv4 = GCNConv(hidden_channels, 2*hidden_channels)
        # self.lin1 = nn.Linear(2*hidden_channels, hidden_channels)
        # self.lin2 = nn.Linear(hidden_channels, num_classes)
        # self.lin3 = nn.Linear(num_node_features, hidden_channels)
        # self.fc1  = torch.nn.Linear(2*hidden_channels, hidden_channels)
        # self.fc2  = torch.nn.Linear(hidden_channels, num_classes)
        # self.bn1 = BatchNorm(hidden_channels)
        # self.bn2 = BatchNorm(hidden_channels)
        # self.bn3 = BatchNorm(hidden_channels)
        # self.bn4 = BatchNorm(2*hidden_channels)
       
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)


        self.TORCH_DEVICE_STR = (
            "cuda"
            if torch.cuda.is_available() and use_cuda
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = x.to(self.TORCH_DEVICE)
        # if batch is None:
        #     batch = x.new_zeros(x.size(0), dtype=torch.long)
        edge_index = edge_index.to(self.TORCH_DEVICE)
        batch = batch.to(self.TORCH_DEVICE)


        
        
        # x = self.conv1(x, edge_index)
        # #x = self.bn1(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv2(x, edge_index)
        # #x = self.bn2(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv3(x, edge_index)
        # #x = self.bn3(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv4(x, edge_index)
        # #x = self.bn4(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = global_mean_pool(x, batch) 
        # x = self.fc1(x)
        # x = self.fc2(x)


        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        #print("after pooling")
        #print(x)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.lin(x)


        return x




class Dataset(Geoset):
    """
    Dataset class for training. It is used by torch.utils.data.DataLoader to
    generate training batches for the training loop. Training examples include
    phylogenetic-state tensors, auxiliary data tensors, and labels.
    """
    # Constructor
    def __init__(self, phy_data, node_data, edges_data, aux_data, idx_data, labels_num, labels_cat, graph_ids, num_nodes, num_edges):
        self.phy_data    = torch.from_numpy(np.transpose(phy_data, axes=[0,2,1]).astype('float32'))
        self.aux_data    = torch.from_numpy(aux_data.astype('float32'))
        self.idx_data    = torch.from_numpy(idx_data.astype('int'))
        self.labels_num  = torch.from_numpy(labels_num.astype('float32'))
        self.labels_cat  = torch.from_numpy(labels_cat.astype('int'))
        self.len         = len(self.labels_num) #self.labels_num.shape[0]

        edges_data = torch.from_numpy(edges_data.astype('int'))
        min_index =   edges_data.min()
        edges_data = torch.sub(edges_data, min_index)
        unique = torch.unique(edges_data)
        all = torch.arange(edges_data.max()+1)
        difference = all[torch.isin(all, unique, invert=True)]
        reduction = torch.searchsorted(difference, all, right=False)
        edges_data=  torch.sub(edges_data, reduction[edges_data])
        #self.graph_data = GeoData(x=torch.transpose(torch.from_numpy(node_data).view(1,-1),0,1).float(), edge_index=edges_data, y=labels_cat)
        self.graph_dat = []
        self.id_list = graph_ids
        i = 0
        num_nodes = num_nodes.astype(int)
        num_edges = num_edges.astype(int)
        if num_nodes[0] != num_edges[0] + 1:
            print("nodes:", num_nodes)
            print("edges:", num_edges)
            quit()
        prev_edge_ind = 0
        prev_node_ind = 0
        # print("edges:")
        # print(edges_data)
        # print(edges_data.shape)
        for i in range(len(graph_ids)):
            # print("graph =", graph_ids[i], "nodes: ", num_nodes[i], "edges:", num_edges[i])
            current_edge_ind = prev_edge_ind + int(num_edges[i])
            current_node_ind = prev_node_ind + int(num_nodes[i])
            selected_nodes = node_data[prev_node_ind:current_node_ind]
            selected_edges = edges_data[:, prev_edge_ind:current_edge_ind]
        
            #print("i = ", i, "nodes", selected_nodes.shape, "edges", selected_edges.shape)
            self.graph_dat.append(GeoData(x=torch.transpose(torch.from_numpy(selected_nodes).view(1,-1),0,1).float(), edge_index=selected_edges, y=labels_cat[i]))#, phy_data=self.phy_data, aux_data=self.aux_data))
            prev_edge_ind = current_edge_ind
            prev_node_ind = current_node_ind
        # print("self graph dat")
        # print(self.graph_dat)

    # Getting the data
    def __getitem__(self, index):
        #print("getting graph  index", str(index), ":",self.graph_dat[index] )
        #return (#self.phy_data[index], self.graph_dat[index], #[index],
                #self.aux_data[index], self.idx_data[index],
                #self.labels_num[index], self.labels_cat[index])
        return(self.graph_dat[index], self.idx_data[index],
                self.labels_num[index], self.labels_cat[index])
    
    # Getting length of the data
    def __len__(self):
        return self.len

##################################################


class ParameterEstimationNetwork(nn.Module):
    """
    Parameter estimation neural network. This class defines the network
    structure, activation functions, and forward pass behavior for of input
    to predict labels.
    
    Args:
            args (dict): Contains phyddle settings.
    """
    def __init__(self, num_node_features, num_classes, phy_dat_width, phy_dat_height, aux_dat_width,
                 lbl_width, param_cat, phylo_pool, graph_conv, args):


        
        # initialize base class
        super(ParameterEstimationNetwork, self).__init__()


        # width for key input/output
        self.phy_dat_width  = phy_dat_width
        self.phy_dat_height = phy_dat_height
        self.aux_dat_width  = aux_dat_width
        self.lbl_width      = lbl_width
        self.param_cat      = param_cat
        self.param_cat_size = dict()
        self.model_type_categ_ffnn = []
        
        self.phylo_pool = phylo_pool

        self.has_param_num  = self.lbl_width > 0
        self.has_param_cat  = len(self.param_cat) > 0

        # collect args
        self.network_type = list(args['network_type'])
        self.phy_std_hidden_size = args['phy_hidden_size']
        self.phy_std_out_size       = list(args['phy_channel_plain'])
        self.phy_std_kernel_size    = list(args['phy_kernel_plain'])
        self.phy_stride_out_size    = list(args['phy_channel_stride'])
        self.phy_stride_kernel_size = list(args['phy_kernel_stride'])
        self.phy_stride_stride_size = list(args['phy_stride_stride'])
        self.phy_dilate_out_size    = list(args['phy_channel_dilate'])
        self.phy_dilate_kernel_size = list(args['phy_kernel_dilate'])
        self.phy_dilate_dilate_size = list(args['phy_dilate_dilate'])
        self.aux_out_size           = list(args['aux_channel'])
        self.lbl_channel            = list(args['lbl_channel'])
        self.activation_func        = args['activation_func']
        self.use_cuda               = args['use_cuda']

        self.TORCH_DEVICE_STR = (
            "cuda"
            if torch.cuda.is_available() and self.use_cuda
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)

        # define activation function
        self.fwd_func = func.relu
        if self.activation_func == 'relu':
            pass
        elif self.activation_func == 'leaky_relu':
            self.fwd_func = func.leaky_relu
        elif self.activation_func == 'elu':
            self.fwd_func = func.elu
        elif self.activation_func == 'tanh':
            self.fwd_func = func.tanh
        elif self.activation_func == 'sigmoid':
            self.fwd_func = func.sigmoid

        # standard convolution and pooling layers for CPV+S
        self.phy_std_in_size = [ self.phy_dat_width ] + self.phy_std_out_size[:-1]
        assert len(self.phy_std_out_size) == len(self.phy_std_kernel_size)

        # stride convolution and pooling layers for CPV+S
        self.phy_stride_in_size = [ self.phy_dat_width ] + self.phy_stride_out_size[:-1]
        assert len(self.phy_stride_out_size) == len(self.phy_stride_kernel_size)
        assert len(self.phy_stride_out_size) == len(self.phy_stride_stride_size)
        
        # dilate convolution and pooling layers for CPV+S
        self.phy_dilate_in_size = [ self.phy_dat_width ] + self.phy_dilate_out_size[:-1]
        assert len(self.phy_dilate_out_size) == len(self.phy_dilate_kernel_size)
        assert len(self.phy_dilate_out_size) == len(self.phy_dilate_dilate_size)
        
        # dense feed-forward layers for aux. data
        self.aux_in_size = [ self.aux_dat_width ] + self.aux_out_size[:-1]
        
        # concatenation layer size (used in build_network
        self.concat_size = self.phy_std_out_size[-1] + \
                           self.phy_stride_out_size[-1] + \
                           self.phy_dilate_out_size[-1] + \
                           self.aux_out_size[-1]
        
        # dense layers for output predictions
        self.label_num_out_size = self.lbl_channel + [ self.lbl_width ]
        self.label_in_size = [ self.concat_size ] + self.label_num_out_size[:-1]
        self.label_cat_out_size = dict()
        for k,v in self.param_cat.items():
            self.label_cat_out_size[k] = self.lbl_channel + [ int(v) ]
        
        # build network
        self.phy_std = nn.ModuleList([])
        if self.network_type == "CNN" or self.network_type == None:
            # standard convolution and pooling layers for CPV+S
            # self.phy_std = nn.ModuleList([])
            for i in range(len(self.phy_std_out_size)):
                c_in  = self.phy_std_in_size[i]
                c_out = self.phy_std_out_size[i]
                k     = self.phy_std_kernel_size[i]
                torch.set_num_threads(self.num_proc)
                self.phy_std.append(nn.Conv1d(in_channels=c_in,
                                          out_channels=c_out,
                                          kernel_size=k,
                                          padding='same'))
            self.phy_std.append(nn.AdaptiveAvgPool1d(1))

            # stride convolution and pooling layers for CPV+S
            self.phy_stride = nn.ModuleList([])
            for i in range(len(self.phy_stride_out_size)):
                c_in  = self.phy_stride_in_size[i]
                c_out = self.phy_stride_out_size[i]
                k     = self.phy_stride_kernel_size[i]
                s     = self.phy_stride_stride_size[i]
                self.phy_stride.append(nn.Conv1d(in_channels=c_in,
                                                 out_channels=c_out,
                                                 kernel_size=k,
                                                 stride=s))
            self.phy_stride.append(nn.AdaptiveAvgPool1d(1))

            # dilate convolution and pooling layers for CPV+S
            self.phy_dilate = nn.ModuleList([])
            for i in range(len(self.phy_dilate_out_size)):
                c_in  = self.phy_dilate_in_size[i]
                c_out = self.phy_dilate_out_size[i]
                k     = self.phy_dilate_kernel_size[i]
                d     = self.phy_dilate_dilate_size[i]
                self.phy_dilate.append(nn.Conv1d(in_channels=c_in,
                                                 out_channels=c_out,
                                                 kernel_size=k,
                                                 dilation=d, padding='same'))
            self.phy_dilate.append(nn.AdaptiveAvgPool1d(1))

            # dense feed-forward layers for aux. data
            self.aux_ffnn = nn.ModuleList([])
            for i in range(len(self.aux_out_size)):
                c_in  = self.aux_in_size[i]
                c_out = self.aux_out_size[i]
                self.aux_ffnn.append(nn.Linear(c_in, c_out))

            if self.has_param_num:
                # dense layers for point/bound estimates
                self.point_ffnn = nn.ModuleList([])
                self.lower_ffnn = nn.ModuleList([])
                self.upper_ffnn = nn.ModuleList([])
                for i in range(len(self.label_num_out_size)):
                    c_in  = self.label_in_size[i]
                    c_out = self.label_num_out_size[i]
                    self.point_ffnn.append(nn.Linear(c_in, c_out))
                    self.lower_ffnn.append(nn.Linear(c_in, c_out))
                    self.upper_ffnn.append(nn.Linear(c_in, c_out))

            if self.has_param_cat:
                # dense layers for categorical predictions
                self.categ_ffnn = dict()
                for k,v in self.param_cat.items():
                    k_str = f'{k}_categ_ffnn'
                    k_mod_list = nn.ModuleList([])
                    for i in range(len(self.label_num_out_size)):
                        c_in  = self.label_in_size[i]
                        c_out = self.label_cat_out_size[k][i]
                        k_mod_list.append(nn.Linear(c_in, c_out))
                    setattr(self, k_str, k_mod_list)
        else:
            print("GNN phy_std appending")
            print("num features: " + str(num_node_features))
            print("hidden size: " + str(self.phy_std_hidden_size))
            print("num classes: " + str(num_classes))


            # self.phy_std.append(GCNConv(num_node_features, self.phy_std_hidden_size))
            # for i in range(1,5):
            #     self.phy_std.append(GCNConv(self.phy_std_hidden_size, self.phy_std_hidden_size))
            # self.phy_std.append(GCNConv(self.phy_std_hidden_size, self.phy_std_hidden_size*2))
            # self.phy_std.append(nn.Linear(2*self.phy_std_hidden_size, self.phy_std_hidden_size))
            # self.phy_std.append(nn.Linear(self.phy_std_hidden_size, num_classes))    
            if (self.phylo_pool):
                self.phy_std = GCN_PhyloPool(num_node_features = num_node_features, hidden_channels = self.phy_std_hidden_size, num_classes = num_classes, use_cuda = self.use_cuda)
            else:
                self.phy_std = GCN(num_node_features = num_node_features, hidden_channels = self.phy_std_hidden_size, num_classes = num_classes, graph_conv = graph_conv, use_cuda = self.use_cuda)

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
        
        # Phylogenetic Tensor forwarding
        num_sample = 500 # phy_dat.shape[0]

        # MJL: Does this need to be set? Seems like no.
        # phy_dat.requires_grad = True
        # aux_dat.requires_grad = True
        
        # if self.network_type == "CNN" or self.network_type == None:
        #     # standard conv + pool layers
        #     x_std = phy_dat
        #     for i in range(len(self.phy_std)-1):
        #         # AMT: Segfault when using Tesla T4 GPU. Occures on next line, second pass of the loop 
        #         x_std = self.fwd_func(self.phy_std[i](x_std))
        #     x_std = self.phy_std[-1](x_std)
        
        #     # stride conv + pool layers
        #     x_stride = phy_dat
        #     for i in range(len(self.phy_stride)-1 ):
        #         x_stride = self.fwd_func(self.phy_stride[i](x_stride))
        #     x_stride = self.phy_stride[-1](x_stride)
        
        #     # dilation conv + pool layers
        #     x_dilate = phy_dat
        #     for i in range(len(self.phy_dilate)-1):
        #         x_dilate = self.fwd_func(self.phy_dilate[i](x_dilate))
        #     x_dilate = self.phy_dilate[-1](x_dilate)
        
        #     # dense aux. dat layers
        #     x_aux = aux_dat
        #     for i in range(len(self.aux_ffnn)):
        #         x_aux = self.fwd_func(self.aux_ffnn[i](x_aux))
        #     x_aux = x_aux.unsqueeze(dim=2)

        #     # Concatenate phylo and aux layers
        #     x_concat = torch.cat((x_std, x_stride, x_dilate, x_aux), dim=1).squeeze()
        # else:
        if True:
            # x_std = graph_dat
            # for i in range(len(self.phy_std)-1):
            #     x_std = self.fwd_func(self.phy_std[i](x_std.x))
            # x_concat = x_std
            graph_dat = graph_dat.to(self.TORCH_DEVICE)

            # if graph_dat.batch is None:
            #     graph_dat.batch = graph_dat.x.new_zeros(graph_dat.x.size(0), dtype=torch.long)

            #print("data.edge_index")
            #print(graph_dat.edge_index)

            norm_features = (graph_dat.x - 0.5)*2
            # print("norm features")
            # print(norm_features.flatten())
            # print(norm_features.shape)
            # print(graph_dat.edge_index.shape)
            # print(graph_dat.batch.shape)
            x_concat = self.phy_std(norm_features, graph_dat.edge_index, graph_dat.batch)
            x_point = torch.empty((num_sample,0), device=self.TORCH_DEVICE)
            x_lower = torch.empty((num_sample,0), device=self.TORCH_DEVICE)
            x_upper = torch.empty((num_sample,0), device=self.TORCH_DEVICE)

        if self.has_param_num:
            # Point estimate path
            x_point = x_concat
            for i in range(len(self.point_ffnn)-1):
                x_point = self.fwd_func(self.point_ffnn[i](x_point))
            x_point = self.point_ffnn[-1](x_point)
    
            # Lower quantile path
            x_lower = x_concat
            for i in range(len(self.lower_ffnn)-1):
                x_lower = self.fwd_func(self.lower_ffnn[i](x_lower))
            x_lower = self.lower_ffnn[-1](x_lower)
    
            # Upper quantile path
            x_upper = x_concat
            for i in range(len(self.upper_ffnn)-1):
                x_upper = self.fwd_func(self.upper_ffnn[i](x_upper))
            x_upper = self.upper_ffnn[-1](x_upper)
        else:
            # DataParallel and CUDA apparently require that torch.empty structures
            # that appear inside the training loop are explicitly placed on
            # the CUDA device
            # see: https://discuss.pytorch.org/t/assertionerror-gather-function-not-implemented-for-cpu-tensors/142088/2
            x_point = torch.empty((num_sample,0), device=self.TORCH_DEVICE)
            x_lower = torch.empty((num_sample,0), device=self.TORCH_DEVICE)
            x_upper = torch.empty((num_sample,0), device=self.TORCH_DEVICE)

        x_categ = dict()
        if self.has_param_cat and (self.network_type == "CNN" or self.network_type == None):
            # Categorical paths
            for k,v in self.param_cat.items():
                # take initial input from x_concat
                if k not in x_categ:
                    x_categ[k] = x_concat
                k_str = f'{k}_categ_ffnn'
                k_mod_list = getattr(self, k_str)
                for i in range(len(k_mod_list)-1):
                    x_categ[k] = self.fwd_func(k_mod_list[i](x_categ[k]))
                # x_categ[k] = func.softmax(k_mod_list[-1](x_categ[k]), dim=1)
                x_categ[k] = k_mod_list[-1](x_categ[k])
                setattr(self, k_str, k_mod_list)
        else:
            x_categ = x_concat#.to(self.TORCH_DEVICE)
            # print("x categ")
            # print(x_categ)
            
        # return loss
        return x_point, x_lower, x_upper, x_categ

##################################################


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
        weights = 1 / torch.bincount(targets).float()
        # print("weights:", weights)
        weight_tensor = targets.clone().float()
        if len(weight_tensor) < 4 or min(weight_tensor) == 0:
            loss_func = torch.nn.CrossEntropyLoss(reduction = 'mean')
        else:
            weight_tensor[weight_tensor == 0] = weights[0]
            weight_tensor[weight_tensor == 1] = weights[1]
            weight_tensor[weight_tensor == 2] = weights[2]
            weight_tensor[weight_tensor == 3] = weights[3]
        #targets = targets.float()
            weight_tensor = weight_tensor.flatten().unsqueeze(1)   
            loss_func = torch.nn.CrossEntropyLoss(reduction = 'mean', weight=weights)
        #print("predictions *******")
        ##predictions = predictions.flatten().unsqueeze(1) # needed for bcewithlogitloss
        #print(predictions)
        #print("targets")
        targets  = targets.unsqueeze(1)
        #print(targets.flatten())
        loss_list = [loss_func(predictions, targets.flatten())]  

        #loss_func = torch.nn.BCEWithLogitsLoss(reduction = 'mean', weight=weight_tensor) #weight=weights
        #predictions = predictions.flatten().unsqueeze(1) # needed for bcewithlogitloss
        #targets  = targets.unsqueeze(1)
        #loss_list = [loss_func(predictions, targets)]  

        # assumes that order of entries in predictions
        # matches order of entries in targets; could be unsafe
        #for i,(k,v) in enumerate(predictions.items()):
           #loss_list.append(loss_func(v, targets[:,i]))
        
        # print("list")
        # print(loss_list)


        return torch.sum(torch.stack(loss_list))

