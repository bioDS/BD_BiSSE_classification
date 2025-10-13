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
from torch_geometric.nn import GraphConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Dataset as Geoset, Data as GeoData, Batch as GeoBatch
from torch_geometric.utils import to_torch_coo_tensor
import torch.nn.functional as F

# phyddle imports
# from phyddle import utilities as util

##################################################

# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=HvhgQoO8Svw4
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, use_cuda):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
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
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        #print("batch: ", (batch))
        edge_index = edge_index.to(self.TORCH_DEVICE)
        batch = batch.to(self.TORCH_DEVICE)
        x = self.conv1(x, edge_index)
        #print("edge index")
        #print(edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # print("end shape")
        # print(x.shape)
        return x




class Dataset(Geoset):
    """
    Dataset class for training. It is used by torch.utils.data.DataLoader to
    generate training batches for the training loop. Training examples include
    phylogenetic-state tensors, auxiliary data tensors, and labels.
    """
    # Constructor
    def __init__(self, phy_data, node_data, edges_data, aux_data, idx_data, labels_num, labels_cat):
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
        self.graph_data = GeoData(x=torch.transpose(torch.from_numpy(node_data).view(1,-1),0,1).float(), edge_index=edges_data, y=labels_cat)



    # Getting the data
    def __getitem__(self, index):
        return (self.phy_data[index], self.graph_data, #[index],
                self.aux_data[index], self.idx_data[index],
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
                 lbl_width, param_cat, args):


        
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
           
            self.phy_std = GCN(num_node_features = num_node_features, hidden_channels = self.phy_std_hidden_size, num_classes = num_classes, use_cuda = self.use_cuda)

        # initialize weights for layers
        #self._initialize_weights()

        return

    def _initialize_weights(self):
        """Initializes weights for network."""
        print("module types")
        for m in self.modules():
            print(type(m))
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
        return

    def forward(self, phy_dat, graph_dat, aux_dat):
        """Forward-pass function of input through network to output labels."""
        
        # Phylogenetic Tensor forwarding
        num_sample = phy_dat.shape[0]

        # MJL: Does this need to be set? Seems like no.
        # phy_dat.requires_grad = True
        # aux_dat.requires_grad = True
        
        if self.network_type == "CNN" or self.network_type == None:
            # standard conv + pool layers
            x_std = phy_dat
            for i in range(len(self.phy_std)-1):
                # AMT: Segfault when using Tesla T4 GPU. Occures on next line, second pass of the loop 
                x_std = self.fwd_func(self.phy_std[i](x_std))
            x_std = self.phy_std[-1](x_std)
        
            # stride conv + pool layers
            x_stride = phy_dat
            for i in range(len(self.phy_stride)-1 ):
                x_stride = self.fwd_func(self.phy_stride[i](x_stride))
            x_stride = self.phy_stride[-1](x_stride)
        
            # dilation conv + pool layers
            x_dilate = phy_dat
            for i in range(len(self.phy_dilate)-1):
                x_dilate = self.fwd_func(self.phy_dilate[i](x_dilate))
            x_dilate = self.phy_dilate[-1](x_dilate)
        
            # dense aux. dat layers
            x_aux = aux_dat
            for i in range(len(self.aux_ffnn)):
                x_aux = self.fwd_func(self.aux_ffnn[i](x_aux))
            x_aux = x_aux.unsqueeze(dim=2)

            # Concatenate phylo and aux layers
            x_concat = torch.cat((x_std, x_stride, x_dilate, x_aux), dim=1).squeeze()
        else:

            # x_std = graph_dat
            # for i in range(len(self.phy_std)-1):
            #     x_std = self.fwd_func(self.phy_std[i](x_std.x))
            # x_concat = x_std
            graph_dat = graph_dat.to(self.TORCH_DEVICE)

            if graph_dat.batch is None:
                graph_dat.batch = graph_dat.x.new_zeros(graph_dat.x.size(0), dtype=torch.long)


            x_concat = self.phy_std(graph_dat.x, graph_dat.edge_index, graph_dat.batch)
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
        targets = targets.flatten().long()
        predictions = predictions.float()

        # print("predictions")
        # print(predictions)
        # print("targets")
        # print(targets)
        loss_func = torch.nn.CrossEntropyLoss(reduction = 'mean')

        
        # assumes that order of entries in predictions
        # matches order of entries in targets; could be unsafe
        #for i,(k,v) in enumerate(predictions.items()):
           #loss_list.append(loss_func(v, targets[:,i]))
        loss_list = [loss_func(predictions, targets)]
        # print("list")
        # print(loss_list)


        return torch.sum(torch.stack(loss_list))

