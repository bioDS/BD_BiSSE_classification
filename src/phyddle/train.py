#!/usr/bin/env python
"""
train
=====
Defines classes and methods for the Training step, which builds and trains a
network using the tensor data from the Formatting step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2025, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
import os

# external imports
import h5py
import numpy as np
import pandas as pd
import torch
from multiprocessing import cpu_count
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# phyddle imports
from phyddle import utilities as util
from phyddle import network
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoLoader, DataListLoader
from torch.utils.data import DataLoader, Subset
from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.data import Batch, Data as GeoData
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch import linalg as LA

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8, device=0)

# dist.init_process_group(backend='nccl')
# local_rank = int(os.environ['LOCAL_RANK'])
# device = torch.device(f'cuda:{local_rank}')

##################################################

#ChatGPT code to read data into memory in batches
class HDF5BlockDataset():
    def __init__(self, file_path, blocks, trainer = None):
        self.file_path = file_path
        # self.chunk_size = chunk_size
        self.trainer = trainer
        self.blocks = blocks

        with h5py.File(file_path, 'r') as f:
            self.idx = f["idx"][:]
            self.num_edges = f["num_edges"][:]
            num_nodes = f["num_nodes"][:]
            # self.label_names = [x.decode("utf-8") for x in f["label_names"][:].ravel()]
            label_names = f["label_names"][:].squeeze()
            label_names = [x.decode("utf-8") for x in label_names]
            # self.label_names = []
            # for l in label_names:
            #     self.label_names.append(l)
            self.label_names = label_names#.to(torch.float32)
            print("self.label_names", self.label_names)
            self.labels = f["labels"][:]
            self.aux_data = f["aux_data"][:]
            self.aux_data_names = [x.decode("utf-8") for x in f["aux_data_names"][:].ravel()]        # full_labels_num, full_labels_cat = self.separate_labels(self.labels)
        self.offsets = np.zeros(len(self.idx)+1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.num_edges)
        self.node_offsets = np.zeros(len(self.idx)+1, dtype=np.int64)
        self.node_offsets[1:] = np.cumsum(num_nodes)
        # print("cumulative sum", self.offsets)
        self.file = None
    
    def __len__(self):
        return len(self.blocks)

    def adjust_num_labels(self, labels_num):
        labels_num = labels_num.to(torch.float32)
        col0 = labels_num[:,0]
        col1 = labels_num[:,1]
        col2 = labels_num[:,2]
        col3 = labels_num[:,3]
        col4 = labels_num[:,4]
        col5 = labels_num[:,5]
        adjusted_labels = torch.zeros((labels_num.shape[0], 4),
                            device=labels_num.device)
        adjusted_labels[:, 0] = torch.where(col0 < 1e7, col0, col2)
        adjusted_labels[:, 1] = torch.where(col0 < 1e7, col0, col3)
        adjusted_labels[:, 2] = torch.where(col1 < 1e7, col1, col4)
        adjusted_labels[:, 3] = torch.where(col1 < 1e7, col1, col5)
        return adjusted_labels

    def __getitem__(self, i):
        # start = i * self.chunk_size
        # end = min((i+1)*self.chunk_size, len(self.idx))
        # print("start:", start, "end:", end)
       
        # print(graph_indices)
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')        
            self.phy_data = self.file["phy_data"]
            self.graph_id = self.file["graph_id"]
            self.node_1 = self.file["node_1"].astype(np.int64)
            self.node_2 = self.file["node_2"].astype(np.int64)
            self.node_attr = self.file["node_attr"]

        graph_indices = self.blocks[i]
        graph_indices = np.sort(graph_indices)
        # print("graph indices:", graph_indices)
        phy_data = torch.from_numpy(self.phy_data[graph_indices].astype(np.float32))
        labels = self.labels[graph_indices]
        idx = torch.as_tensor(self.idx[graph_indices], dtype=torch.long)
        labels_num, labels_cat = self.trainer.separate_labels(labels, self.label_names)
        labels_num = self.adjust_num_labels(labels_num)
        labels_cat = labels_cat.long()
        labels_num = labels_num.float()

        aux_data = torch.from_numpy(self.aux_data[graph_indices].astype(np.float32))

        node_1_list = []
        node_2_list = []
        node_attr_list = []
        graph_id_list = []
        # graph_data = []

        # print("graph indices:", graph_indices)
        graph_list = []
        ind_count = 0
        for g in graph_indices:
            s = self.offsets[g]
            e = self.offsets[g + 1]
            s_nodes = self.node_offsets[g]
            e_nodes = self.node_offsets[g+1]
            edge_index =  torch.from_numpy(np.stack([self.node_1[s:e], self.node_2[s:e]],axis=0).astype(np.int64))
            min_index =   edge_index.min()
            edge_index = torch.sub(edge_index, min_index)
            unique = torch.unique(edge_index)
            all = torch.arange(edge_index.max()+1)
            difference = all[torch.isin(all, unique, invert=True)]
            reduction = torch.searchsorted(difference, all, right=False)
            edge_index=  torch.sub(edge_index, reduction[edge_index])

            graph_list.append(GeoData(x=torch.from_numpy(np.asarray(self.node_attr[s_nodes:e_nodes],dtype=np.float32)), edge_index = edge_index, y = labels_cat[ind_count], idx=g, lbl_cat=labels_cat[ind_count], lbl_num=labels_num[ind_count], aux_dat=torch.from_numpy(self.aux_data[g].astype(np.float32)).view(1,-1)))
            ind_count += 1
        # graph_dat.append(GeoData(x=torch.transpose(selected_nodes, 0, 1), edge_index=selected_edges, y=labels_cat[i]))#, phy_data=self.phy_data, aux_data=self.aux_data))
        #graph_dat = GeoData(x=node_attr, edge_index = torch.from_numpy(np.stack([node_1, node_2],axis=0).astype(np.int64)), y = labels_num)
        # return (graph_dat, idx, labels_num, labels_cat, node_attr, aux_data)
        return graph_list
        # return phy_data, node_attr, node_1, node_2, aux_data, self.aux_data_names, labels, self.label_names, graph_id, self.num_nodes, self.num_edges

def load(args):
    """Load a Trainer object.

    This function creates an instance of the Trainer class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.

    """

    # load object
    train_method = 'default'
    if train_method == 'default':
        return CnnTrainer(args)
    else:
        #return GnnTrainer(args)
        return NotImplementedError
    
##################################################

  # https://medium.com/@piyushkashyap045/how-to-save-and-load-checkpoints-for-training-a-cnn-with-pytorch-e17395cdbd3d
def save_checkpoint(state, filename="train/checkpoint.tar"):
    #print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

##################################################


def custom_collate(batch):
    #print(">>> ENTERED CUSTOM COLLATE <<<", flush=True)
    #phy, graphs, aux, idx, lbl_num, lbl_cat = zip(*batch)
    graphs, idx, lbl_num, lbl_cat = zip(*batch)
    
    #graphs = Batch.from_data_list(graphs) # needed if not using cuda
    return (
        #torch.stack(phy), 
        list(graphs), # remove list() if not using cuda
        #torch.stack(aux),
        torch.stack(idx),
        torch.stack(lbl_num),
        torch.stack(lbl_cat)
    )

class Trainer:
    """
    Class for training neural networks with CPV+S and auxiliary data tensors
    tensors from the Format step. Results from Trainer objects are used in the
    Estimate and Plot steps.
    """

    def __init__(self, args):
        """Initializes a new Trainer object.

        Args:
            args (dict): Contains phyddle settings.

        """
        
        # args
        self.args                   = args

        self.network_type = str(args['network_type'])

        # filesystem
        self.fmt_prefix             = str(args['fmt_prefix'])
        self.trn_prefix             = str(args['trn_prefix'])
        self.fmt_dir                = str(args['fmt_dir'])
        self.trn_dir                = str(args['trn_dir'])
        self.log_dir                = str(args['log_dir'])
        
        # analysis settings
        self.verbose            = bool(args['verbose'])
        self.num_proc           = int(args['num_proc'])
        self.use_parallel       = bool(args['use_parallel'])
        
        # dataset dimensions
        self.num_char           = int(args['num_char'])
        self.num_states         = int(args['num_states'])
        self.tree_width         = int(args['tree_width'])
        
        # dataset processing
        self.tree_encode        = str(args['tree_encode'])
        self.char_encode        = str(args['char_encode'])
        self.brlen_encode       = str(args['brlen_encode'])
        self.char_format        = str(args['char_format'])
        self.tensor_format      = str(args['tensor_format'])
        self.param_est          = dict(args['param_est'])
        self.param_data         = dict(args['param_data'])
        self.log_offset         = float(args['log_offset'])
        self.save_phyenc_csv    = bool(args['save_phyenc_csv'])
        
        # train settings
        self.prop_cal           = float(args['prop_cal'])
        self.prop_val           = float(args['prop_val'])
        self.num_epochs         = int(args['num_epochs'])
        self.load_model         =bool(args['load_model'])
        self.phylo_pool         =bool(args['phylo_pool'])
        self.graph_conv         =bool(args['graph_conv'])
        self.trn_batch_size     = int(args['trn_batch_size'])
        self.cpi_coverage       = float(args['cpi_coverage'])
        self.cpi_asymmetric     = bool(args['cpi_asymmetric'])
        self.loss_numerical     = str(args['loss_numerical'])
        self.use_cuda           = bool(args['use_cuda'])
        self.num_early_stop     = int(args['num_early_stop'])
        self.learning_rate      = float(args['learning_rate'])
        self.activation_func    = str(args['activation_func'])
        self.optimizer          = str(args['optimizer'])
        self.phy_hidden_size    = int(args['phy_hidden_size'])
        self.regularisation = str(args['regularisation'])
        self.regression = bool(args['regression'])
        # initialized later
        self.phy_tensors        = dict()   # init with encode_all()
        self.train_dataset      = None     # init with load_input()
        self.val_dataset        = None     # init with load_input()
        self.calib_dataset      = None     # init with load_input()
        self.train_blocks       = None
        self.cal_blocks         = None
        self.val_blocks         = None
        self.num_msd            = None
        self.attr_msd           = None
        self.aux_msd            = None

        
        # set CPUs
        if self.num_proc <= 0:
            self.num_proc = cpu_count() + self.num_proc
        if self.num_proc <= 0:
            self.num_proc = 1

        # get size of CPV+S tensors
        self.num_tree_col = util.get_num_tree_col(self.tree_encode,
                                                  self.brlen_encode)
        self.num_char_col = util.get_num_char_col(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        self.num_data_col = self.num_tree_col + self.num_char_col

        # create logger to track runtime info
        self.logger = util.Logger(args)
        
        # set torch device
        # NOTE: need to test against cuda
        self.TORCH_DEVICE_STR = (
            "cuda"
            if torch.cuda.is_available() and self.use_cuda
            # else "mps"
            # if torch.backends.mps.is_available()
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)
        
        # done
        return

    def run(self):
        """Builds and trains the network.

        This method loads all training examples, builds the network, trains the
        network, collects results, then saves results to file.

        """
        verbose = self.verbose

        # print header
        util.print_step_header('trn', [self.fmt_dir], self.trn_dir,
                               [self.fmt_prefix], self.trn_prefix, verbose)
        
        # prepare workspace
        os.makedirs(self.trn_dir, exist_ok=True)

        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        print("self.regression: ", self.regression)
        # perform run tasks
        util.print_str('▪ Loading input:', verbose)
        self.load_input()

        num_rjust = len(str(sum(block.shape[0] for block in self.train_blocks)))
        util.print_str(f'  ▪ ' + str(sum(block.shape[0] for block in self.train_blocks)).rjust(num_rjust) + ' training examples', verbose)
        util.print_str(f'  ▪ ' + str(sum(block.shape[0] for block in self.cal_blocks)).rjust(num_rjust) + ' calibration examples', verbose)
        util.print_str(f'  ▪ ' + str(sum(block.shape[0] for block in self.val_blocks)).rjust(num_rjust) + ' validation examples', verbose)

        util.print_str('▪ Training targets:', verbose)
        num_ljust = max([len(k) for k in self.param_est.keys()])
        for k,v in self.param_est.items():
            util.print_str(f'  ▪ {k.ljust(num_ljust)}  [type: {v}]', verbose)


        util.print_str('▪ Building network', verbose)
        self.build_network()

        util.print_str('▪ Training network', verbose)
        device_info = ''
        if self.TORCH_DEVICE_STR == 'cuda':
            device_info = '  ▪ using CUDA + GPU'
            device_info += ' [device: ' + torch.cuda.get_device_properties(0).name + ']'
        elif self.TORCH_DEVICE_STR == 'cpu':
            num_cpu = os.cpu_count()
            device_info = '  ▪ using CPUs [num: ' + str(num_cpu) + ']'
        if device_info != '':
            util.print_str(device_info, verbose)
        self.train()

        util.print_str('▪ Processing results', verbose)
        self.make_results()

        util.print_str('▪ Saving results', verbose)
        self.save_results()

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        util.print_str('▪ ... done!', verbose)
        return
    
    def load_input(self):
        """Loads the input data for the network."""
        raise NotImplementedError

    def build_network(self):
        """Builds the network architecture."""
        raise NotImplementedError

    def train(self):
        """Trains the network using the loaded input data."""
        raise NotImplementedError

    def make_results(self):
        """Generates the results using the trained network."""
        raise NotImplementedError

    def save_results(self):
        """Saves the generated results to a file or storage."""
        raise NotImplementedError

################################################################################


class CnnTrainer(Trainer):
    """
    Class for Convolutional Neural Network (CNN) Trainer.
    """
    def __init__(self, args):
        """
        Initializes a new CnnTrainer object.

        Args:
            args (dict): Contains phyddle settings.

        """
        # initialize base class
        super().__init__(args)
        
        self.aux_data_names = list()    # init with load_input()
        self.label_names    = list()    # init with load_input()
        self.num_aux_data   = int()     # init with load_input()
        self.param_cat_names = list()   # init with load_input()
        self.param_num_names = list()  # init with load_input()
        self.num_param_num = int()     # init with load_input()
        self.num_param_cat  = int()     # init with load_input()
        self.param_cat      = dict()    # init with load_input()

        # todo: revisit and simplify, provide types
        self.train_dataset = None       # init with load_input()
        self.val_dataset   = None       # init with load_input()
        self.calib_dataset = None       # init with load_input()
        self.model = None               # init with build_network()
        self.train_label_num_est = None
        self.train_label_num_true = None
        self.train_label_cat_est = None
        self.train_label_cat_true = None
        self.train_label_num_est_calib = None
        self.train_label_index = None
        self.calib_phy_data_tensor = None
        self.train_history = None       # init with train()
        self.train_label_true = None    # init with load_input()
        self.train_aux_data_mean_sd = (0,0)
        self.train_labels_num_mean_sd = (0,0)
        self.cpi_adjustments = np.array([0,0])
        self.norm_calib_labels_num = None
        self.has_label_cat = False
        self.has_label_num = False
        self.ignore_label_num = False
        # self.regression = True
        self.block_size = 0

        self.scheduler="manual"
        
        return
    
    # splits input into training, test, validation, and calibration
    def split_tensor_idx(self, num_sample):
        """
        Split tensor into parts.

        This function splits the indexes for training examples into training,
        validation, and calibration sets.

        Args:
            num_sample (int): The total number of samples in the dataset.

        Returns:
            train_idx (numpy.ndarray): The indices for the training subset.
            val_idx (numpy.ndarray): The indices for the validation subset.
            calib_idx (numpy.ndarray): The indices for the calibration subset.

        """

        # get number of training, validation, and calibration datapoints
        num_calib = int(np.floor(num_sample * self.prop_cal))
        num_val   = int(np.floor(num_sample * self.prop_val))
        num_train = num_sample - (num_val + num_calib)
        assert num_train > 0

        # create input subsets
        train_idx = np.arange(num_train, dtype='int')
        val_idx   = np.arange(num_val, dtype='int') + num_train
        calib_idx = np.arange(num_calib, dtype='int') + num_train + num_val

        # return
        return train_idx, val_idx, calib_idx
    
    def validate_tensor_idx(self, train_idx, val_idx, calib_idx):
        """
        Validates input tensors.

        Checks that training, validation, and calibration input tensors are
        each non-empty.

        Args:
            train_idx (list): Training example indices.
            val_idx (list): Validation example indices.
            calib_idx (list): Calibration example indices.

        Returns:
            ValueError if any of the datasets are empty, otherwise returns None.

        """

        msg = ''
        if len(train_idx) == 0:
            msg = 'Training dataset is empty: len(train_idx) == 0'
        elif len(val_idx) == 0:
            msg = 'Validation dataset is empty: len(val_idx) == 0'
        elif len(calib_idx) == 0:
            msg = 'Calibration dataset is empty: len(calib_idx) == 0'
        if msg != '':
            self.logger.write_log('trn', msg)
            raise ValueError(msg)

        return



    def load_input(self):
        """Load input data for the model.

        This function loads input data based on the specified tensor format
        (csv or hdf5). It performs necessary preprocessing steps such as reading
        data from files, reshaping tensors, normalizing summary stats and
        labels, randomizing data, and splitting the dataset into training,
        validation, test, and calibration parts.

        """

        # input dataset filenames for csv or hdf5
        path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.train'
        input_phy_data_fn = f'{path_prefix}.phy_data.csv'
        #input_node_1_data_fn = f'{path_prefix}.node_1.csv'
        input_aux_data_fn = f'{path_prefix}.aux_data.csv'
        input_labels_fn   = f'{path_prefix}.labels.csv'
        input_idx_data_fn = f'{path_prefix}.index.csv'
        input_hdf5_fn = f'{path_prefix}.hdf5'

        N = 0
        with h5py.File(input_hdf5_fn, "r") as f:
            N = f["phy_data"].shape[0]
        perm = np.random.permutation(N)
        print("perm:", perm)
        self.block_size = 3 #100
        blocks = [perm[i:i+self.block_size] for i in range(0,N,self.block_size)]
        split_1 = int(self.prop_cal*len(blocks))
        split_2 = split_1 + int(self.prop_val*len(blocks))
        # print("prop cal:", self.prop_cal)
        # print("split 1:", split_1)
        # print("split 2:", split_2)
        self.cal_blocks = blocks[:split_1]
        self.val_blocks = blocks[split_1:split_2]
        self.train_blocks = blocks[split_2:]
        assert len(self.val_blocks) > 0
        assert len(self.train_blocks) > 0
        assert len(self.cal_blocks) > 0
        # num_calib = int(np.floor(num_sample * self.prop_cal))
        # num_val   = int(np.floor(num_sample * self.prop_val))
        # num_train = num_sample - (num_val + num_calib)
        # assert num_train > 0

        # # create input subsets
        # train_idx = np.arange(num_train, dtype='int')
        # val_idx   = np.arange(num_val, dtype='int') + num_train
        # calib_idx = np.arange(num_calib, dtype='int') + num_train + num_val

        total_num_sum = torch.zeros(1)
        total_num_sq_sum = torch.zeros(1)
        total_num_count = 0
        self.num_node_features = 2 # 3
        total_attr_sum = torch.zeros(self.num_node_features)
        total_attr_sq_sum = 0
        total_attr_count = 0
        total_aux_sum = torch.zeros(16)
        total_aux_sq_sum = torch.zeros(16)
        total_aux_count = 0

        print("LOADING", input_hdf5_fn)
        # val_dataset = HDF5BlockDataset(input_hdf5_fn, val_blocks, self)
        # cal_dataset = HDF5BlockDataset(input_hdf5_fn, cal_blocks, self)
        self.train_dataset = HDF5BlockDataset(input_hdf5_fn, self.train_blocks, self)
        # self.num_classes = len(np.unique(full_labels_cat))
        classes = []
        # #for j, (phy_data, node_attr, node_1, node_2, aux_data, aux_data_names, labels_num, labels_cat, label_names, graph_id, num_nodes, num_edges) in tqdm(enumerate(train_loader),
        # for j, graph_list in tqdm(enumerate(train_loader), #, idx, labels_num, labels_cat, node_attr, aux_data

        #                                                             total=len(self.train_blocks),
        #                                                             # desc=train_msg,
        #                                                             smoothing=0):
        #     print("graph_dat:", graph_list)
        # print("before batch calculation")

        # print("total batches:", self.total_batches)
        pbar = tqdm(total=len(self.train_dataset))
        for block_idx in range(len(self.train_dataset)):
            # print("block idx:", block_idx)
            graphs = self.train_dataset[block_idx]
            # print("graphs:", graphs)
            labels_num =  [g.lbl_num for g in graphs]
            labels_cat =  [g.lbl_cat for g in graphs]
            node_attr = [g.x for g in graphs]
            # for n in node_attr:
            #     print(n)
            node_sums = []
            for g in graphs:
                x = torch.as_tensor(g.x, dtype=torch.float32)  # [N_i, 2]
                labels_num = torch.as_tensor(g.y, dtype=torch.float32)  # [N_i, 2]
                aux = torch.as_tensor(g.aux_dat, dtype=torch.float32)
                total_num_sum += labels_num.sum(dim=0)
                total_num_sq_sum += (labels_num ** 2).sum(dim=0)
                total_num_count += labels_num.shape[0]
                total_attr_sum += x.sum(dim=0)
                total_attr_sq_sum += (x ** 2).sum(dim=0)
                total_attr_count += x.shape[0]
                total_aux_sum += aux.squeeze(0)
                total_aux_sq_sum += (aux.squeeze(0) ** 2)
                total_aux_count += 1
                for lab in np.unique(g.lbl_cat):
                    if lab not in classes:
                        classes.append(lab)
            pbar.update(1)
        pbar.close()
        self.num_classes = len(classes)
        print("total num count:", total_num_count)
        print("total num sum:", total_num_sum)
        print("total num sq:", total_num_sq_sum)
        mean_num = total_num_sum / total_num_count
        sd_num = total_num_sq_sum / total_num_count - mean_num
        mean_attr = total_attr_sum / total_attr_count
        sd_attr = total_attr_sq_sum / total_attr_count - mean_attr
        mean_aux = total_aux_sum / total_aux_count

        sd_aux = total_aux_sq_sum / total_aux_count - mean_aux
        self.num_msd = (mean_num, sd_num)
        self.attr_msd = (mean_attr, sd_attr)
        self.aux_msd = (mean_aux, sd_aux)
        self.num_msd = tuple(t.to(self.TORCH_DEVICE) for t in self.num_msd)
        self.aux_msd = tuple(t.to(self.TORCH_DEVICE) for t in self.aux_msd)
        self.attr_msd = tuple(t.to(self.TORCH_DEVICE) for t in self.attr_msd)
        print("self.num_msd", self.num_msd)
        print("self.attr_msd", self.attr_msd)
        print("self.aux_msd", self.aux_msd)
        # self.num_node_features = 2
        # self.train_dataset = HDF5BlockDataset(input_hdf5_fn, self.train_blocks, self)
        # self.train_loader = GeoLoader(self.train_dataset, batch_size = 1, shuffle = False)
        self.val_dataset = HDF5BlockDataset(input_hdf5_fn, self.val_blocks, self)
        # self.val_loader = GeoLoader(self.val_dataset, batch_size = 1, shuffle = False)
        self.cal_dataset = HDF5BlockDataset(input_hdf5_fn, self.cal_blocks, self)

        return 
    
    def separate_labels(self, input_labels, label_names=None):
        """Separates labels for categorical param_est targets.
        
        This function separates labels into numerical and categorical subsets
        based on the param_est dictionary.
        
        Args:
            labels (numpy.ndarray): The input labels.
            
        Returns:
            labels_num (numpy.ndarray): The numerical-valued labels.
            labels_cat (numpy.ndarray): The categorical labels.
        
        """
        if label_names is not None:
            self.label_names = label_names
        idx_num = list()
        idx_cat = list()
        # print(type(input_labels))
        labels = torch.from_numpy(input_labels).clone()
        # print(type(labels))
        # print(labels)
        # print("items:", self.param_est.items())
        # print(self.label_names)
        for k,v in self.param_est.items():
            if v == 'cat':
                self.has_label_cat = True
                idx = self.label_names.index(k)
                unique_cats, encoded_cats = torch.unique(labels[:,idx],
                                                      return_inverse=True)
                self.param_cat[k] = len(unique_cats)
                labels[:,idx] = encoded_cats.clone().detach()#torch.tensor(encoded_cats, dtype=labels.dtype)
                idx_cat.append( idx )
                if k not in self.param_cat_names:
                    self.param_cat_names.append(k)
                
            # ignore numerical labels as a hack to pass parameter information for plotting without using it for predictions.
            elif v == 'num' and self.ignore_label_num == False:
                self.has_label_num = True
                # print(self.label_names)
                idx_num.append( self.label_names.index(k) )
                if k not in self.param_num_names:
                    self.param_num_names.append(k)
        
        if not self.has_label_num and not self.has_label_cat:
            util.print_err(f"No training labels found.", exit=True)

        # print("idx num", idx_num)
        # print("idx cat", idx_cat)
        # get data subsets
        labels_num = labels[:,idx_num].clone()
        labels_cat = labels[:,idx_cat].clone()

        # done
        return labels_num, labels_cat

##################################################

    def build_network(self):
       
        # torch multiprocessing, eventually need to get working with cuda
        torch.set_num_threads(self.num_proc)
       # if self.network_type == "CNN" or self.network_type == None:
        if True:
            # build model architecture
            self.model = network.ParameterEstimationNetwork(phy_dat_width=self.num_data_col,
                                                        phy_dat_height=self.tree_width,
                                                        num_node_features = self.num_node_features,
                                                        num_classes = self.num_classes,
                                                        aux_dat_width=self.num_aux_data,
                                                        lbl_width=self.num_param_num,
                                                        param_cat=self.param_cat,
                                                        phylo_pool = self.phylo_pool,
                                                        graph_conv = self.graph_conv,
                                                        args=self.args)
        
        #self.model.phy_dat_shape = (self.num_data_col, self.tree_width)
        #self.model.aux_dat_shape = (self.num_aux_data,)

        # print(self.model)
        if self.use_cuda:
            self.model = DistributedDataParallel(self.model)
        self.model.to(self.TORCH_DEVICE)

        return


##################################################

    def make_loss_numerical_func(self):
        """Makes loss function for numerical-valued labels.

        This function makes a loss function for numerical-valued labels based on the
        specified loss function in the phyddle settings.

        Returns:
            loss_func (torch.nn.Module): The loss function for numerical-valued labels.

        """
        if self.loss_numerical == 'mse':
            loss_func = torch.nn.MSELoss()
        elif self.loss_numerical == 'mae':
            loss_func = torch.nn.L1Loss()
        else:
            raise ValueError(f'Unknown loss function: {self.loss_numerical}')
        
        return loss_func

    def train(self):
        """Trains the neural network model.

        This function compiles the network model to prepare it for training.
        Perform training by compiling the model with appropriate loss functions
        and metrics, and then fitting the model to the training data. Training
        produces a history dictionary, which is saved.

        Returns:
            None

        """

        n_train_graphs = sum(block.shape[0] for block in self.train_blocks)
        n_val_graphs = sum(block.shape[0] for block in self.val_blocks)
        n_calib_graphs = sum(block.shape[0] for block in self.cal_blocks)


        # training dataset
        # train_loader = DataLoader(dataset=self.train_dataset,
        #                                            batch_size=self.trn_batch_size, collate_fn = custom_collate)
        # val_loader = DataLoader(dataset=self.val_dataset,
        #                                            batch_size=self.trn_batch_size, collate_fn = custom_collate)

        # print("training batch size: ", str(self.trn_batch_size))
        # print("graph shape",len(self.train_dataset.graph_dat))
        # n_train_graphs = len(self.train_dataset.graph_dat)
        # n_val_graphs = len(self.val_dataset.graph_dat)
        # n_calib_graphs = len(self.calib_dataset.graph_dat)

        # num_batches = int(np.ceil(n_train_graphs / self.trn_batch_size))
        # val_num_batches = int(np.ceil(n_val_graphs / self.trn_batch_size))
        # calib_num_batches = int(np.ceil(n_calib_graphs / self.trn_batch_size))


        print("number of batches: " + str(len(self.train_blocks)))

        # validation dataset

        # val_idx_dat  = torch.Tensor(self.val_dataset.idx_data).to(self.TORCH_DEVICE)
        # val_lbl_num  = torch.Tensor(self.val_dataset.labels_num).to(self.TORCH_DEVICE)
        # val_lbl_cat  = torch.Tensor(self.val_dataset.labels_cat).to(self.TORCH_DEVICE)
        # val_graph_dat = [self.val_dataset.graph_dat[i].to(self.TORCH_DEVICE) for i in range(len(self.val_dataset.graph_dat))]
        val_bad_count = 0
        val_cycle_good_count = 0


        # loss functions
        q_width = self.cpi_coverage
        q_tail  = (1.0 - q_width) / 2
        q_lower = q_tail
        q_upper = 1.0 - q_tail
        loss_value_func = self.make_loss_numerical_func()
        loss_lower_func = network.QuantileLoss(alpha=q_lower)
        loss_upper_func = network.QuantileLoss(alpha=q_upper)
        loss_categ_func = network.CrossEntropyLoss()
        loss_aggregation = 'median'
        
        print("learning rate: ", str(self.learning_rate))

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        if self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        elif self.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #factor=0.5, patience=3, threshold=0.0001, threshold_mode='abs')
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)

            
        # optimizer = torch.optim.Adam(self.model.parameters(),
        #                              lr=0.001,
        #                              weight_decay = 0.002)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001,
        #                               betas=(0.9, 0.999), eps=1e-08,
        #                               weight_decay=0.01, ams_grad=False)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                             step_size = 50,
        #                                             gamma = 0.1)
        


        # TODO: simplify training logging!!
        # gather training history
        history_col_names = ['epoch', 'dataset', 'metric', 'value']
        self.train_history = pd.DataFrame(columns=history_col_names)

        # training

        prev_trn_loss_combined = None
        prev_val_loss_combined = None
        # prev_trn_acc_combined = None
        # prev_val_acc_combined = None

        #checkpointing code from https://medium.com/@piyushkashyap045/how-to-save-and-load-checkpoints-for-training-a-cnn-with-pytorch-e17395cdbd3d
        if self.load_model:
            checkpoint = torch.load("sim_altered_params/500_epochs/500_epochs_GraphConv.tar")#f'{self.trn_dir}'+""/checkpoint.tar")
            load_checkpoint(checkpoint, self.model, optimizer)
            print("Model's state_dict:")
            for param_tensor in self.model.state_dict():
                print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
            print("Optimizer's state_dict:")
            for var_name in optimizer.state_dict():
                print(var_name, "\t", optimizer.state_dict()[var_name])


        learning_rate = self.learning_rate
        old_learning_rate = learning_rate

        print("plotting on")
        plt.ion()
        fig, ax = plt.subplots()
        train_line, = ax.plot([], [], label="Train")
        val_line, = ax.plot([], [], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()
        trn_losses = []
        val_losses = []
        num_batches = len(self.train_blocks)
        for i in range(self.num_epochs):

            if old_learning_rate != learning_rate:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate  
                print("changing learning rate to:", learning_rate)
            old_learning_rate = learning_rate
            # print('-----')
            trn_loss_value = 0.
            trn_loss_lower = 0.
            trn_loss_upper = 0.
            trn_loss_combined = 0.
            trn_acc_combined = 0.
            val_acc_combined = 0.
            train_correct = 0.
            val_correct = 0.
            train_length = 0.
            val_length = 0.
            val_loss_combined = 0.
            val_loss_all_batches = 0
            trn_mse_value = 0.
            trn_mape_value = 0.
            trn_mae_value = 0.

            train_msg = f'Training epoch {i+1} of {self.num_epochs}'
            correct = 0
            total_graphs = 0
            self.model.train()
            accumulation_steps = 15

            optimizer.zero_grad()

            self.total_batches = len(self.train_dataset)*(math.ceil(self.block_size / self.trn_batch_size))
            pbar = tqdm(total=self.total_batches)

            for j in range(len(self.train_dataset)):
                graph_list = self.train_dataset[j]
                loader = DataListLoader(graph_list, batch_size=self.trn_batch_size)
                for batch in loader:
                    lbl_cat = torch.cat([g.lbl_cat for g in batch], dim=0).to(self.TORCH_DEVICE)
                    for g in batch:
                        g.x = util.normalize(g.x, self.attr_msd)
                        g.lbl_num = util.normalize(g.lbl_num,  self.num_msd)
                        g.aux_dat = util.normalize(g.aux_dat, self.aux_msd)
                    # batch.lbl_num = util.normalize(torch.stack([g.lbl_num for g in batch], dim=0).to(self.TORCH_DEVICE), self.num_msd)
                    # batch.aux_dat = util.normalize(torch.stack([g.aux_dat for g in batch], dim=0).to(self.TORCH_DEVICE), self.aux_msd)
                    # batch.x = util.normalize(torch.stack([g.x for g in batch], dim=0).to(self.TORCH_DEVICE), self.attr_msd)
                    # print("updated batch:", batch)
                    try:
                        lbls_hat = self.model(batch)
                        pbar.update(1)
                    except Exception as e:
                        print("exception in lbls_hat", e)
                        quit()
                   
                    third_arg = lbls_hat[3] # [0:int(len(lbls_hat[3])/4)]
                    preds = third_arg #.long()
                    
                    if j % 5 == 0: # % 150
                        amax = preds.argmax(dim=1)

                    loss_list = list()
                    if self.has_label_num and self.regression:
                        # loss_value = loss_value_func(lbls_hat[0], lbl_num)
                        # loss_lower = loss_lower_func(lbls_hat[1], lbl_num)
                        # loss_upper = loss_upper_func(lbls_hat[2], lbl_num)
                        # loss_list += [ loss_value, loss_lower, loss_upper ]
                        loss_value = loss_value_func(lbls_hat[3], lbl_num)
                        loss_list += [ loss_value ]

                    if self.has_label_cat and self.regression == False:
                        loss_categ = loss_categ_func(preds, lbl_cat.flatten())
                        amax = preds.argmax(dim=1)
                        train_correct += int((amax == lbl_cat.flatten()).sum())
                        train_length += len(amax)

                        loss_list += [ loss_categ ]
                    
            if loss_aggregation == 'sum':
                loss_combined = torch.stack(loss_list).sum()
            elif loss_aggregation == 'geometric':
                loss_combined = torch.exp(torch.mean(torch.log(torch.stack(loss_list))))
            elif loss_aggregation == 'median':
                print("loss list", loss_list)
                loss_combined = torch.median(torch.stack(loss_list))
            
            #https://www.geeksforgeeks.org/machine-learning/l1l2-regularization-in-pytorch/

            if self.regularisation == "L1" or self.regularisation == "L1L2":
                l1_lambda = 1e-5
                l1_norm = 0 #sum(p.abs().sum() for p in self.model.parameters())
                l1_norm = sum(p.abs().sum() for p in self.model.parameters()).item()
                # print("loss combined", loss_combined, " l1 norm", l1_norm)
                loss_combined = loss_combined + l1_lambda*l1_norm
            # https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
            if self.regularisation == "L2" or self.regularisation == "L1L2":
                l2_lambda = 1e-4
                l2_norm = sum(p.square().sum() for p in self.model.parameters()).item()
                loss_combined = loss_combined + l2_lambda*l2_norm
                
            if self.regression:
                trn_mse_value     += (torch.mean((lbl_num - lbls_hat[3])**2)).item() / num_batches
                trn_mae_value     += (torch.mean(torch.abs(lbl_num - lbls_hat[3]))).item() / num_batches
                trn_mape_value    += 100. * (torch.median(torch.abs((lbl_num - lbls_hat[3])/lbl_num))).item() / num_batches
            trn_loss_combined += loss_combined.item() / num_batches
                

                
            loss_combined.backward()

            # update network parameters
            if ((j + 1) % accumulation_steps == 0 or (j + 1) == num_batches):
                #print("updating optimizer")
                optimizer.step()


                # reset gradients for tensors
                optimizer.zero_grad()
            pbar.close()
            if self.regression == False:
                trn_acc_combined += train_correct / train_length
                metric_names = ['loss', 'loss_combined', 'accuracy_combined']#, 'accuracy_combined']
                train_metric_vals = [ trn_loss_value, trn_loss_combined, trn_acc_combined ] 
            else:
                metric_names = ['loss', 'loss_combined', 'mse', 'mae', 'mape']#, 'accuracy_combined']
                train_metric_vals = [ trn_loss_value, trn_loss_combined, trn_mse_value, trn_mae_value, trn_mape_value]
            trn_losses.append(trn_loss_combined)

            trn_loss_str = f'    Train        --   loss: {"{0:.4f}".format(trn_loss_combined)}\t'
            if not self.regression:
                trn_acc_str = f'--   acc: {"{0:.4f}".format(trn_acc_combined)}'

            # print("pre model eval")

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total_graphs = 0
                total_val_batches = len(self.val_dataset)*(math.ceil(self.block_size / self.trn_batch_size))
                pbar = tqdm(total=total_val_batches)

                for j in range(len(self.val_dataset)):
                    graph_list = self.val_dataset[j]
                    val_loader = DataListLoader(graph_list, batch_size = self.trn_batch_size)
                    
                    for batch in val_loader:
                        val_lbl_cat = torch.cat([g.lbl_cat for g in batch], dim=0).to(self.TORCH_DEVICE)
                        for g in batch:
                            g.x = util.normalize(g.x, self.attr_msd)
                            g.lbl_num = util.normalize(g.lbl_num,  self.num_msd)
                            g.aux_dat = util.normalize(g.aux_dat, self.aux_msd)
                        val_lbl_num = torch.cat([g.lbl_num for g in batch], dim=0).to(self.TORCH_DEVICE)


                    # # forward pass of validation to estimate labels
                        try:
                            val_lbls_hat       = self.model(batch)
                            pbar.update(1)
                        except Exception as e:
                            print("val exception in lbls_hat, j:", j, e)
                            quit()

                        third_arg = val_lbls_hat[3]#[0:int(len(lbls_hat[3])/4)]
                        # print("hat third:", val_lbls_hat[3])
                        # collect validation metrics
                        val_loss_list = list()
                        val_loss_value = 0.
                        val_loss_lower = 0.
                        val_loss_upper = 0.
                    
                        if self.has_label_num and self.regression:
                            val_loss_value = loss_value_func(lbls_hat[3], val_lbl_num).item()
                            val_loss_list += [ val_loss_value ] # val_loss_lower, val_loss_upper

                        if self.has_label_cat and not self.regression:
                            val_loss_categ = loss_categ_func(third_arg, val_lbl_cat.flatten()).item()
                            val_loss_list += [ val_loss_categ ]

                            # if not self.regression:
                                # predicted_class = third_arg.argmax(dim=1)
                                # val_acc_combined = int((predicted_class.flatten() == val_lbl_cat.flatten()).sum())/len(val_lbl_cat) #BCE
                            pred_class = third_arg.argmax(dim=1) 
                            val_correct += int((pred_class == val_lbl_cat.flatten()).sum())
                            val_length += len(val_lbl_cat)

                if loss_aggregation == 'sum':
                    val_loss_combined = np.sum(val_loss_list)
                elif loss_aggregation == 'geometric':
                    val_loss_combined = np.exp(np.mean(np.log(val_loss_list)))
                elif loss_aggregation == 'median':
                    print("val loss list", val_loss_list)
                    val_loss_combined = np.median(val_loss_list)
                print("val loss combined initial:", val_loss_combined)
                l1_norm = 0 
                l1_lambda = 1e-5
                l2_lambda = 1e-4
                if self.regularisation == "L1" or self.regularisation == "L1L2":
                    for p in self.model.parameters():
                        l1_norm += p.abs().sum()
                    l1_norm = l1_norm.item()
                    val_loss_combined = val_loss_combined + l1_lambda*l1_norm
                if self.regularisation == "L2" or self.regularisation == "L1L2":
                    l2_norm = sum(p.square().sum() for p in self.model.parameters()).item()
                    val_loss_combined = val_loss_combined + l2_lambda*l2_norm
                val_mse_value = 0.
                val_mae_value = 0.
                val_mape_value = 0.
                
                if self.has_label_num and self.regression:
                    val_mse_value      = (torch.mean((val_lbl_num - val_lbls_hat[3])**2)).item()
                    val_mae_value      = (torch.mean(torch.abs(val_lbl_num - val_lbls_hat[3]))).item()
                    val_mape_value     = 100. * (torch.median(torch.abs((val_lbl_num - val_lbls_hat[3]) / val_lbl_num))).item()
                    val_loss_all_batches += val_loss_combined.item() / val_num_batches
                    print("val loss all batches", val_loss_all_batches)
                    val_losses.append(val_loss_all_batches)
                if self.regression == False:              
                    val_acc_combined = val_correct/val_length    
                    print("total correct:", val_correct, " out of ", val_length)
                    val_metric_vals = [ val_loss_value,
                                            val_loss_combined, val_acc_combined]
                    val_acc_str = f'--   acc: {"{0:.4f}".format(val_acc_combined)}'
                    print("val loss combined:", val_loss_combined)
                    val_losses.append(val_loss_combined)
                else:
                    val_metric_vals = [ val_loss_value, 
                                            val_loss_combined, val_mse_value, val_mae_value,
                                            val_mape_value ]

                pbar.close()
                val_loss_str = f'    Validation   --   loss: {"{0:.4f}".format(val_loss_combined)}\t'
        


            # changes in training metrics between epochs
                if self.regression:
                    trn_acc_combined = 0
                    prev_trn_acc_combined = 0
                if i > 0:
                    
                    diff_trn_loss = trn_loss_combined - prev_trn_loss_combined
                    diff_val_loss = val_loss_combined - prev_val_loss_combined
                    if (prev_trn_loss_combined == 0):
                        rat_trn_loss = 100
                    else:
                        rat_trn_loss  = 100 * round(trn_loss_combined / prev_trn_loss_combined - 1.0, ndigits=4)
                    if (prev_val_loss_combined == 0):
                        rat_val_loss = 100
                    else:
                        rat_val_loss  = 100 * round(val_loss_combined / prev_val_loss_combined - 1.0, ndigits=4)
                    
                    diff_trn_loss_str = '{0:+.4f}'.format(diff_trn_loss)
                    diff_val_loss_str = '{0:+.4f}'.format(diff_val_loss)
                    rat_trn_loss_str  = '{0:+.2f}'.format(rat_trn_loss).rjust(4, ' ')
                    rat_val_loss_str  = '{0:+.2f}'.format(rat_val_loss).rjust(4, ' ')

                    
                    trn_color = 31 if diff_trn_loss >= 0 else 32  # green or red
                    val_color = 31 if diff_val_loss >= 0 else 32  # green or red
                    if not self.regression:
                        diff_trn_acc = trn_acc_combined - prev_trn_acc_combined
                        diff_val_acc = val_acc_combined - prev_val_acc_combined
                        if (prev_trn_acc_combined == 0):
                            rat_trn_acc = 100
                        else:
                            rat_trn_acc  = 100 * round(trn_acc_combined / prev_trn_acc_combined - 1.0, ndigits=4)
                        if (prev_val_acc_combined == 0):
                            rat_val_acc = 100
                        else:
                            rat_val_acc  = 100 * round(val_acc_combined / prev_val_acc_combined - 1.0, ndigits=4)
                        diff_trn_acc_str = '{0:+.4f}'.format(diff_trn_acc)
                        diff_val_acc_str = '{0:+.4f}'.format(diff_val_acc)
                        rat_trn_acc_str  = '{0:+.2f}'.format(rat_trn_acc).rjust(4, ' ')
                        rat_val_acc_str  = '{0:+.2f}'.format(rat_val_acc).rjust(4, ' ')
                        trn_acc_change_str  = f'  abs: {util.phyddle_str(diff_trn_acc_str, style=0, color=trn_color)}'
                        trn_acc_change_str += f'  rel: {util.phyddle_str(rat_trn_acc_str, style=0, color=trn_color)}%'
                        val_acc_change_str  = f'  abs: {util.phyddle_str(diff_val_acc_str, style=0, color=val_color)}'
                        val_acc_change_str += f'  rel: {util.phyddle_str(rat_val_acc_str, style=0, color=val_color)}%'
                        trn_acc_str += trn_acc_change_str
                        val_acc_str += val_acc_change_str
                
                    

                    trn_loss_change_str  = f'  abs: {util.phyddle_str(diff_trn_loss_str, style=0, color=trn_color)}'
                    trn_loss_change_str += f'  rel: {util.phyddle_str(rat_trn_loss_str, style=0, color=trn_color)}%'
                    val_loss_change_str  = f'  abs: {util.phyddle_str(diff_val_loss_str, style=0, color=val_color)}'
                    val_loss_change_str += f'  rel: {util.phyddle_str(rat_val_loss_str, style=0, color=val_color)}%'


                    trn_loss_str += trn_loss_change_str
                    val_loss_str += val_loss_change_str



                    if diff_val_loss >= 0:
                        val_bad_count += 1
                        # if learning_rate > 0.00005 and i > 5 and val_cycle_good_count != 0:
                        #     learning_rate = learning_rate / 1.25
                        val_cycle_good_count = 0
                    else:
                        val_bad_count = 0
                        val_cycle_good_count += 1
                        # if val_cycle_good_count == 3:
                        #     learning_rate = learning_rate * 1.25
                        #     val_cycle_good_count = 0

                # if i >= 15:
                    # if (val_bad_count - 2) % 3 == 0:
                    #     learning_rate = learning_rate / 1.25
                    # if val_cycle_good_count != 0 and val_cycle_good_count % 5 == 0:
                    #     learning_rate = learning_rate * 1.25

                prev_trn_loss_combined = trn_loss_combined
                prev_val_loss_combined = val_loss_combined

                if not self.regression:
                    prev_trn_acc_combined = trn_acc_combined
                    prev_val_acc_combined = val_acc_combined
                # display training metric progress
                train_line.set_data(range(len(trn_losses)), trn_losses)                
                val_line.set_data(range(len(val_losses)), val_losses)
                print("trn losses", trn_losses)
                print("val losses", val_losses)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)  # small delay to refresh UI


                print("regression:", self.regression)
                if self.regression:
                    print(trn_loss_str)
                    print("")
                    print(val_loss_str)
                    print("")
                else:
                    print(trn_loss_str, trn_acc_str)
                    print("")
                    print(val_loss_str, val_acc_str)
                    print('')

    
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
                # update network parameters

            
            # update train history log
            self.update_train_history(i, metric_names, train_metric_vals, 'train')
            self.update_train_history(i, metric_names, val_metric_vals, 'validation')
            path_prefix = f'{self.trn_dir}/{self.trn_prefix}'
            model_history_fn = f'{path_prefix}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_history.csv'
            self.train_history.to_csv(model_history_fn, index=False, sep=',',
                                  float_format=util.PANDAS_FLOAT_FMT_STR)

            checkpoint = {'state_dict': self.model.state_dict(),
                            'optimizer': optimizer.state_dict(),}
            save_checkpoint(checkpoint, filename=f'{self.trn_dir}'+"/checkpoint.tar")

      
            # early stopping
            if val_bad_count >= self.num_early_stop and self.num_early_stop > 0 and i < 15:
                print(f'Early stop: validation loss increased for num_early_stop={self.num_early_stop} consecutive epochs')
                plt.close()
                break
        
        # plt.ioff()
        # plt.show()
        plt.close()

        # print(self.train_history)

        return
    
    def update_train_history(self, epoch, metric_names, metric_vals, dataset_name='train',):
        """Updates train history dataframe.
        
        This function appends new rows to the train history dataframe.
        
        Args:
            epoch (int): current epoch
            metric_names (list): names for metrics to be logged
            metric_vals (list): values for metrics to be logged
            dataset_name (str): name of dataset that is logged (e.g. train or validation)

        """
        assert len(metric_names) == len(metric_vals)
        
        for i,(j,k) in enumerate(zip(metric_names, metric_vals)):
            self.train_history.loc[len(self.train_history.index)] = [ epoch, dataset_name, j, k ]
        
        return

    def perform_cpi_calibration(self):
        """Performs CPI calibration.

        This function performs CPI calibration to estimate the CPI adjustment
        terms for the training dataset.

        """
        print("CPI CALIBRATION")

        # temporarily switch to CPU
        # self.model.to('cpu')

        # make initial CPI estimates

        for k in range(len(self.cal_dataset)):
            graph_list = self.cal_dataset[k]
            loader = DataListLoader(graph_list, batch_size=self.trn_batch_size)
            for step, data in enumerate(loader):
                print(f'Calib step {step + 1}:')
                print()
        calib_batch = next(iter(loader))
        try:
            calib_label_est = self.model(calib_batch)
        except Exception as e:
            print("calib exception", e)
            quit()

        # make CPI adjustments
        if self.regression:
            norm_calib_label_num_est = torch.stack(calib_label_est[0:3]).cpu().detach().numpy()
            norm_calib_num_est_quantiles = norm_calib_label_num_est[1:,:,:]
            self.cpi_adjustments = self.get_cqr_constant(norm_calib_num_est_quantiles,
                                                        self.norm_calib_labels_num,
                                                        inner_quantile=self.cpi_coverage,
                                                        asymmetric=self.cpi_asymmetric)
            self.cpi_adjustments = np.array(self.cpi_adjustments).reshape((2,-1))

        # restore device
        # if self.use_cuda:
        #     self.model = torch.nn.DataParallel(self.model)
        # self.model.to(self.TORCH_DEVICE)

        # done
        return
    
    def make_results(self):
        """Makes all results from the Train step.

        This function undoes all the transformation and rescaling for the
        input and output datasets.

        """

        
        # get uncalibrated estimates
        # training label estimates
        first_batch = True
        pbar = tqdm(total=self.total_batches)
        for j in range(len(self.train_dataset)):
            graph_list = self.train_dataset[j]
            loader = DataListLoader(graph_list, batch_size=self.trn_batch_size)

            for batch in loader:
                pbar.update(1)
                # batch = batch.to(self.TORCH_DEVICE)
                train_labels_cat = torch.cat([g.lbl_cat for g in batch], dim=0).to(self.TORCH_DEVICE)
                train_labels_num = torch.cat([g.lbl_num for g in batch], dim=0).to(self.TORCH_DEVICE)

                label_est = self.model(batch)
                labels_num_est = label_est[0:3]
                labels_num_est = torch.stack(labels_num_est).cpu().detach().numpy()
                labels_cat_est = label_est[3]

                if self.has_label_num:
                    train_labels_num = train_labels_num.cpu().detach().numpy()
                    if j == 0 and first_batch:
                        self.train_label_num_true = train_labels_num.copy()
                    else:
                        self.train_label_num_true = np.append(self.train_label_num_true, train_labels_num)
                    
                    # uncalibrated training estimates of numerical labels
                    # self.train_label_num_est = labels_num_est.copy()
                    if j == 0 and first_batch:
                        self.train_label_num_est = labels_num_est
                    else:
                        self.train_label_num_est = np.append(self.train_label_num_est, labels_num_est)
                    # self.train_label_num_est = util.denormalize(labels_num_est.copy(),
                    #                                             self.train_labels_num_mean_sd)
                


                # generate calibration factors

                if self.has_label_cat:
                    train_label_cat = train_labels_cat.cpu().detach().numpy().astype('int')
                    if j == 0 and first_batch:
                        # print("j =", j, "cat true is none")
                        self.train_label_cat_true = train_label_cat.copy()
                    else:
                        self.train_label_cat_true = np.append(self.train_label_cat_true, train_label_cat)
                    # print("UPDATE:")
                    # print(self.train_label_cat_true )
                    if j == 0 and first_batch:
                        self.train_label_cat_est = labels_cat_est.argmax(dim=1).cpu().detach().numpy()
                    else:
                        self.train_label_cat_est = np.append(self.train_label_cat_est, labels_cat_est.argmax(dim=1).cpu().detach().numpy())
                    total_correct = (self.train_label_cat_true == self.train_label_cat_est).sum()
                    # print("accuracy:")
                    # print(total_correct/len(self.train_label_cat_true))
                else:
                    print("no label cat")
                first_batch = False
            if self.regression:
                self.perform_cpi_calibration()

            # calibrate original estimates

            if self.has_label_num and self.regression:    
                labels_num_est_calib = self.train_label_num_est.copy()
                labels_num_est_calib[1,:,:] = labels_num_est_calib[1,:,:] + self.cpi_adjustments[0,:]
                labels_num_est_calib[2,:,:] = labels_num_est_calib[2,:,:] + self.cpi_adjustments[1,:]
        # # denormalize calibrated estimates
        # if self.train_label_num_est_calib == None:
                self.train_label_num_est_calib = labels_num_est_calib
        pbar.close()   

        # else:
        #     self.train_label_num_est_calib = np.append(self.train_label_num_true, labels_num_est_calib)
        # self.train_label_num_est_calib = labels_num_est_calib

        # reformat categorical estimates, if they exist
        return

    def format_label_cat(self, x):
        """Formats categorical labels.

        Formats categorical labels for training and validation datasets.

        """
        
        df_list = list()
        for k,v in x.items():
            # print("applying softmax")
            v = torch.softmax(v, dim=1).cpu().detach().numpy()
            col_names = [ f'{k}_{i}' for i in range(v.shape[1]) ]
            df = pd.DataFrame(v, columns=col_names)
            df_list.append(df)
            
        return pd.concat(df_list, axis=1)

    def save_results(self):
        """Save training results.

        Saves all results from training procedure. Saved results include the
        trained network, the normalization parameters for training/calibration,
        CPI adjustment terms, and the training history.

        """
        # max_idx = 1000

        path_prefix = f'{self.trn_dir}/{self.trn_prefix}'
        
        # output network model info
        model_arch_fn                = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.trained_model.pkl'
        model_history_fn             = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_history.csv'
        model_cpi_fn                 = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.cpi_adjustments.csv'
        # model_weights_fn           = f'{path_prefix}.train_weights.hdf5'
        
        # output scaling terms
        train_labels_num_norm_fn    = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_norm.labels_num.csv'
        train_aux_data_norm_fn       = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_norm.aux_data.csv'

        # output training labels
        train_label_num_true_fn     = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_true.labels_num.csv'
        train_label_num_est_fn      = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_est.labels_num.csv'
        train_label_est_nocalib_fn   = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_est.labels_num_nocalib.csv'
        train_label_cat_true_fn      = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_true.labels_cat.csv'
        train_label_cat_est_fn       = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.regularisation}.train_est.labels_cat.csv'
        
        # save model to file
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save, model_arch_fn)
        write_mode = 'w'


        # save json history from running MASTER
        self.train_history.to_csv(model_history_fn, index=False, sep=',',
                                  float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)

        # save aux_data names, means, sd for new test dataset normalization
        df_aux_data = pd.DataFrame({'name':self.aux_data_names,
                                    'mean':self.train_aux_data_mean_sd[0],
                                    'sd':self.train_aux_data_mean_sd[1]})
        df_aux_data.to_csv(train_aux_data_norm_fn, index=False, sep=',',
                           float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
        
        # training example index
        df_train_label_idx = pd.DataFrame(self.train_label_index, columns=['idx'])
        # print("df_train_label_idx")
        # print(df_train_label_idx)
 
        if self.has_label_num and self.regression:
            # save label names, means, sd for new test dataset normalization
            df_labels = pd.DataFrame({'name':self.param_num_names,
                                      'mean':self.train_labels_num_mean_sd[0],
                                      'sd':self.train_labels_num_mean_sd[1]})
            df_labels.to_csv(train_labels_num_norm_fn, index=False, sep=',',
                             float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
    
            # save CPI intervals
            df_cpi_intervals = pd.DataFrame(self.cpi_adjustments,
                                            columns=self.param_num_names)
            df_cpi_intervals.to_csv(model_cpi_fn,
                                    index=False, sep=',',
                                    float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
            
            # downsample all true training labels
            df_train_label_true = pd.DataFrame(self.train_label_num_true,
                                               columns=self.param_num_names )
            
            # save true values for train numerical labels
            df_train_label_num_true = df_train_label_true[self.param_num_names]
            df_train_label_num_true = util.denormalize(df_train_label_num_true.copy(),
                                                        self.train_labels_num_mean_sd)
            df_train_label_num_true = pd.concat([df_train_label_idx,
                                                 df_train_label_num_true], axis=1 )
            
            df_train_label_num_true.to_csv(train_label_num_true_fn,
                                            index=False, sep=',',
                                            float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
            
            # save train numerical label estimates
            self.train_label_num_est = util.denormalize(self.train_label_num_est,
                                                        self.train_labels_num_mean_sd)
            self.train_label_num_est_calib = util.denormalize(self.train_label_num_est_calib,
                                                              self.train_labels_num_mean_sd)
            df_train_label_num_est_nocalib = util.make_param_VLU_mtx(self.train_label_num_est,
                                                                      self.param_num_names )
            df_train_label_num_est_calib   = util.make_param_VLU_mtx(self.train_label_num_est_calib,
                                                                      self.param_num_names )
            df_train_label_num_est_calib = pd.concat([df_train_label_idx,
                                                     df_train_label_num_est_calib], axis=1 )
            df_train_label_num_est_nocalib = pd.concat([df_train_label_idx,
                                                        df_train_label_num_est_nocalib], axis=1 )
            # print('true')
            # print(df_train_label_num_true.head(n=5))
            # print('est')
            # print(df_train_label_num_est_calib.head(n=5))
    
            # convert to csv and save
            df_train_label_num_est_nocalib.to_csv(train_label_est_nocalib_fn,
                                                   index=False, sep=',',
                                                   float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
            df_train_label_num_est_calib.to_csv(train_label_num_est_fn,
                                                 index=False, sep=',',
                                                 float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
    
        if self.has_label_cat:
            # save true values for train categ. labels
            print("end", self.train_label_cat_true)
            print(self.param_cat_names)
            df_train_label_cat_true = pd.DataFrame(self.train_label_cat_true,
                                                   columns=self.param_cat_names )
            df_train_label_cat_true = pd.concat([df_train_label_idx, df_train_label_cat_true], axis=1 )
            df_train_label_cat_true.to_csv(train_label_cat_true_fn,
                                           index=False, sep=',', mode=write_mode)
    
            # save train categorical label estimates
            #print(self.train_label_cat_est)
            df_train_label_cat_est = self.train_label_cat_est #pd.DataFrame(self.train_label_cat_est[0:max_idx,:],
            #                                      columns=self.param_cat_names )
            df_train_label_cat_est = pd.concat([df_train_label_idx, pd.DataFrame(df_train_label_cat_est)], axis=1 )
            df_train_label_cat_est.to_csv(train_label_cat_est_fn,
                                          index=False, sep=',',
                                          float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
        print("returning from save results")

        return

##################################################
        
    def get_cqr_constant(self, ests, true, inner_quantile=0.95, asymmetric=True):
        """Computes the conformalized quantile regression (CQR) constants.
        
        This function computes symmetric or asymmetric CQR constants for the
        specified inner-quantile range.

        Notes:
            # ests axis 0 is the lower and upper quants,
            # axis 1 is the replicates, and axis 2 is the params

        Arguments:
            ests (array-like): The input data.
            true (array-like): The target data.
            inner_quantile (float): The inner quantile range.
            asymmetric (bool): If True, computes asymmetric CQR constants.

        Returns:
            array-like: The conformity scores.

        """
        
        # compute non-comformity scores
        q_score = np.empty((2, ests.shape[2]))
        
        for i in range(ests.shape[2]):
            if asymmetric:
                # Asymmetric non-comformity score
                lower_s = np.array(true[:,i] - ests[0][:,i])
                upper_s = np.array(true[:,i] - ests[1][:,i])
                lower_p = (1 - inner_quantile)/2 * (1 + 1/ests.shape[1])
                upper_p = (1 + inner_quantile)/2 * (1 + 1/ests.shape[1])
                if lower_p < 0.:
                    self.logger.write_log('trn',
                                          'get_cqr_constant: lower_p >= 0.')
                    lower_p = 0.
                if upper_p > 1.:
                    self.logger.write_log('trn',
                                          'get_cqr_constant: upper_p <= 1.')
                    upper_p = 1.
                lower_q = np.quantile(lower_s, lower_p)
                upper_q = np.quantile(upper_s, upper_p)
            else:
                # Symmetric non-comformity score
                s = np.amax(np.array((ests[0][:,i]-true[:,i], true[:,i]-ests[1][:,i])), axis=0)
                # get adjustment constant: 1 - alpha/2's quantile of non-comformity scores
                symm_p = inner_quantile * (1 + 1/ests.shape[1])
                if symm_p < 0.:
                    self.logger.write_log('trn',
                                          'get_cqr_constant: symm_p >= 0.')
                    symm_p = 0.
                elif symm_p > 1.:
                    self.logger.write_log('trn',
                                          'get_cqr_constant: symm_p <= 1.')
                    symm_p = 1.
                lower_q = np.quantile(s, symm_p)
                upper_q = lower_q
                # Q[:,i] = np.array([lower_q, upper_q])

            q_score[:,i] = np.array([lower_q, upper_q])
                                
        return q_score
    
##################################################
