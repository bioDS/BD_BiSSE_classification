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

# phyddle imports
from phyddle import utilities as util
from phyddle import network
from torch_geometric.data import Data as GeoData
#from torch_geometric.loader import DataLoader as GeoLoader
from torch.utils.data import DataLoader
from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.data import Batch
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8, device=0)

# dist.init_process_group(backend='nccl')
# local_rank = int(os.environ['LOCAL_RANK'])
# device = torch.device(f'cuda:{local_rank}')

##################################################

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
        self.prop_test          = float(args['prop_test'])
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
        # initialized later
        self.phy_tensors        = dict()   # init with encode_all()
        self.train_dataset      = None     # init with load_input()
        self.val_dataset        = None     # init with load_input()
        self.calib_dataset      = None     # init with load_input()
        
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

        # perform run tasks
        util.print_str('▪ Loading input:', verbose)
        self.load_input()
        num_rjust = len(str(len(self.train_dataset)))
        util.print_str(f'  ▪ ' + str(len(self.train_dataset)).rjust(num_rjust) + ' training examples', verbose)
        util.print_str(f'  ▪ ' + str(len(self.calib_dataset)).rjust(num_rjust) + ' calibration examples', verbose)
        util.print_str(f'  ▪ ' + str(len(self.val_dataset)).rjust(num_rjust) + ' validation examples', verbose)

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

        self.scheduler="NA"
        
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
        
        # read phy. data, aux. data, and labels
        full_phy_data = None
        full_aux_data = None
        full_idx_data = None
        full_labels   = None
        if self.tensor_format == 'csv':
            full_phy_data = pd.read_csv(input_phy_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            full_aux_data = pd.read_csv(input_aux_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            full_labels   = pd.read_csv(input_labels_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            full_idx_data = pd.read_csv(input_idx_data_fn,
                                        on_bad_lines='skip').to_numpy()
            # extract headers
            self.aux_data_names = full_aux_data[0,:].tolist()
            self.label_names    = full_labels[0,:].tolist()
            full_aux_data       = full_aux_data[1:,:].astype('float64')
            full_labels         = full_labels[1:,:].astype('float64')

        elif self.tensor_format == 'hdf5':
            hdf5_file = h5py.File(input_hdf5_fn, 'r')
            self.aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            self.label_names    = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            full_phy_data       = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
            full_aux_data       = pd.DataFrame(hdf5_file['aux_data']).to_numpy()
            full_idx_data       = pd.DataFrame(hdf5_file['idx']).to_numpy()
            full_labels         = pd.DataFrame(hdf5_file['labels']).to_numpy()
            full_node_1_data = pd.DataFrame(hdf5_file['node_1'])#.to_numpy()
            full_node_2_data = pd.DataFrame(hdf5_file['node_2'])#.to_numpy()
            full_nodes_dist = pd.DataFrame(hdf5_file['nodes_dist']).to_numpy()
            full_graph_ids = pd.DataFrame(hdf5_file['graph_id']).to_numpy()
            full_num_edges = pd.DataFrame(hdf5_file['num_edges']).to_numpy()
            full_num_nodes = pd.DataFrame(hdf5_file['num_nodes']).to_numpy()

            #full_descendants = pd.DataFrame(hdf5_file['descendant']).to_numpy()
            #full_ancestors = pd.DataFrame(hdf5_file['ancestor']).to_numpy()
            #full_time_asym = pd.DataFrame(hdf5_file['descendant']).to_numpy()
            #full_clade_asym = pd.DataFrame(hdf5_file['clade_asym']).to_numpy()

            hdf5_file.close()
       
        full_edges_matrix = pd.concat((full_node_1_data, full_node_2_data), axis=1).to_numpy()
       
        #full_node_attributes = full_node_attributes[0]
       
        full_node_attributes = full_nodes_dist.flatten()
        #full_node_attributes = [([[np.diag(y) for y in x]  for x in full_nodes_dist ])]

        # separate labels for categorical param_est targets
        # print("full labels")
        # print(full_labels)
        full_labels_num, full_labels_cat = self.separate_labels(full_labels)
        # print("full_labels_cat")
        # print(full_labels_cat)
        # print("full_labels_num")
        # print(full_labels_num)
        # print("length of node attributes")
        # print(len(full_node_attributes))
        # print(full_node_attributes)
        # print("total num_nodes")
        # print((full_num_nodes.astype(int).sum()))
        # print("length of edges")
        # print(len(full_node_1_data))
        # print("total num_edges")
        # print((full_num_edges.astype(int).sum()))
        # quit()
        
        # data dimensions
        num_sample             = full_phy_data.shape[0]
        self.num_param_num    = full_labels_num.shape[1]
        self.num_param_cat     = full_labels_cat.shape[1]
        self.num_aux_data      = full_aux_data.shape[1]
        
        # shuffle datasets
        randomized_idx     =  np.random.permutation(full_phy_data.shape[0]) # np.arange(full_phy_data.shape[0]) #
        print("randomized idx")
        print(randomized_idx)
        unique_ids = np.unique(full_graph_ids)
        #for (unique_id in unique_ids):

        cumulative_edges = np.insert(np.cumsum(full_num_edges, dtype=np.int64), 0, 0)
        cumulative_nodes = np.insert(np.cumsum(full_num_nodes, dtype=np.int64), 0, 0)
        # print("cumulative edges")
        # print(cumulative_edges)
        # print("cumulative nodes")
        # print(cumulative_nodes)
        split_edges = np.split(full_edges_matrix, cumulative_edges[1:])

        joined_edges = np.concatenate([split_edges[x] for x in randomized_idx])
        split_nodes = np.split(full_node_attributes, cumulative_nodes[1:])
        print("length split nodes")
        print(len(split_nodes))
        joined_nodes = np.concatenate([split_nodes[x] for x in randomized_idx])
        
        print(np.sum(full_num_nodes).astype(np.int64))
        print(np.sum(full_num_nodes, dtype=np.int64))

        full_num_nodes = full_num_nodes[randomized_idx]
        full_num_edges = full_num_edges[randomized_idx]
        full_phy_data      = full_phy_data[randomized_idx,:]
        full_node_attributes = joined_nodes
        full_edges_matrix = joined_edges
        full_aux_data      = full_aux_data[randomized_idx,:]
        full_labels_num    = full_labels_num[randomized_idx,:]
        full_labels_cat    = full_labels_cat[randomized_idx,:]
        full_idx_data      = full_idx_data[randomized_idx]
        full_ids      = unique_ids[randomized_idx]
        # reshape phylogenetic tensor data based on CPV+S
        full_phy_data.shape = (num_sample, -1, self.num_data_col)

        self.num_classes = len(np.unique(full_labels_cat))
        print("num classes = ", self.num_classes)
        print(len(full_node_attributes))
        print(full_num_nodes.dtype)
        print(np.sum(full_num_nodes, dtype=np.int64))


        for r in range(len(full_num_nodes)):
            if full_num_nodes[r] != full_num_edges[r] + 1:
                print(full_idx_data[r], full_num_nodes[r], full_num_edges[r])
                quit()

        # split dataset into training, test, validation, and calibration parts
        train_idx, val_idx, calib_idx = self.split_tensor_idx(num_sample)
        self.validate_tensor_idx(train_idx, val_idx, calib_idx)
        print("train idx: " + str(train_idx))
        print("LENGTH OF TRAINING DATA", len(train_idx))
        print("val idx: " + str(val_idx))
        print("calib idx: " + str(calib_idx))
        
        # save original training input
        self.train_label_true = full_labels[train_idx,:]
        self.train_label_index = full_idx_data[train_idx]

        # normalize auxiliary data
        norm_train_aux_data, train_aux_data_means, train_aux_data_sd = util.normalize(full_aux_data[train_idx,:])
        self.train_aux_data_mean_sd = (train_aux_data_means, train_aux_data_sd)
        norm_val_aux_data = util.normalize(full_aux_data[val_idx,:],
                                           self.train_aux_data_mean_sd)
        norm_calib_aux_data = util.normalize(full_aux_data[calib_idx,:],
                                             self.train_aux_data_mean_sd)
        
        
        #norm_train_node_attributes = util.normalize(full_node_attributes[train_idx])
        #norm_val_node_attributes = util.normalize(full_node_attributes[val_idx])
        #norm_calib_node_attributes = util.normalize(full_node_attributes[calib_idx])

        # normalize labels
        norm_train_labels_num, train_labels_num_means, train_labels_num_sd = util.normalize(full_labels_num[train_idx,:])
        self.train_labels_num_mean_sd = (train_labels_num_means, train_labels_num_sd)
        norm_val_labels_num = util.normalize(full_labels_num[val_idx,:],
                                              self.train_labels_num_mean_sd)
        self.norm_calib_labels_num = util.normalize(full_labels_num[calib_idx,:],
                                                     self.train_labels_num_mean_sd)

        # create phylogenetic data tensors
        train_phy_data_tensor = full_phy_data[train_idx,:,:]
        val_phy_data_tensor = full_phy_data[val_idx,:,:]
        self.calib_phy_data_tensor = full_phy_data[calib_idx,:,:]
        
        self.num_node_features = 1

        # train_node_attr_tensor = [torch.tensor(full_node_attributes[i]) for i in train_idx]
        # val_node_attr_tensor = [torch.tensor(full_node_attributes[i]) for i in val_idx]
        # calib_node_attr_tensor = [torch.tensor(full_node_attributes[i]) for i in calib_idx]
        train_node_attr_tensor = np.concatenate([split_nodes[randomized_idx[x]] for x in train_idx])
        val_node_attr_tensor = np.concatenate([split_nodes[randomized_idx[x]] for x in val_idx])
        calib_node_attr_tensor = np.concatenate([split_nodes[randomized_idx[x]] for x in calib_idx])



        train_edges_tensor = np.transpose(np.concatenate([split_edges[randomized_idx[x]] for x in train_idx]))
        val_edges_tensor = np.transpose(np.concatenate([split_edges[randomized_idx[x]] for x in val_idx]))
        calib_edges_tensor = np.transpose(np.concatenate([split_edges[randomized_idx[x]] for x in calib_idx]))
        


        # create categorical label tensors
        train_labels_cat = full_labels_cat[train_idx,:]
        val_labels_cat = full_labels_cat[val_idx,:]
        calib_labels_cat = full_labels_cat[calib_idx,:]
        calib_ids = full_ids[calib_idx]
        val_ids = full_ids[val_idx]

        # create index tensors
        train_idx_tensor = full_idx_data[train_idx,:]
        val_idx_tensor = full_idx_data[val_idx,:]
        calib_idx_tensor = full_idx_data[calib_idx,:]
        train_ids = full_ids[train_idx]
        train_num_nodes = full_num_nodes[train_idx]
        train_num_edges = full_num_edges[train_idx]
        calib_num_nodes = full_num_nodes[calib_idx]
        calib_num_edges = full_num_edges[calib_idx]
        val_num_nodes = full_num_nodes[val_idx]
        val_num_edges = full_num_edges[val_idx]

        print("training data dim")
        print(len(train_node_attr_tensor))
        print(np.sum(train_num_nodes, dtype=np.int64))

        print("ids:")
        print("train:", train_ids)
        print("val:", val_ids)
        print("calib:", calib_ids)
        print("training labels")
        print("label 0:", sum(train_labels_cat ==0))
        print("label 1:", sum(train_labels_cat ==1))
        print("label 2:", sum(train_labels_cat ==2))
        print("label 3:", sum(train_labels_cat ==3))

        print("validation labels")
        print(val_labels_cat)
        print("label 0:", sum(val_labels_cat ==0))
        print("label 1:", sum(val_labels_cat ==1))
        print("label 2:", sum(val_labels_cat ==2))
        print("label 3:", sum(val_labels_cat ==3))
        


        # torch datasets
        self.train_dataset = network.Dataset(train_phy_data_tensor, train_node_attr_tensor,
                                             train_edges_tensor, norm_train_aux_data,
                                             train_idx_tensor,
                                             norm_train_labels_num,
                                             train_labels_cat, train_ids, train_num_nodes, train_num_edges)
        self.calib_dataset = network.Dataset(self.calib_phy_data_tensor, calib_node_attr_tensor,
                                             calib_edges_tensor, norm_calib_aux_data,
                                             calib_idx_tensor,
                                             self.norm_calib_labels_num,
                                             calib_labels_cat, calib_ids, calib_num_nodes, calib_num_edges)
        self.val_dataset   = network.Dataset(val_phy_data_tensor, val_node_attr_tensor,
                                             val_edges_tensor, norm_val_aux_data,
                                             val_idx_tensor,
                                             norm_val_labels_num,
                                             val_labels_cat, val_ids, val_num_nodes, val_num_edges)
        return

    def separate_labels(self, labels):
        """Separates labels for categorical param_est targets.
        
        This function separates labels into numerical and categorical subsets
        based on the param_est dictionary.
        
        Args:
            labels (numpy.ndarray): The input labels.
            
        Returns:
            labels_num (numpy.ndarray): The numerical-valued labels.
            labels_cat (numpy.ndarray): The categorical labels.
        
        """

        idx_num = list()
        idx_cat = list()
        
        for k,v in self.param_est.items():
            if v == 'cat':
                self.has_label_cat = True
                idx = self.label_names.index(k)
                unique_cats, encoded_cats = np.unique(labels[:,idx],
                                                      return_inverse=True)
                print(unique_cats)
                self.param_cat[k] = len(unique_cats)
                labels[:,idx] = encoded_cats
                idx_cat.append( idx )
                self.param_cat_names.append(k)
                
            elif v == 'num':
                self.has_label_num = True
                # print(self.label_names)
                idx_num.append( self.label_names.index(k) )
                self.param_num_names.append(k)
        
        if not self.has_label_num and not self.has_label_cat:
            util.print_err(f"No training labels found.", exit=True)
               
        # get data subsets
        labels_num = labels[:,idx_num].copy()
        labels_cat = labels[:,idx_cat].copy()

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
        #self.model = CreateModel()
        if self.use_cuda:
            #self.model = torch.nn.DataParallel(self.model)
            self.model = DataParallel(self.model)
            # self.model = DDP(self.model)
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
        # training dataset
        train_loader = DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.trn_batch_size, collate_fn = custom_collate)
        val_loader = DataLoader(dataset=self.val_dataset,
                                                   batch_size=self.trn_batch_size, collate_fn = custom_collate)

        print("training batch size: ", str(self.trn_batch_size))
        print("graph shape",len(self.train_dataset.graph_dat))
        n_train_graphs = len(self.train_dataset.graph_dat)
        n_val_graphs = len(self.val_dataset.graph_dat)
        n_calib_graphs = len(self.calib_dataset.graph_dat)

        num_batches = int(np.ceil(n_train_graphs / self.trn_batch_size))
        val_num_batches = int(np.ceil(n_val_graphs / self.trn_batch_size))
        calib_num_batches = int(np.ceil(n_calib_graphs / self.trn_batch_size))

        print("number of batches: " + str(num_batches))
        # for data in train_loader:
        #     print(f"Batch assignment: {data.batch}")  # Should be [0,0,...,1,1,...,2,2,...]
        #     print(f"Unique batch IDs: {data.batch.unique()}")  # Should match number of graphs
        #     break
        # for data in val_loader:
        #     print(f"Val Batch assignment: {data.batch}")  # Should be [0,0,...,1,1,...,2,2,...]
        #     print(f"Val Unique batch IDs: {data.batch.unique()}")  # Should match number of graphs
        #     break

            # print(data)
            # print()

        # validation dataset

        #val_phy_dat  = torch.Tensor(self.val_dataset.phy_data).to(self.TORCH_DEVICE)
        #val_aux_dat  = torch.Tensor(self.val_dataset.aux_data).to(self.TORCH_DEVICE)
        val_idx_dat  = torch.Tensor(self.val_dataset.idx_data).to(self.TORCH_DEVICE)
        val_lbl_num  = torch.Tensor(self.val_dataset.labels_num).to(self.TORCH_DEVICE)
        val_lbl_cat  = torch.Tensor(self.val_dataset.labels_cat).to(self.TORCH_DEVICE)
        # print("VALIDATION GRAPH DATA")
        # print(self.val_dataset.graph_data)
        #val_graph_dat = self.val_dataset.graph_dat.to(self.TORCH_DEVICE)
        val_graph_dat = [self.val_dataset.graph_dat[i].to(self.TORCH_DEVICE) for i in range(len(self.val_dataset.graph_dat))]
        val_bad_count = 0

        #print("val_node_data")
        #print(self.val_dataset.node_data)

        #val_node_data = [torch.Tensor(z).to(self.TORCH_DEVICE) for z in self.val_dataset.node_data]
        #val_edges_data = [[[torch.Tensor(z).to(self.TORCH_DEVICE) for z in y] for y in x] for x in self.val_dataset.edges_data]

        # model device
        # self.model.to(self.TORCH_DEVICE)

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
        metric_names = ['loss_lower', 'loss_upper', 'loss_value',
                        'loss_combined', 'mse_value', 'mae_value', 'medape_value', 'accuracy_combined']
        prev_trn_loss_combined = None
        prev_val_loss_combined = None
        prev_trn_acc_combined = None
        prev_val_acc_combined = None

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

        for i in range(self.num_epochs):
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
            trn_mse_value = 0.
            trn_mape_value = 0.
            trn_mae_value = 0.

            train_msg = f'Training epoch {i+1} of {self.num_epochs}'
            correct = 0
            total_graphs = 0
            self.model.train()
            accumulation_steps = 30

            # (phy_dat, graph_dat, aux_dat, idx_dat, lbl_num, lbl_cat) = next(iter(train_loader))
            # for epoch in range(300):
            optimizer.zero_grad()
            #     logits = self.model(phy_dat, graph_dat, aux_dat)
            #     preds = logits[3]
            #     loss = loss_categ_func(preds, lbl_cat.flatten())
            #     loss.backward()
            #     optimizer.step()
            #     if epoch % 30 == 0:
            #         print(epoch, loss.item())
            # quit()

#             for j, (phy_dat, graph_dat, aux_dat, idx_dat, lbl_num, lbl_cat) in tqdm(enumerate(train_loader),

            for j, (graph_dat, idx_dat, lbl_num, lbl_cat) in tqdm(enumerate(train_loader),
                                                                    total=num_batches,
                                                                    desc=train_msg,
                                                                    smoothing=0):
                                            
                    
                    # short cut batches for training
                    # if j > 1:
                    #     break

                    # print("lbl_cat")
                    # print(lbl_cat)
                    
                    # send labels to device
                    #phy_dat = phy_dat.to(self.TORCH_DEVICE)
                    #aux_dat = aux_dat.to(self.TORCH_DEVICE)
                for g in graph_dat:
                    if (g.x.shape[0] != g.edge_index.shape[1] + 1):
                        print(g)
                if j > 1200:
                    print(graph_dat)
                #     print("j:", j)
                #     print(len(idx_dat))
                #     print(len(lbl_num))
                #     print(len(lbl_cat))
                #     print(len(graph_dat))
                idx_dat = idx_dat.to(self.TORCH_DEVICE)
                lbl_num = lbl_num.to(self.TORCH_DEVICE)
                lbl_cat = lbl_cat.to(self.TORCH_DEVICE)
                    #graph_dat = graph_dat.to(self.TORCH_DEVICE)



                    # forward pass of training data to estimate labels
                    # print("graphs :)")
                    # print(graph_dat)
                for k in range(1):
                    try:
                        lbls_hat = self.model(graph_dat)
                    except Exception as e:
                        print("exception in lbls_hat, j:", j, e)
                        quit()

                    # print("params")
                    # if j == 1:
                    #     for name, p in self.model.named_parameters():
                    #         print(name, p)
                    #         if p.grad is not None:
                    #             print(name, p.grad.abs().max())
                    # if i > 2:
                    #     quit()
                    third_arg = lbls_hat[3] # [0:int(len(lbls_hat[3])/4)]
                    preds = third_arg #.long()
                    # print(f"Output stats - Mean: {third_arg.mean():.4f}, Std: {third_arg.std():.4f}")
                    # print(f"Output range: [{third_arg.min():.4f}, {third_arg.max():.4f}]")
                    # print(f"Unique values: {torch.unique(third_arg).numel()}")
                    # correct += int((preds.argmax(dim=1).flatten() == lbl_cat.flatten()).sum())
                    # total_graphs = total_graphs + len(lbl_cat.flatten())
                    #if j == 1:
                    if j % 5 == 0: # % 150
                        amax = preds.argmax(dim=1)
                        # print("pred val:\t\t", amax) 
                        # print("actual labels:\t\t", lbl_cat.flatten())
                        actual_labels = lbl_cat.flatten()
                        selected_labels_1 = actual_labels == 1
                        # print(amax[selected_labels_1])
                        # print(actual_labels[selected_labels_1])
                        selected_labels_0 = actual_labels == 0
                        # print("number correct:\t\t", int((amax == lbl_cat.flatten()).sum()), "\twhere actual class = 1:", int((amax[selected_labels_1] == actual_labels[selected_labels_1]).sum()), "\twhere actual class = 0:", int((amax[selected_labels_0] == actual_labels[selected_labels_0]).sum()))

                    #print("most recent out:")
                    #print(preds)
                    # print("labels:")
                    # print(lbl_cat.flatten())
                    loss_list = list()
                    if self.has_label_num:
                        loss_value = loss_value_func(lbls_hat[0], lbl_num)
                        loss_lower = loss_lower_func(lbls_hat[1], lbl_num)
                        loss_upper = loss_upper_func(lbls_hat[2], lbl_num)
                        loss_list += [ loss_value, loss_lower, loss_upper ]
                    if self.has_label_cat:
                        loss_categ = loss_categ_func(preds, lbl_cat.flatten())
                        # print("preds")
                        #p = torch.sigmoid(preds)
                        #predicted_class = (p >= 0.5).long()
                        # print("truth:", lbl_cat.flatten())
                        #acc_categ = int((predicted_class.flatten() == lbl_cat.flatten()).sum())/len(lbl_cat) #BCE
                        amax = preds.argmax(dim=1)
                        # print("pred class")
                        # print(amax)
                        #print("matches")
                        #print(amax == lbl_cat.flatten())
                        train_correct += int((amax == lbl_cat.flatten()).sum())
                        train_length += len(amax)
                        # print("correct:", train_correct)
                        #print("percent")
                        #print(acc_categ)

                        loss_list += [ loss_categ ]
                    # loss_combined = torch.stack(loss_list).sum()
                    
                    if loss_aggregation == 'sum':
                        loss_combined = torch.stack(loss_list).sum()
                    elif loss_aggregation == 'geometric':
                        loss_combined = torch.exp(torch.mean(torch.log(torch.stack(loss_list))))
                    elif loss_aggregation == 'median':
                        loss_combined = torch.median(torch.stack(loss_list))
                    # collect history stats
                    if self.has_label_num:
                        trn_loss_value    += loss_value.item() / num_batches
                        trn_loss_lower    += loss_lower.item() / num_batches
                        trn_loss_upper    += loss_upper.item() / num_batches
                        trn_mse_value     += (torch.mean((lbl_num - lbls_hat[0])**2)).item() / num_batches
                        trn_mae_value     += (torch.mean(torch.abs(lbl_num - lbls_hat[0]))).item() / num_batches
                        trn_mape_value    += 100. * (torch.median(torch.abs((lbl_num - lbls_hat[0])/lbl_num))).item() / num_batches
                    trn_loss_combined += loss_combined.item() / num_batches
                
                
                # backward pass to update gradients
                #loss_combined.requires_grad = True #KT
                
                loss_combined.backward()

                # self.model.train()
                # update network parameters
                if ((j + 1) % accumulation_steps == 0 or (j + 1) == num_batches):
                    #print("updating optimizer")
                    optimizer.step()

                    # reset gradients for tensors
                    optimizer.zero_grad()
            #scheduler.step()
            #print(scheduler.get_last_lr())
            
            trn_acc_combined += train_correct / train_length
            # print("total correct:", train_correct, " out of ", train_length)

            train_metric_vals = [ trn_loss_lower, trn_loss_upper, trn_loss_value,
                                  trn_loss_combined, trn_mse_value,
                                  trn_mae_value, trn_mape_value, trn_acc_combined ]


            trn_loss_str = f'    Train        --   loss: {"{0:.4f}".format(trn_loss_combined)}\t'
            trn_acc_str = f'--   acc: {"{0:.4f}".format(trn_acc_combined)}'

            # print("pre model eval")

            self.model.eval()
            with torch.no_grad():

                for k in range(1):
                    correct = 0
                    total_graphs = 0
                    # print("pre val loader")
                    #for j, (val_phy_dat, val_graph_dat, val_aux_dat, val_idx_dat, val_lbl_num, val_lbl_cat) in tqdm(enumerate(val_loader),
                    for j, (val_graph_dat, val_idx_dat, val_lbl_num, val_lbl_cat) in tqdm(enumerate(val_loader),
                                                                    total=val_num_batches,
                                                                    desc=train_msg,
                                                                    smoothing=0):
                    
                    #val_phy_dat = val_phy_dat.to(self.TORCH_DEVICE)
                    #val_aux_dat = val_aux_dat.to(self.TORCH_DEVICE)
                        val_idx_dat = val_idx_dat.to(self.TORCH_DEVICE)
                        val_lbl_num = val_lbl_num.to(self.TORCH_DEVICE)
                        val_lbl_cat = val_lbl_cat.to(self.TORCH_DEVICE)
                        # print("VALIDATION GRAPH DATA")
                        # print(val_idx_dat)
                        # print(val_lbl_num)
                        # print(val_graph_dat)
                    #val_graph_dat = val_graph_dat.to(self.TORCH_DEVICE)

                    # reset gradients for tensors

                    # forward pass of training data to estimate labels


                    # # forward pass of validation to estimate labels
                        # print("pre val model")
                        try:
                            val_lbls_hat       = self.model(val_graph_dat)
                        except Exception as e:
                            print("val exception in lbls_hat, j:", j, e)
                            quit()
                        
                        #self.model(val_phy_dat, val_graph_dat, val_aux_dat)


                        #print("val lbls hat 3")
                        #print(val_lbls_hat[3])
                        third_arg = val_lbls_hat[3]#[0:int(len(lbls_hat[3])/4)]
                        # print(" val out:")
                        # print(third_arg.flatten())
                        # print(f"Val Output stats - Mean: {third_arg.mean():.4f}, Std: {third_arg.std():.4f}")
                        # print(f"Val Output range: [{third_arg.min():.4f}, {third_arg.max():.4f}]")
                        # print(f"Val Unique values: {torch.unique(third_arg).numel()}")
                        # collect validation metrics
                        val_loss_list = list()
                        val_loss_value = 0.
                        val_loss_lower = 0.
                        val_loss_upper = 0.
                        # correct += int((third_arg.argmax(dim=1).flatten() == val_lbl_cat.flatten()).sum())
                        # total_graphs = total_graphs + len(val_lbl_cat.flatten())

                        if self.has_label_num:
                            val_loss_value = loss_value_func(val_lbls_hat[0], val_lbl_num).item()
                            val_loss_lower = loss_lower_func(val_lbls_hat[1], val_lbl_num).item()
                            val_loss_upper = loss_upper_func(val_lbls_hat[2], val_lbl_num).item()
                            val_loss_list += [ val_loss_value, val_loss_lower, val_loss_upper ]
                        # val_loss_combined  = val_loss_value + val_loss_lower + val_loss_upper
                        if self.has_label_cat:
                            val_loss_categ = loss_categ_func(third_arg, val_lbl_cat.flatten()).item()
                            val_loss_list += [ val_loss_categ ]
                            #p = torch.sigmoid(third_arg)
                            #predicted_class = (p >= 0.5).long()
                            # print("truth:", lbl_cat.flatten())
                            # val_acc_combined = int((predicted_class.flatten() == val_lbl_cat.flatten()).sum())/len(val_lbl_cat) #BCE
                            pred_class = third_arg.argmax(dim=1) 
                            if j % 60 == 0:
                                # print("pred val:\t\t", pred_class) 
                                # print("actual labels:\t\t", val_lbl_cat.flatten())
                                actual_labels = val_lbl_cat.flatten()
                                selected_labels_1 = actual_labels == 1
                                selected_labels_0 = actual_labels == 0

                                # print("number correct:\t\t", int((pred_class == val_lbl_cat.flatten()).sum()), "\twhere actual class = 1:", int((pred_class[selected_labels_1] == actual_labels[selected_labels_1]).sum()), "\twhere actual class = 0:", int((pred_class[selected_labels_0] == actual_labels[selected_labels_0]).sum()))

                                # print(pred_class[selected_labels])
                                # print(actual_labels[selected_labels])
                            val_correct += int((pred_class == val_lbl_cat.flatten()).sum())
                                # print("correct:", int((pred_class.flatten() == val_lbl_cat.flatten()).sum()))
                            val_length += len(val_lbl_cat)

                        # val_loss_combined = sum(val_loss_list)
                        if loss_aggregation == 'sum':
                            # val_loss_combined = torch.stack(val_loss_list).sum()
                            val_loss_combined = np.sum(val_loss_list)
                        elif loss_aggregation == 'geometric':
                            val_loss_combined = np.exp(np.mean(np.log(val_loss_list)))
                            # val_loss_combined = torch.exp(torch.mean(torch.log(torch.stack(val_loss_list))))
                        elif loss_aggregation == 'median':
                            # val_loss_combined = torch.median(torch.stack(val_loss_list))
                            val_loss_combined = np.median(val_loss_list)
                            
                        val_mse_value = 0.
                        val_mae_value = 0.
                        val_mape_value = 0.
                        if self.has_label_num:
                            val_mse_value      = (torch.mean((val_lbl_num - val_lbls_hat[0])**2)).item()
                            val_mae_value      = (torch.mean(torch.abs(val_lbl_num - val_lbls_hat[0]))).item()
                            val_mape_value     = 100. * (torch.median(torch.abs((val_lbl_num - val_lbls_hat[0]) / val_lbl_num))).item()



                    print("total correct:", val_correct, " out of ", val_length)
                    val_acc_combined = val_correct/val_length  
                    val_metric_vals = [ val_loss_value, val_loss_lower, val_loss_upper,
                                            val_loss_combined, val_mse_value, val_mae_value,
                                            val_mape_value, val_acc_combined ]
                    # acc = correct/(total_graphs)
                    # print("correct: ", str(correct), "graphs: ", str(total_graphs), "accuracy: ", str(acc))
                # raw training metrics for epoch
                    val_loss_str = f'    Validation   --   loss: {"{0:.4f}".format(val_loss_combined)}\t'
                    val_acc_str = f'--   acc: {"{0:.4f}".format(val_acc_combined)}'

            
            #scheduler.step(val_loss_combined)
            #current_lr = optimizer.param_groups[0]['lr']
            #print(f"LR: {current_lr:.6f}")

            # changes in training metrics between epochs
                if i > 0:
                    diff_trn_acc = trn_acc_combined - prev_trn_acc_combined
                    diff_val_acc = val_acc_combined - prev_val_acc_combined
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
                    if (prev_trn_acc_combined == 0):
                        rat_trn_acc = 100
                    else:
                        rat_trn_acc  = 100 * round(trn_acc_combined / prev_trn_acc_combined - 1.0, ndigits=4)
                    if (prev_val_acc_combined == 0):
                        rat_val_acc = 100
                    else:
                        rat_val_acc  = 100 * round(val_acc_combined / prev_val_acc_combined - 1.0, ndigits=4)
                    
                    diff_trn_loss_str = '{0:+.4f}'.format(diff_trn_loss)
                    diff_val_loss_str = '{0:+.4f}'.format(diff_val_loss)
                    rat_trn_loss_str  = '{0:+.2f}'.format(rat_trn_loss).rjust(4, ' ')
                    rat_val_loss_str  = '{0:+.2f}'.format(rat_val_loss).rjust(4, ' ')

                    diff_trn_acc_str = '{0:+.4f}'.format(diff_trn_acc)
                    diff_val_acc_str = '{0:+.4f}'.format(diff_val_acc)
                    rat_trn_acc_str  = '{0:+.2f}'.format(rat_trn_acc).rjust(4, ' ')
                    rat_val_acc_str  = '{0:+.2f}'.format(rat_val_acc).rjust(4, ' ')
                
                    trn_color = 31 if diff_trn_loss >= 0 else 32  # green or red
                    val_color = 31 if diff_val_loss >= 0 else 32  # green or red
                    trn_loss_change_str  = f'  abs: {util.phyddle_str(diff_trn_loss_str, style=0, color=trn_color)}'
                    trn_loss_change_str += f'  rel: {util.phyddle_str(rat_trn_loss_str, style=0, color=trn_color)}%'
                    val_loss_change_str  = f'  abs: {util.phyddle_str(diff_val_loss_str, style=0, color=val_color)}'
                    val_loss_change_str += f'  rel: {util.phyddle_str(rat_val_loss_str, style=0, color=val_color)}%'

                    trn_acc_change_str  = f'  abs: {util.phyddle_str(diff_trn_acc_str, style=0, color=trn_color)}'
                    trn_acc_change_str += f'  rel: {util.phyddle_str(rat_trn_acc_str, style=0, color=trn_color)}%'
                    val_acc_change_str  = f'  abs: {util.phyddle_str(diff_val_acc_str, style=0, color=val_color)}'
                    val_acc_change_str += f'  rel: {util.phyddle_str(rat_val_acc_str, style=0, color=val_color)}%'
                    trn_loss_str += trn_loss_change_str
                    val_loss_str += val_loss_change_str

                    trn_acc_str += trn_acc_change_str
                    val_acc_str += val_acc_change_str

                    if diff_val_loss >= 0:
                        val_bad_count += 1
                    else:
                        val_bad_count = 0

                prev_trn_loss_combined = trn_loss_combined
                prev_val_loss_combined = val_loss_combined

                prev_trn_acc_combined = trn_acc_combined
                prev_val_acc_combined = val_acc_combined
                # display training metric progress
                print(trn_loss_str, trn_acc_str)
                print("")
                print(val_loss_str, val_acc_str)
                print('')

            # update train history log
            self.update_train_history(i, metric_names, train_metric_vals, 'train')
            self.update_train_history(i, metric_names, val_metric_vals, 'validation')
            path_prefix = f'{self.trn_dir}/{self.trn_prefix}'
            model_history_fn = f'{path_prefix}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_history.csv'
            self.train_history.to_csv(model_history_fn, index=False, sep=',',
                                  float_format=util.PANDAS_FLOAT_FMT_STR)

            checkpoint = {'state_dict': self.model.state_dict(),
                            'optimizer': optimizer.state_dict(),}
            save_checkpoint(checkpoint, filename=f'{self.trn_dir}'+"/checkpoint.tar")

      
            # early stopping
            if val_bad_count >= self.num_early_stop and self.num_early_stop > 0 and i < 15:
                print(f'Early stop: validation loss increased for num_early_stop={self.num_early_stop} consecutive epochs')
                break

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

        calib_loader = DataLoader(dataset=self.calib_dataset,
                                                   batch_size=self.trn_batch_size, collate_fn = custom_collate)#num_calib_examples)

        for step, data in enumerate(calib_loader):
            print(f'Calib step {step + 1}:')
            print()
        calib_batch = next(iter(calib_loader))
        calib_graph_dat, calib_idx_dat, calib_lbl_num, calib_lbl_cat = calib_batch
        #calib_phy_dat, calib_graph_dat, calib_aux_dat = calib_batch[0], calib_batch[1], calib_batch[2], third_arg

        # calib_phy_dat = calib_phy_dat.to('cpu')
        # calib_aux_dat = calib_aux_dat.to('cpu')
        
        # get calib estimates
        print("----------------\ncalib est\n\n")

        #calib_label_est = self.model(calib_phy_dat, calib_graph_dat, calib_aux_dat)
        try:
            calib_label_est = self.model(calib_graph_dat)
        except Exception as e:
            print("calib exception", e)
            quit()

        # make CPI adjustments
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

        print("make results function")
        
        # get uncalibrated estimates
        # training label estimates
        print("dataset length", len(self.train_dataset))
        train_loader = DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.trn_batch_size, collate_fn = custom_collate)#num_train_examples)

        # for step, data in enumerate(train_loader):
        #     print(f'Train step {step + 1}:')
        #     #print(data)
                                      

        #train_batch = next(iter(train_loader))
        # for j, (graph_dat, idx_dat, lbl_num, lbl_cat) in tqdm(enumerate(train_loader),
        for j, (train_graph_dat, train_idx_dat, train_labels_num, train_labels_cat) in enumerate(train_loader):
            # print("j = ", j)

            
            #train_phy_dat = train_phy_dat.to(self.TORCH_DEVICE)
            #train_aux_dat = train_aux_dat.to(self.TORCH_DEVICE)
            train_labels_num = train_labels_num.to(self.TORCH_DEVICE)
            train_labels_cat = train_labels_cat.to(self.TORCH_DEVICE)
            #train_graph_dat = train_graph_dat.to(self.TORCH_DEVICE)

            #train_nodes_dat = [z.to(self.TORCH_DEVICE) for z in train_nodes_dat]
            #train_edges_dat = [[[z.to(self.TORCH_DEVICE) for z in y] for y in x] for x in train_edges_dat]
            # NOTE: train_idx[0:1000] will be equal to self.train_label_index[0:1000,:]
            #       because DataLoader batches are indexed in same order
            
            # get train estimates
            label_est = self.model(train_graph_dat)
            #label_est = self.model(train_phy_dat, train_graph_dat, train_aux_dat)

        
            # numerical vs. cat estimates
            labels_num_est = label_est[0:3]
            labels_num_est = torch.stack(labels_num_est).cpu().detach().numpy()
            labels_cat_est = label_est[3]



            if self.has_label_num:
                train_labels_num = train_labels_num.cpu().detach().numpy()
                if j == 0:
                    self.train_label_num_true = train_labels_num.copy()
                else:
                    self.train_label_num_true = np.append(self.train_label_num_true, train_labels_num)
                
                # uncalibrated training estimates of numerical labels
                # self.train_label_num_est = labels_num_est.copy()
                if j == 0:
                    self.train_label_num_est = labels_num_est
                else:
                    self.train_label_num_est = np.append(self.train_label_num_est, labels_num_est)
                # self.train_label_num_est = util.denormalize(labels_num_est.copy(),
                #                                             self.train_labels_num_mean_sd)
                

                # generate calibration factors

            if self.has_label_cat:
                train_label_cat = train_labels_cat.cpu().detach().numpy().astype('int')
                if j == 0:
                    # print("j =", j, "cat true is none")
                    self.train_label_cat_true = train_label_cat.copy()
                else:
                    self.train_label_cat_true = np.append(self.train_label_cat_true, train_label_cat)
                # print("UPDATE:")
                # print(self.train_label_cat_true )
                if j == 0:
                    self.train_label_cat_est = labels_cat_est.argmax(dim=1).cpu().detach().numpy()
                else:
                    self.train_label_cat_est = np.append(self.train_label_cat_est, labels_cat_est.argmax(dim=1).cpu().detach().numpy())
                total_correct = (self.train_label_cat_true == self.train_label_cat_est).sum()
                # print("accuracy:")
                # print(total_correct/len(self.train_label_cat_true))
            else:
                print("no label cat")
        self.perform_cpi_calibration()

            # calibrate original estimates
        if self.has_label_num:    
            labels_num_est_calib = self.train_label_num_est.copy()
            labels_num_est_calib[1,:,:] = labels_num_est_calib[1,:,:] + self.cpi_adjustments[0,:]
            labels_num_est_calib[2,:,:] = labels_num_est_calib[2,:,:] + self.cpi_adjustments[1,:]
            
        # # denormalize calibrated estimates
        # if self.train_label_num_est_calib == None:
            self.train_label_num_est_calib = labels_num_est_calib
        # else:
        #     self.train_label_num_est_calib = np.append(self.train_label_num_true, labels_num_est_calib)
        # self.train_label_num_est_calib = labels_num_est_calib

        # reformat categorical estimates, if they exist
        
        print("returning from make results")
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
        model_arch_fn                = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.trained_model.pkl'
        model_history_fn             = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_history.csv'
        model_cpi_fn                 = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.cpi_adjustments.csv'
        # model_weights_fn           = f'{path_prefix}.train_weights.hdf5'
        
        # output scaling terms
        train_labels_num_norm_fn    = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_norm.labels_num.csv'
        train_aux_data_norm_fn       = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_norm.aux_data.csv'

        # output training labels
        train_label_num_true_fn     = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_true.labels_num.csv'
        train_label_num_est_fn      = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_est.labels_num.csv'
        train_label_est_nocalib_fn   = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_est.labels_num_nocalib.csv'
        train_label_cat_true_fn      = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_true.labels_cat.csv'
        train_label_cat_est_fn       = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.train_est.labels_cat.csv'
        
        # save model to file
        torch.save(self.model.module, model_arch_fn)
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
 
        if self.has_label_num:
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
