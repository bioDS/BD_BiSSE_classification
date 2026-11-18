#!/usr/bin/env python
"""
train
=====
Defines classes and methods for the Training step, which builds and trains a
network using the tensor data from the Formatting step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2025, Michael Landis and Ammon Thompson
License:   MIT

# Edits for BD / BiSSE / MuSSE_3 / MuSSE_4 / MuSSE_5 classification:
#  - Classes are the model_type values present in the data (any subset of
#    0..4). They are mapped to contiguous indices 0..K-1 for the network, and
#    mapped back to model_type values when results are saved.
#  - Numerical labels are matched by name, not position. Per-state rates
#    (log10_birth_rate_tI / log10_death_rate_tI) are filled from the BD rates
#    (log10_birth_rate / log10_death_rate) for BD trees. Rates a model does not
#    have (e.g. state 5 on a BiSSE tree, log10_trans_rate on a BD tree) are NaN
#    and are left out of the normalisation statistics. With only BD and BiSSE
#    this gives the same four columns as before.
"""

# standard imports
import os
import re
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
import copy
import json
import time

# phyddle imports
from phyddle import utilities as util
from phyddle import network
from phyddle.subtree_partition import node_groups
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoLoader, DataListLoader
from torch.utils.data import DataLoader, Subset
from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.data import Batch, Data as GeoData
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch import linalg as LA
from torch_geometric.utils import to_undirected

# GPU memory fraction for this process (the sweep sets PHYDDLE_GPU_MEM_FRACTION
# when several training runs share a GPU); default 0.8 as before
GPU_MEM_FRACTION = float(os.environ.get('PHYDDLE_GPU_MEM_FRACTION', '0.8'))
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION, device=0)
np.random.seed(42)


# dist.init_process_group(backend='nccl')
# local_rank = int(os.environ['LOCAL_RANK'])
# device = torch.device(f'cuda:{local_rank}')

##################################################
# Model classes and numerical labels
##################################################

# model_type value -> name (matches PACKED_MODELS in format.py)
MODEL_TYPE_NAMES = {0: 'BD', 1: 'BiSSE', 2: 'MuSSE_3', 3: 'MuSSE_4', 4: 'MuSSE_5'}

# numerical labels at or above this value mean "not defined for this model"
# (format.py writes 10000000)
LABEL_PLACEHOLDER_MIN = 1e6

_BIRTH_T_RE = re.compile(r'^log10_birth_rate_t(\d+)$')
_DEATH_T_RE = re.compile(r'^log10_death_rate_t(\d+)$')


def model_type_name(v):
    """Readable name for a model_type value."""
    return MODEL_TYPE_NAMES.get(int(v), f'class {int(v)}')


# model_type value -> number of character states of that model
MODEL_TYPE_STATES = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}


def parse_max_model_states(v):
    """Config value of max_model_states -> int, or None if not set
    (None, '', 'none', 'all' or -1). Same rule as format.py."""
    if v is None or str(v).strip().lower() in ('', 'none', 'na', 'all', '-1'):
        return None
    k = int(v)
    if k < 1:
        raise ValueError(f'max_model_states must be 1 or more (got {v})')
    return k


def _flag(v, default=False):
    """Config flag -> bool ('T', 'True', True, 1 -> True; 'F', False -> False)."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().upper() in ('T', 'TRUE', '1', 'YES', 'Y')


def _torch_load(fn):
    """torch.load that also reads the non-tensor training state
    (newer torch versions default to weights_only=True)."""
    try:
        return torch.load(fn, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(fn, map_location='cpu')


def _to_jsonable(x):
    """numpy / torch values -> plain Python for json.dump."""
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if torch.is_tensor(x):
        return _to_jsonable(x.detach().cpu().tolist())
    return x

##################################################

#Code edited from ChatGPT to read data into memory in batches
class HDF5BlockDataset():
    def __init__(self, file_path, blocks, trainer = None, n_parts = 8):
        self.file_path = file_path
        self.trainer = trainer
        self.blocks = blocks
        self.n_parts = n_parts
        self._part_cache = {}          # graph index -> LongTensor of subtree ids

        # Read columns in from hdf5 storage file
        with h5py.File(file_path, 'r') as f:
            self.idx = f["idx"][:]
            self.num_edges = f["num_edges"][:]
            num_nodes = np.asarray(f["num_nodes"][:], dtype=np.int64)
            label_names = f["label_names"][:].squeeze()
            label_names = [x.decode("utf-8") for x in np.atleast_1d(label_names)]
            self.label_names = label_names
            self.labels = f["labels"][:]
            self.aux_data = f["aux_data"][:]
            self.aux_data_names = [x.decode("utf-8") for x in f["aux_data_names"][:].ravel()]        # full_labels_num, full_labels_cat = self.separate_labels(self.labels)
        
        # To divide one large graph in sub-graphs we need to know where each graph starts and stops for
        # stored edges and stored nodes.
        self.offsets = np.zeros(len(self.idx)+1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.num_edges, dtype=np.int64)
        self.node_offsets = np.zeros(len(self.idx)+1, dtype=np.int64)
        self.node_offsets[1:] = np.cumsum(num_nodes)

        self.file = None
        self.total_graphs = 0
    
    def __len__(self):
        return len(self.blocks)
    
      
    def adjust_num_labels(self, labels_num):
        """Convert the numerical param_est columns into a fixed set of
        per-state columns shared by all models.

        Columns are matched by name (see CnnTrainer.build_num_label_plan):
          - log10_birth_rate_t1..tK, then log10_death_rate_t1..tK, where K is
            the largest state number in param_est. A tree's own per-state rate
            is used when defined; otherwise the BD rate (log10_birth_rate /
            log10_death_rate) is used, so BD trees repeat the same value
            across states, as before.
          - any other numerical labels (e.g. log10_trans_rate) as they are.
        Values that are not defined for a tree's model (placeholders of
        10000000 from format.py, e.g. state 5 on a BiSSE tree or the
        transition rate on a BD tree) become NaN.

        With only BD and BiSSE labels this returns the same four columns as
        before: birth_t1, birth_t2, death_t1, death_t2.
        """
        trainer = self.trainer
        if trainer.num_label_plan is None:
            trainer.build_num_label_plan()
        plan = trainer.num_label_plan

        x = labels_num.to(torch.float32)
        nan = float('nan')
        x = torch.where(x < LABEL_PLACEHOLDER_MIN, x, torch.full_like(x, nan))

        n = x.shape[0]
        adjusted_labels = torch.full((n, len(plan)), nan,
                                     dtype=torch.float32, device=x.device)
        for j, (_, primary, fallback) in enumerate(plan):
            col = torch.full((n,), nan, dtype=torch.float32, device=x.device)
            if fallback is not None:
                col = x[:, fallback]
            if primary is not None:
                col = torch.where(torch.isfinite(x[:, primary]), x[:, primary], col)
            adjusted_labels[:, j] = col
        return adjusted_labels

    def __getitem__(self, i):
        # Read in necessary columns based on file path
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')        
            self.phy_data = self.file["phy_data"]
            self.graph_id = self.file["graph_id"]
            self.node_1 = self.file["node_1"].astype(np.int64)
            self.node_2 = self.file["node_2"].astype(np.int64)
            self.node_attr = self.file["node_attr"]

        graph_indices = self.blocks[i]
        self.total_graphs += len(graph_indices)
        graph_indices = np.sort(graph_indices)
        phy_data = torch.from_numpy(self.phy_data[graph_indices].astype(np.float32))
        labels = self.labels[graph_indices]
       
        # Sort numerical and categorical labels into different groups.
        # separate labels might not be robust if entire block was same label
        labels_num, labels_cat = self.trainer.separate_labels(labels, self.label_names)
        # Convert numerical labels to one set of per-state columns so all
        # models (BD, BiSSE, MuSSE) have the same dimension.
        labels_num = self.adjust_num_labels(labels_num)
        # model_type values (0..4) -> contiguous class indices (0..K-1)
        labels_cat_raw = torch.round(labels_cat).long()
        labels_cat = self.trainer.map_classes(labels_cat_raw)
        labels_num = labels_num.float()
        

        aux_data = torch.from_numpy(self.aux_data[graph_indices].astype(np.float32))

        node_1_list = []
        node_2_list = []
        node_attr_list = []
        graph_id_list = []

        graph_list = []
        ind_count = 0

        node_counts = np.diff(self.node_offsets)

        # Find stored sub-graphs for each (sub)graph in this block
        for g in graph_indices:
            s = self.offsets[g]
            e = self.offsets[g + 1]
            s_nodes = self.node_offsets[g]
            idx = torch.as_tensor(self.idx[g], dtype=torch.long)
            e_nodes = self.node_offsets[g+1]
            edge_index =  torch.from_numpy(np.stack([self.node_1[s:e], self.node_2[s:e]],axis=0).astype(np.int64))
            try:
                min_index =   edge_index.min()
            except:
                print("Exception in parsing graphs")
                print(s, e)
                print(self.node_1[s:e].shape)
                print(self.node_2[s:e].shape)
                print(edge_index.shape)
                quit()

            # Find nodes which are connected in this (sub)graph
            edge_nodes = np.unique(
                    np.concatenate([self.node_1[s:e], self.node_2[s:e]])
            )

            # node numbers may not start at zero for this subgraph, so re-label edges and nodes accordingly
            edge_index = torch.sub(edge_index, min_index)
            unique = torch.unique(edge_index)
            all = torch.arange(edge_index.max()+1)
            difference = all[torch.isin(all, unique, invert=True)]
            reduction = torch.searchsorted(difference, all, right=False)
            edge_index=  torch.sub(edge_index, reduction[edge_index])
            x_np = np.asarray(self.node_attr[s_nodes:e_nodes], dtype=np.float32)
            if 2*(x_np.shape[0] - 1) != edge_index.shape[1]:
                print("Node misalignment")
                quit()

            # old index convention check (raw model_type, not the class index)
            if labels_cat_raw.numel() > 0 and idx >= 1000000 and labels_cat_raw[ind_count] == 0:
                print("Found graph with an index indicating BiSSE but a label for BD")
                quit()

            # subtree id per node: computed once per tree, then reused
            part = self._part_cache.get(g)
            if part is None:
                n = x_np.shape[0]
                assert edge_index.min() >= 0 and edge_index.max() < n   # local node numbering
                part_np, _ = node_groups(edge_index.numpy(), x_np.shape[0], self.n_parts)
                part = torch.from_numpy(part_np)
                self._part_cache[g] = part
                assert part.numel() == n and part.max() < self.n_parts

            # add (sub)graph to list for this block
            graph_list.append(GeoData(x=torch.from_numpy(x_np),
                                      edge_index=edge_index,
                                      part=part,
                                      y=labels_cat[ind_count], idx=idx,
                                      lbl_cat=labels_cat[ind_count],
                                      lbl_num=labels_num[ind_count],
                                      # True where the numerical label is defined for this model
                                      lbl_num_mask=torch.isfinite(labels_num[ind_count]),
                                      aux_dat=torch.from_numpy(self.aux_data[g].astype(np.float32)).view(1, -1)))
            ind_count += 1
        return graph_list
    
def find_hdf5_fn(fmt_dir, fmt_prefix, data_str, enc, max_model_states=None):
    """Path of an existing Format HDF5 file for one split and encoding.

    Tries, in order (same naming as format.py):
      <fmt_prefix>.<data_str>.<enc>.states<K>.hdf5   (K = max_model_states)
      the same with a larger K (K+1..5; Train skips the extra models)
      <fmt_prefix>.<data_str>.<enc>.hdf5             (no states tag)
      <fmt_prefix>.<data_str>.hdf5                   (no encoding)
    Without max_model_states only the last two are tried.
    Raises FileNotFoundError if none exists.
    """
    enc = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(enc))   # same naming as format.py
    base = f'{fmt_dir}/{fmt_prefix}.{data_str}.{enc}'
    tried = []
    if max_model_states is not None:
        tried += [f'{base}.states{k}.hdf5'
                  for k in range(int(max_model_states), 6)]
    tried.append(f'{base}.hdf5')
    tried.append(f'{fmt_dir}/{fmt_prefix}.{data_str}.hdf5')

    for i, fn in enumerate(tried):
        if os.path.exists(fn):
            if i > 0:
                print(f'▪ {os.path.basename(tried[0])} not found; using '
                      f'{os.path.basename(fn)}')
            return fn

    raise FileNotFoundError(f'No HDF5 file for {data_str}. Tried: '
                            + ', '.join(tried))

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
    # write to a temporary file first, so a run killed while saving never
    # leaves a broken checkpoint behind
    tmp = f'{filename}.tmp'
    torch.save(state, tmp)
    os.replace(tmp, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

##################################################


def custom_collate(batch):
    graphs, idx, lbl_num, lbl_cat = zip(*batch)
    
    #graphs = Batch.from_data_list(graphs) # needed if not using cuda
    return (
        list(graphs), # remove list() if not using cuda
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


        # filesystem
        self.fmt_prefix             = str(args['fmt_prefix'])
        self.trn_prefix             = str(args['trn_prefix'])
        self.fmt_dir                = str(args['fmt_dir'])
        self.trn_dir                = str(args['trn_dir'])
        self.log_dir                = str(args['log_dir'])
        
        # analysis settings
        self.verbose            = _flag(args['verbose'])
        self.num_proc           = int(args['num_proc'])
        self.use_parallel       = _flag(args['use_parallel'])
        
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
        self.save_phyenc_csv    = _flag(args['save_phyenc_csv'])
        
        # train settings
        self.n_val_blocks           = int(args['n_val_blocks'])
        self.num_epochs         = int(args['num_epochs'])
        self.load_model         = _flag(args['load_model'])
        self.phylo_pool         = _flag(args['phylo_pool'])
        self.subtree_pool         = _flag(args['subtree_pool'])
        self.extra_layers       = _flag(args['extra_layers'])
        self.phylo_n_layers     =int(args['n_layers_phylo'])
        self.avg_n_layers       =int(args['n_layers_avg'])
        self.graph_conv         = _flag(args['graph_conv'])
        self.trn_batch_size     = int(args['trn_batch_size'])
        self.loss_numerical     = str(args['loss_numerical'])
        self.use_cuda           = _flag(args['use_cuda'])
        self.num_early_stop     = int(args['num_early_stop'])
        self.learning_rate      = float(args['learning_rate'])
        self.activation_func    = str(args['activation_func'])
        self.optimizer          = str(args['optimizer'])
        self.hidden_size    = int(args['phy_hidden_size'])
        self.regularisation = str(args['regularisation'])
        self.regression = _flag(args['regression'])
        self.num_classes = int(args['num_classes'])    # replaced by the number of classes found in the data
        print("load model", args['load_model'], bool(args['load_model']))
        print("reg: ", args['regression'], " ", bool(args['regression']))
        # initialized later
        self.phy_tensors        = dict()   # init with encode_all()
        self.train_dataset      = None     # init with load_input()
        self.val_dataset        = None     # init with load_input()
        self.train_blocks       = None
        self.val_blocks         = None
        self.num_msd            = None
        self.attr_msd           = None
        self.aux_msd            = None
        self.first_block = True
        self.idx_num = list()
        self.idx_cat = list()
        self.epochs_since = 0
        self.best_val_loss = -1

        # classes (init with setup_classes())
        self.class_values       = list()   # model_type value of each class index
        self.class_lut          = None     # model_type value -> class index
        self.class_label_name   = None     # categorical label used as the class
        self.class_label_col    = None     # its column in the HDF5 labels array

        # numerical label layout (init with build_num_label_plan())
        self.num_label_plan      = None    # [(out name, own column, BD fallback column)]
        self.num_label_out_names = list()  # names of the adjusted numerical columns
        
        # set CPUs
        if self.num_proc <= 0:
            self.num_proc = cpu_count() + self.num_proc
        if self.num_proc <= 0:
            self.num_proc = 1

        # max_model_states = K: train only on models with at most K character
        # states (K = 3 -> BD, BiSSE, MuSSE_3). Trees of other models in the
        # HDF5 file are skipped. num_states is raised to K as in Format, so the
        # phylo-state tensor width matches the Format output.
        self.max_model_states = parse_max_model_states(
            args.get('max_model_states') if hasattr(args, 'get') else None)
        self.allowed_classes = None

        # resume_training = T: continue from <trn_dir>/checkpoint.tar if it
        # exists (written after every epoch). test_after_train = T: evaluate
        # the best model on the test HDF5 file and write test_* files.
        get = args.get if hasattr(args, 'get') else (lambda k, d=None: d)
        self.resume_training  = _flag(get('resume_training'), False)
        self.test_after_train = _flag(get('test_after_train'), True)
        if self.max_model_states is not None:
            k = self.max_model_states
            self.allowed_classes = sorted(mt for mt, s in MODEL_TYPE_STATES.items() if s <= k)
            print(f'▪ max_model_states = {k}: training on '
                  + ', '.join(model_type_name(v) for v in self.allowed_classes))
            if self.num_states < k:
                self.num_states = k

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
        self._run_start = time.time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        print("self.regression in training: ", self.regression)
        # perform run tasks
        util.print_str('▪ Loading input:', verbose)
        self.load_input()

        num_rjust = len(str(sum(block.shape[0] for block in self.train_blocks)))
        util.print_str(f'  ▪ ' + str(sum(block.shape[0] for block in self.train_blocks)).rjust(num_rjust) + ' training examples', verbose)
        util.print_str(f'  ▪ ' + str(sum(block.shape[0] for block in self.val_blocks)).rjust(num_rjust) + ' validation examples', verbose)

        util.print_str('▪ Training targets:', verbose)
        num_ljust = max([len(k) for k in self.param_est.keys()])
        for k,v in self.param_est.items():
            util.print_str(f'  ▪ {k.ljust(num_ljust)}  [type: {v}]', verbose)


        util.print_str('▪ Building network', verbose)
        self.build_network()

        # write the normalisation files now (not only at the end), so a run
        # that is stopped early can still be evaluated from its checkpoints
        self.save_norm_files()

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

        test_metrics = None
        if self.test_after_train:
            util.print_str('▪ Testing best model', verbose)
            test_metrics = self.test_model()

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        self.write_metrics(test_metrics, time.time() - self._run_start)

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
        self.best_state = None
        self.best_epoch = -1
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
        self.model = None               # init with build_network()
        self.train_label_num_est = None
        self.train_label_num_true = None
        self.train_label_cat_est = None
        self.train_label_cat_true = None
        self.train_label_num_est_calib = None
        self.train_label_index = None
        self.train_history = None       # init with train()
        self.train_label_true = None    # init with load_input()
        self.trn_aux_data_mean_sd = (0,0)
        self.train_labels_num_mean_sd = (0,0)
        self.has_label_cat = False
        self.has_label_num = False
        self.ignore_label_num = False
        self.n_parts =  int(args['n_parts'])
        # self.regression = True
        self.block_size = int(args['block_size'])
        self.dataset_size = int(args['dataset_size'])
        self.accumulation_steps = int(args['accumulation_steps'])
        self.scheduler="manual"
        self.dropout = float(args['dropout'])

        self.encoding_type = str(args['encoding_type'])
        
        return

    ##########################################################
    # Classes and numerical labels
    ##########################################################

    def setup_classes(self, hdf5_fn, rows):
        """Find the classes (model_type values) present in the data.

        The network needs class indices 0..K-1, but model_type can be any
        subset of 0..4 (e.g. BD, BiSSE and MuSSE_4 only = 0, 1, 3). This
        reads the categorical label for all rows used (train + validation),
        sets self.num_classes, and builds the model_type -> class index map.
        With only BD (0) and BiSSE (1) the map is the identity.
        """
        cat_names = [k for k, v in self.param_est.items() if v == 'cat']
        if len(cat_names) == 0:
            return
        if len(cat_names) > 1:
            util.print_warn(f'Several categorical labels {cat_names}; '
                            f'using {cat_names[0]} as the class label.')
        name = cat_names[0]

        with h5py.File(hdf5_fn, 'r') as f:
            names = [x.decode('utf-8') for x in f['label_names'][:].ravel()]
            if name not in names:
                util.print_err(f'Categorical label {name} not found in {hdf5_fn}. '
                               f'Labels are: {names}', exit=True)
            col = names.index(name)
            vals = f['labels'][:, col]

        vals = np.rint(vals[np.sort(np.asarray(rows))]).astype(np.int64)
        values, counts = np.unique(vals, return_counts=True)
        if values.min() < 0:
            util.print_err(f'Negative values found for {name}: {values}', exit=True)

        self.class_label_name = name
        self.class_label_col = col
        self.class_values = [int(v) for v in values]
        lut = torch.full((int(values.max()) + 1,), -1, dtype=torch.long)
        lut[torch.as_tensor(values, dtype=torch.long)] = torch.arange(len(values), dtype=torch.long)
        self.class_lut = lut

        n_args = self.num_classes
        self.num_classes = len(self.class_values)
        if n_args != self.num_classes:
            print(f'▪ num_classes = {n_args} in the config, but {self.num_classes} '
                  f'classes are present in the data; using {self.num_classes}')
        print('▪ Classes (class index: model_type, name, examples):')
        for i, (v, c) in enumerate(zip(self.class_values, counts)):
            print(f'    {i}: {v}  {model_type_name(v):8s} {int(c)}')
        if self.num_classes < 2:
            util.print_err('Fewer than two classes found in the data; check '
                           'select_classes in Format.', exit=True)

    def map_classes(self, labels_cat_raw):
        """model_type values -> class indices (identity if no map is set)."""
        if self.class_lut is None or labels_cat_raw.numel() == 0:
            return labels_cat_raw
        lut = self.class_lut
        bad = (labels_cat_raw < 0) | (labels_cat_raw >= lut.numel())
        if not bad.any():
            mapped = lut[labels_cat_raw]
            bad = mapped < 0
        if bad.any():
            raise ValueError(f'Unexpected {self.class_label_name} values '
                             f'{torch.unique(labels_cat_raw[bad]).tolist()}; '
                             f'known classes are {self.class_values}')
        return mapped

    def class_counts(self, blocks):
        """Number of examples per class index in the given blocks."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        if self.class_label_col is None:
            return counts
        rows = np.sort(np.concatenate([np.asarray(b) for b in blocks]))
        vals = np.rint(self.train_dataset.labels[rows, self.class_label_col]).astype(np.int64)
        for i, v in enumerate(self.class_values):
            counts[i] = int((vals == v).sum())
        return counts

    def to_model_type(self, class_idx):
        """Class indices -> model_type values (for saved results)."""
        class_idx = np.asarray(class_idx).astype(np.int64)
        if len(self.class_values) == 0:
            return class_idx
        return np.asarray(self.class_values, dtype=np.int64)[class_idx]

    def build_num_label_plan(self):
        """Work out the adjusted numerical label columns from param_num_names.

        Output columns:
          log10_birth_rate_t1..tK, log10_death_rate_t1..tK (K = largest state
          number among the numerical labels), each taken from the tree's own
          per-state column, or from log10_birth_rate / log10_death_rate when
          that is not defined (BD trees);
          then every other numerical label as it is (e.g. log10_trans_rate).
        If there are no per-state labels, all labels are kept as they are.
        """
        names = list(self.param_num_names)
        pos = {n: i for i, n in enumerate(names)}

        births, deaths = {}, {}
        for n in names:
            m = _BIRTH_T_RE.match(n)
            if m:
                births[int(m.group(1))] = pos[n]
            m = _DEATH_T_RE.match(n)
            if m:
                deaths[int(m.group(1))] = pos[n]
        n_states = max(list(births) + list(deaths), default=0)

        plan = []
        used = set()
        if n_states > 0:
            bd_birth = pos.get('log10_birth_rate')
            bd_death = pos.get('log10_death_rate')
            for i in range(1, n_states + 1):
                plan.append((f'log10_birth_rate_t{i}', births.get(i), bd_birth))
            for i in range(1, n_states + 1):
                plan.append((f'log10_death_rate_t{i}', deaths.get(i), bd_death))
            used = set(births.values()) | set(deaths.values()) | {bd_birth, bd_death}
            missing = [i for i in range(1, n_states + 1) if i not in births or i not in deaths]
            if missing:
                util.print_warn(f'param_est has per-state rates up to state {n_states} '
                                f'but is missing birth or death rates for states {missing}')
        for n in names:
            if pos[n] not in used:
                plan.append((n, pos[n], None))

        self.num_label_plan = plan
        self.num_label_out_names = [p[0] for p in plan]
        if len(plan) > 0:
            print('▪ Numerical labels:', ', '.join(self.num_label_out_names))

    def norm_lbl_num(self, x):
        """Normalise numerical labels; labels not defined for a tree's model
        (NaN, see lbl_num_mask) become 0, i.e. the training mean."""
        return torch.nan_to_num(util.normalize(x, self.num_msd), nan=0.0)

    ##########################################################

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
        input_aux_data_fn = f'{path_prefix}.aux_data.csv'
        input_labels_fn   = f'{path_prefix}.labels.csv'
        input_idx_data_fn = f'{path_prefix}.index.csv'
        # input_hdf5_fn = f'{path_prefix}.hdf5'
        input_hdf5_fn = find_hdf5_fn(self.fmt_dir, self.fmt_prefix,
                                     'train', self.encoding_type,
                                     self.max_model_states)

        N = 0
        with h5py.File(input_hdf5_fn, "r") as f:
            N = f["phy_data"].shape[0]
            # number of node features in this file (depends on encoding)
            num_node_features = int(f["node_attr"].shape[1])
            num_aux = int(f["aux_data"].shape[1])
            rows = np.arange(N)

            # keep only the classes allowed by max_model_states
            cat_names = [k for k, v in self.param_est.items() if v == 'cat']
            if self.allowed_classes is not None and cat_names:
                names = [x.decode('utf-8') for x in f['label_names'][:].ravel()]
                vals = np.rint(f['labels'][:, names.index(cat_names[0])]).astype(np.int64)
                keep = np.isin(vals, self.allowed_classes)
                dropped, n_drop = np.unique(vals[~keep], return_counts=True)
                for v, c in zip(dropped, n_drop):
                    print(f'▪ Skipping {int(c)} {model_type_name(v)} trees '
                          f'(more than {self.max_model_states} states)')
                rows = rows[keep]
                N = len(rows)
                if N == 0:
                    util.print_err(f'No trees of the allowed classes '
                                   f'{self.allowed_classes} in {input_hdf5_fn}', exit=True)

        if self.dataset_size != -1 and self.dataset_size < N:
            N = self.dataset_size
        else:
            self.dataset_size = N
        print("N = ", N)
        rows = rows[:N]
        perm = rows[np.random.permutation(N)]
        print("self.block_size:", self.block_size)
        blocks = [perm[i:i+self.block_size] for i in range(0,N,self.block_size)]

        self.val_blocks = blocks[:self.n_val_blocks]
        self.train_blocks = blocks[self.n_val_blocks:]
        assert len(self.val_blocks) > 0
        assert len(self.train_blocks) > 0

        # classes present in the data (must be known before the first block is
        # read, because separate_labels() stores num_classes in param_cat)
        self.setup_classes(input_hdf5_fn, perm)

        # numerical label sums are per column and skip NaN (labels a model
        # does not have); created on the first graph, once the width is known
        total_num_sum = None
        total_num_sq_sum = None
        total_num_count = None
        self.num_node_features = num_node_features
        total_attr_sum = torch.zeros(self.num_node_features)
        total_attr_sq_sum = torch.zeros(self.num_node_features)
        total_attr_count = 0
        total_aux_sum = torch.zeros(num_aux)
        total_aux_sq_sum = torch.zeros(num_aux)
        total_aux_count = 0

        print("LOADING", input_hdf5_fn)
        self.train_dataset = HDF5BlockDataset(input_hdf5_fn, self.train_blocks, n_parts = self.n_parts, trainer = self)

        pbar = tqdm(total=len(self.train_dataset))
        for block_idx in range(len(self.train_dataset)):
            graphs = self.train_dataset[block_idx]
            for g in graphs:
                x = torch.as_tensor(g.x, dtype=torch.float32)
                labels_num = torch.as_tensor(g.lbl_num, dtype=torch.float32).to(self.TORCH_DEVICE)
                ok = torch.isfinite(labels_num)
                labels_num = torch.where(ok, labels_num, torch.zeros_like(labels_num))
                if total_num_sum is None:
                    total_num_sum = torch.zeros_like(labels_num)
                    total_num_sq_sum = torch.zeros_like(labels_num)
                    total_num_count = torch.zeros_like(labels_num)
                aux = torch.as_tensor(g.aux_dat, dtype=torch.float32)
                total_num_sum += labels_num
                total_num_sq_sum += labels_num ** 2
                total_num_count += ok.float()
                total_attr_sum += x.sum(dim=0)
                total_attr_sq_sum += (x ** 2).sum(dim=0)
                total_attr_count += x.shape[0]
                total_aux_sum += aux.squeeze(0)
                total_aux_sq_sum += (aux.squeeze(0) ** 2)
                total_aux_count += 1
            pbar.update(1)

        pbar.close()

        # numerical labels: mean / sd over the trees where each label is defined
        n_ok = total_num_count
        mean_num = total_num_sum / n_ok.clamp(min=1)
        var_num = (total_num_sq_sum - total_num_sum**2 / n_ok.clamp(min=1)) / (n_ok - 1).clamp(min=1)
        sd_num = torch.sqrt(var_num.clamp(min=0))
        mean_num = torch.where(n_ok >= 1, mean_num, torch.zeros_like(mean_num))
        sd_num = torch.where((n_ok >= 2) & (sd_num > 0), sd_num, torch.ones_like(sd_num))
        for name, c in zip(self.num_label_out_names, n_ok.tolist()):
            if c < 2:
                util.print_warn(f'Numerical label {name} is defined for only {int(c)} '
                                f'training trees; its mean/sd are set to 0/1')

        mean_attr = total_attr_sum / total_attr_count
        var_attr = (total_attr_sq_sum -  total_attr_sum**2 / total_attr_count)/(total_attr_count - 1)
        sd_attr = torch.sqrt(var_attr)
        mean_aux = total_aux_sum / total_aux_count
        var_aux = (total_aux_sq_sum -  total_aux_sum**2 / total_aux_count)/(total_aux_count - 1)
        sd_aux = torch.sqrt(var_aux)
        # features that never vary (e.g. a state no tree has) would give
        # sd = 0 and NaN after normalising; use sd = 1 for those
        sd_attr = torch.where(sd_attr > 0, sd_attr, torch.ones_like(sd_attr))
        sd_aux = torch.where(sd_aux > 0, sd_aux, torch.ones_like(sd_aux))
        self.num_msd = (mean_num, sd_num)
        self.attr_msd = (mean_attr, sd_attr)
        self.aux_msd = (mean_aux, sd_aux)
        self.num_msd = tuple(t.to(self.TORCH_DEVICE) for t in self.num_msd)
        self.aux_msd = tuple(t.to(self.TORCH_DEVICE) for t in self.aux_msd)
        self.attr_msd = tuple(t.to(self.TORCH_DEVICE) for t in self.attr_msd)

        self.val_dataset = HDF5BlockDataset(input_hdf5_fn, self.val_blocks, trainer = self, n_parts = self.n_parts)

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
        if label_names is None:
            return
        else:
            self.label_names = label_names
            labels = torch.from_numpy(input_labels).clone()
        if (self.first_block):
            for k,v in self.param_est.items():
                if v == 'cat':
                    self.has_label_cat = True
                    idx = self.label_names.index(k)
                    self.param_cat[k] = self.num_classes 
                    self.idx_cat.append( idx )
                    if k not in self.param_cat_names:
                        self.param_cat_names.append(k)
                    
                # ignore numerical labels as a hack to pass parameter information for plotting without using it for predictions.
                elif v == 'num' and self.ignore_label_num == False:
                    self.has_label_num = True
                    self.idx_num.append( self.label_names.index(k) )
                    if k not in self.param_num_names:
                        self.param_num_names.append(k)
            
            if not self.has_label_num and not self.has_label_cat:
                util.print_err(f"No training labels found.", exit=True)
            self.first_block = False
        
        # get data subsets
        labels_num = labels[:,self.idx_num].clone()
        labels_cat = labels[:,self.idx_cat].clone()

        # done
        return labels_num, labels_cat

##################################################

    def build_network(self):
       
        # torch multiprocessing, eventually need to get working with cuda
        torch.set_num_threads(self.num_proc)
        self.n_layers = self.avg_n_layers
        if self.phylo_pool:
            self.n_layers = self.phylo_n_layers
        if True:
            # build model architecture
            self.model = network.ParameterEstimationNetwork(
                                                        num_node_features = self.num_node_features,
                                                        hidden_size = self.hidden_size,
                                                        num_classes = self.num_classes,
                                                        param_cat=self.param_cat,
                                                        phylo_pool = self.phylo_pool,
                                                        subtree_pool = self.subtree_pool,
                                                        graph_conv = self.graph_conv,
                                                        extra_layers = self.extra_layers,
                                                        n_layers = self.n_layers,
                                                        dropout=self.dropout,
                                                        activation_func=self.activation_func,
                                                        args=self.args,
                                                        n_parts=self.n_parts)
            self.model.phy_std.debug_dir = "debug_trees"


        if self.use_cuda:
            self.model = DataParallel(self.model)
        self.model.to(self.TORCH_DEVICE)
        net = self.model.module if hasattr(self.model, "module") else self.model
        print("pooling model:", type(net.phy_std).__name__)

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

    def evaluate(self, dataset, loss_func, type="train"):
        self.model.eval()

        total_loss = 0.0
        total_n = 0
        total_correct = 0
        all_delta = []
        prob_list = []
        pred_list = []
        label_list = []
        x_embed = []
        with torch.no_grad():
            for j in range(len(dataset)):
                graph_list = dataset[j]
                loader = DataListLoader(
                    graph_list,
                    batch_size=self.trn_batch_size
                )

                for batch in loader:
                    labels = torch.cat(
                        [g.lbl_cat for g in batch], dim=0
                    ).to(self.TORCH_DEVICE)

                    for g in batch:
                        #if type == "val":
                        g.x = util.normalize(g.x, self.attr_msd)
                        g.lbl_num = self.norm_lbl_num(g.lbl_num)
                        g.aux_dat = util.normalize(g.aux_dat, self.aux_msd)

                        if not torch.isfinite(g.x).all():
                            print("BAD g.x")
                            print("idx:", g.idx)
                            print("x min/max:", g.x.min(), g.x.max())
                            raise RuntimeError()

                        if not torch.isfinite(g.lbl_num).all():
                            print("BAD g.lbl_num")
                            print("idx:", g.idx)
                            print("lbl_num:", g.lbl_num)
                            raise RuntimeError()

                        if not torch.isfinite(g.aux_dat).all():
                            print("BAD g.aux_dat")
                            print("idx:", g.idx)
                            print("aux_dat:", g.aux_dat)
                            raise RuntimeError()

                   
                    outputs = self.model(batch)
                    preds = outputs[3]
                    delta = preds[:,1] - preds[:,0]
                    all_delta.append(delta.detach().cpu())
                    loss = loss_func(preds, labels.flatten())
                    amax = preds.argmax(dim=1)

                    n = len(amax)

                    total_loss += loss.item() * n
                    total_n += n
                    total_correct += int(
                        (amax == labels.flatten()).sum()
                    )
        return total_loss / total_n, total_correct / total_n

    def get_reg_loss(self):
        reg_loss = 0.0
        l1_lambda = 1e-5
        l2_lambda = 1e-4

        if self.regularisation in ("L1", "L1L2"):
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            reg_loss += l1_lambda * l1_norm

        if self.regularisation in ("L2", "L1L2"):
            l2_norm = sum(p.square().sum() for p in self.model.parameters())
            reg_loss += l2_lambda * l2_norm

        return reg_loss

    def train(self):
        """Trains the neural network model.

        This function compiles the network model to prepare it for training.
        Perform training by compiling the model with appropriate loss functions
        and metrics, and then fitting the model to the training data. Training
        produces a history dictionary, which is saved.

        Returns:
            None

        """

        n_trn_graphs = sum(block.shape[0] for block in self.train_blocks)
        n_val_graphs = sum(block.shape[0] for block in self.val_blocks)
        print("number of batches: " + str(len(self.train_blocks)))
        print("n train:", n_trn_graphs)
        print("n val:", n_val_graphs)

        # examples per class in each split
        trn_counts = self.class_counts(self.train_blocks)
        val_counts = self.class_counts(self.val_blocks)
        for v, nt, nv in zip(self.class_values, trn_counts, val_counts):
            name = model_type_name(v)
            print(f"n train {name}:", int(nt))
            print(f"n val {name}:", int(nv))

        # validation dataset

        val_bad_count = 0
        val_cycle_good_count = 0


        # loss functions
        loss_categ_func = network.CrossEntropyLoss()
        loss_aggregation = 'median'

        # cross-entropy of a model that has learned nothing is ln(num_classes)
        # (0.693 for two classes); used to stop runs that stay at that level
        chance_loss = math.log(max(self.num_classes, 2))
        chance_loss_threshold = 0.995 * chance_loss
        
        print("learning rate: ", str(self.learning_rate))
        # net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        #for layer in [net.phy_std.gconv1, net.phy_std.gconv2, net.phy_std.gconv3, net.phy_std.gconv4]:
        #    for p in layer.parameters():
        #        p.requires_grad = False

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001,eps=1e-4)
            #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.learning_rate)
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
        


        # TODO: simplify training logging!!
        # gather training history
        history_col_names = ['epoch', 'dataset', 'metric', 'value']
        self.train_history = pd.DataFrame(columns=history_col_names)

        # training

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


        learning_rate = self.learning_rate
        old_learning_rate = learning_rate

        # plt.ion()
        # fig, ax = plt.subplots()
        # trn_line, = ax.plot([], [], label="Train")
        # val_line, = ax.plot([], [], label="Validation")
        # ax.set_xlabel("Epoch")
        # ax.set_ylabel("Loss")
        # ax.set_title("Training vs Validation Loss")
        # ax.legend()
        trn_losses = []
        val_losses = []
        num_batches = len(self.train_blocks)
        chance_level_count = 0
        val_metric_vals = None

        # set here (not only inside the loop) so a resumed run whose training
        # had already finished still has them for make_results / save_results
        self.effective_batch_size = self.accumulation_steps * self.trn_batch_size
        self.total_batches = len(self.train_dataset)*(math.ceil(self.block_size / self.trn_batch_size))
        path_prefix = f'{self.trn_dir}/{self.trn_prefix}'
        model_history_fn = f'{path_prefix}.{self.dataset_size}.{self.effective_batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.subtree_pool}.{self.extra_layers}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.train_history.csv'
        self.stopped_early = False
        self.stop_reason = ''

        # resume from the last epoch checkpoint
        start_epoch = 0
        resume_fn = f'{self.trn_dir}/checkpoint.tar'
        if self.resume_training and os.path.exists(resume_fn):
            ckpt = _torch_load(resume_fn)
            st = ckpt.get('trainer_state')
            if st is None or 'epoch' not in ckpt:
                print(f'▪ {resume_fn} has no training state; starting from epoch 1')
            elif list(ckpt.get('class_values', [])) != list(self.class_values):
                print(f'▪ {resume_fn} was trained on classes {ckpt.get("class_values")} '
                      f'but this run uses {self.class_values}; starting from epoch 1')
            else:
                self.model.load_state_dict(ckpt['state_dict'])
                optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = int(ckpt['epoch']) + 1
                self.train_history = pd.DataFrame(st['history'], columns=history_col_names)
                self.best_state = st['best_state']
                self.best_epoch = st['best_epoch']
                self.best_val_loss = st['best_val_loss']
                self.epochs_since = st['epochs_since']
                val_bad_count = st['val_bad_count']
                chance_level_count = st['chance_level_count']
                prev_trn_loss_combined = st['prev_trn_loss_combined']
                prev_val_loss_combined = st['prev_val_loss_combined']
                prev_trn_acc_combined = st['prev_trn_acc_combined']
                prev_val_acc_combined = st['prev_val_acc_combined']
                trn_losses = list(st['trn_losses'])
                val_losses = list(st['val_losses'])
                self.stopped_early = bool(st.get('stopped', False))
                self.stop_reason = st.get('stop_reason', '')
                try:
                    np.random.set_state(st['np_rng'])
                    torch.set_rng_state(st['torch_rng'])
                    if st.get('cuda_rng') is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(st['cuda_rng'])
                except Exception as e:
                    print('▪ Could not restore random number generator state:', e)
                if self.stopped_early:
                    print(f'▪ Checkpoint: training already stopped after epoch {start_epoch} '
                          f'({self.stop_reason})')
                    start_epoch = self.num_epochs
                else:
                    print(f'▪ Resuming training at epoch {start_epoch+1} of {self.num_epochs} '
                          f'(best val loss so far {self.best_val_loss:.4f} at epoch {self.best_epoch+1})')

        for i in range(start_epoch, self.num_epochs):
            print("Epoch", str(i))
            if old_learning_rate != learning_rate:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate  
                print("changing learning rate to:", learning_rate)
            old_learning_rate = learning_rate
            # print('-----')
            trn_loss_combined = 0.
            trn_acc_combined = 0.
            val_acc_combined = 0.
            trn_correct = 0.
            val_correct = 0.
            trn_length = 0.
            val_length = 0.
            val_loss_combined = 0.
            val_loss_all_batches = 0
            trn_mse_value = 0.
            trn_mape_value = 0.
            trn_mae_value = 0.

            trn_msg = f'Training epoch {i+1} of {self.num_epochs}'
            # self.model.new_epoch(epoch)
            self.model.train()
            accumulation_steps =  self.accumulation_steps
            self.effective_batch_size = accumulation_steps * self.trn_batch_size

            optimizer.zero_grad()

            self.total_batches = len(self.train_dataset)*(math.ceil(self.block_size / self.trn_batch_size))
            pbar = tqdm(total=self.total_batches)
            loss_list = list()

            batch_count = 0
            l1_lambda = 1e-5
            l2_lambda = 1e-4
            trn_answers = 0
            block_order = np.random.permutation(len(self.train_dataset))
            for j in block_order:
                graph_list = self.train_dataset[j]
                loader = DataListLoader(graph_list, batch_size=self.trn_batch_size)
                for batch in loader:
                    # one-off check that subtree ids are attached (first batch of first epoch)
                    if i == 0 and batch_count == 0:
                        g0 = batch[0]
                        print("nodes:", g0.num_nodes,
                              "subtree sizes:", torch.bincount(g0.part, minlength=self.n_parts).tolist())

                    # normalize
                    lbl_cat = torch.cat([g.lbl_cat for g in batch], dim=0).to(self.TORCH_DEVICE)                    
                    lbl_num = torch.stack([g.lbl_num for g in batch], dim=0).to(self.TORCH_DEVICE)
                    for g in batch:
                        # old = g.x
                        g.x = util.normalize(g.x, self.attr_msd)
                        g.lbl_num = self.norm_lbl_num(g.lbl_num)
                        g.aux_dat = util.normalize(g.aux_dat, self.aux_msd)
                    try:
                        # forward
                        lbls_hat = self.model(batch)
                        pbar.update(1)
                    except Exception as e:
                        print("exception in lbls_hat", e)
                        quit()

                   
                    third_arg = lbls_hat[3] # [0:int(len(lbls_hat[3])/4)]
                    preds = third_arg #.long()

                    probs = torch.softmax(preds, dim=1)



                    # loss
                    loss_categ = loss_categ_func(preds, lbl_cat.flatten())
                    amax = preds.argmax(dim=1)

                    loss_categ_scaled = loss_categ / accumulation_steps
                    loss_categ_scaled.backward()

                    batch_count += 1

                    # update network parameters
                    if (batch_count % accumulation_steps == 0):
                        reg_loss = 0
                        if self.regularisation == "L1" or self.regularisation == "L1L2":
                            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                            reg_loss = reg_loss + l1_lambda*l1_norm
                        if self.regularisation == "L2" or self.regularisation == "L1L2":
                            l2_norm = sum(p.square().sum() for p in self.model.parameters())
                            reg_loss = reg_loss + l2_lambda*l2_norm
                        if reg_loss != 0:
                            reg_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        #print("epoch", i, "grad norm sample:", grad_norm.item())

                        max_grad = 0
                        max_grad_name = None
                        # for name, p in self.model.named_parameters():
                        #     if p.grad is not None:
                        #         grad_mean = p.grad.abs().mean().item()
                        #         grad_max = p.grad.abs().max().item()

                        #         print(name, "grad mean:", grad_mean, "grad max:", grad_max)

                        #     if grad_max > max_grad:
                        #         max_grad = grad_max
                        #         max_grad_name = name
                        optimizer.step()
                        optimizer.zero_grad()
            if (batch_count % accumulation_steps != 0):
                #scale = accumulation_steps / (batch_count % accumulation_steps)

                #for p ini self.model.parameters():
                #    if p.grad is not None:
                #        p.grad *= scale

                reg_loss = 0.0

                if self.regularisation in ("L1", "L1L2"):
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    reg_loss += l1_lambda * l1_norm

                if self.regularisation in ("L2", "L1L2"):
                    l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                    reg_loss += l2_lambda * l2_norm

                if reg_loss != 0:
                    reg_loss.backward()

                scale = accumulation_steps / (batch_count % accumulation_steps)
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad *= scale

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
               # print("epoch", i, "grad norm sample (trailing batch):", grad_norm.item())

                optimizer.step()
                optimizer.zero_grad()
            pbar.close()
           # print("truth", lbl_cat.flatten())
            #print("final logits", third_arg)

            # trn_acc_combined += trn_correct / trn_length
            # trn_loss_combined = epoch_data_loss / epoch_n
            

            trn_loss, trn_acc = self.evaluate(
                self.train_dataset,
                loss_categ_func
            )
            reg_loss = 0
            if self.regularisation != "NA":
                reg_loss = self.get_reg_loss().item()
            trn_loss_combined = trn_loss + reg_loss
            metric_names = ['loss_combined', 'accuracy_combined']#, 'accuracy_combined']
            trn_metric_vals = [trn_loss_combined, trn_acc ] 
            trn_losses.append(trn_loss_combined)

            trn_loss_str = f'    Train        --   loss: {"{0:.4f}".format(trn_loss_combined)}\t{"{0:.4f}".format(trn_loss)}\t'
            trn_acc_str = f'--   acc: {"{0:.4f}".format(trn_acc)}\t'

            # Count consecutive epochs with training loss at chance level
            # (ln(num_classes); 0.69 for two classes)
            if trn_loss_combined >= chance_loss_threshold:
                chance_level_count += 1
            else:
                chance_level_count = 0

            self.model.eval()
            with torch.no_grad():
                
                val_loss, val_acc = self.evaluate(
                self.val_dataset,
                loss_categ_func, type="val"
                )
                val_loss_combined = val_loss #+ reg_loss

                val_metric_vals = [
                                        val_loss_combined, val_acc]
                val_acc_str = f'--   acc: {"{0:.4f}".format(val_acc)}'
                # print("val loss combined:", val_loss_combined)
                val_losses.append(val_loss_combined)

                val_loss_str = f'    Validation   --   loss: {"{0:.4f}".format(val_loss_combined)}\t{"{0:.4f}".format(val_loss)}\t'
                if (self.best_val_loss == -1 or val_loss_combined < self.best_val_loss):
                    self.best_val_loss = val_loss_combined
                    self.epochs_since = 0
                    self.best_epoch = i
                    # keep an in-memory copy on CPU (avoids extra GPU memory)
                    self.best_state = {k: v.detach().cpu().clone()
                                    for k, v in self.model.state_dict().items()}
                    save_checkpoint({'epoch': i,
                                    'val_loss': val_loss_combined,
                                    'state_dict': self.best_state,
                                    'optimizer': copy.deepcopy(optimizer.state_dict()),
                                    'class_values': list(self.class_values),
                                    'settings': self.model_settings(),
                                    'norm': self.norm_state()},
                                    filename=f'{self.trn_dir}/best_checkpoint.tar')
                    print(f"  New best validation loss {val_loss_combined:.4f} at epoch {i+1}")
                else:
                    self.epochs_since += 1


            # changes in training metrics between epochs
                if i > 0:

                    diff_trn_loss = trn_loss_combined - prev_trn_loss_combined
                    diff_val_loss = val_loss_combined - prev_val_loss_combined
                    print("diff val loss", diff_val_loss)
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
                    else:
                        val_bad_count = 0

                prev_trn_loss_combined = trn_loss_combined
                prev_val_loss_combined = val_loss_combined

                prev_trn_acc_combined = trn_acc_combined
                prev_val_acc_combined = val_acc_combined

                print(trn_loss_str, trn_acc_str)
                print("")
                print(val_loss_str, val_acc_str)
                print('')
        
            # update train history log
            self.update_train_history(i, metric_names, trn_metric_vals, 'train')
            self.update_train_history(i, metric_names, val_metric_vals, 'validation')
            self.train_history.to_csv(model_history_fn, index=False, sep=',',
                                  float_format=util.PANDAS_FLOAT_FMT_STR)
            self.train_history.to_csv(f'{self.trn_dir}/history.csv', index=False, sep=',',
                                  float_format=util.PANDAS_FLOAT_FMT_STR)

            # early stopping
            stop_msg = None
            if val_bad_count >= self.num_early_stop and self.num_early_stop > 0:
                stop_msg = (f'validation loss increased for num_early_stop={self.num_early_stop} '
                            f'consecutive epochs')
            elif chance_level_count >= 20:
                stop_msg = (f'training loss was at chance level '
                            f'(>= {chance_loss_threshold:.3f}, ln({self.num_classes}) = {chance_loss:.3f}) '
                            f'for 20 consecutive epochs')

            # checkpoint with everything needed to resume after this epoch
            checkpoint = {'epoch': i,
                          'state_dict': self.model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'class_values': list(self.class_values),
                          'settings': self.model_settings(),
                          'norm': self.norm_state(),
                          'trainer_state': {
                              'history': self.train_history.values.tolist(),
                              'best_state': self.best_state,
                              'best_epoch': self.best_epoch,
                              'best_val_loss': self.best_val_loss,
                              'epochs_since': self.epochs_since,
                              'val_bad_count': val_bad_count,
                              'chance_level_count': chance_level_count,
                              'prev_trn_loss_combined': prev_trn_loss_combined,
                              'prev_val_loss_combined': prev_val_loss_combined,
                              'prev_trn_acc_combined': prev_trn_acc_combined,
                              'prev_val_acc_combined': prev_val_acc_combined,
                              'trn_losses': trn_losses,
                              'val_losses': val_losses,
                              'stopped': stop_msg is not None,
                              'stop_reason': stop_msg or '',
                              'np_rng': np.random.get_state(),
                              'torch_rng': torch.get_rng_state(),
                              'cuda_rng': (torch.cuda.get_rng_state_all()
                                           if torch.cuda.is_available() else None),
                          }}
            save_checkpoint(checkpoint, filename=f'{self.trn_dir}'+"/checkpoint.tar")

            if stop_msg is not None:
                print(f'Early stop: {stop_msg}')
                self.stopped_early = True
                self.stop_reason = stop_msg
                break
            #if self.epochs_since >= 20:
            #    print(f'Early stop: lowest validation loss was 20 epochs ago')
            #    history_plot_fn = f'{path_prefix}.{self.dataset_size}.{self.effective_batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.extra_layers}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.train_history.png'
             #   break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"Restored best model from epoch {self.best_epoch+1} "
                f"(val loss {self.best_val_loss:.4f})")
        return
    
    def test_model(self):
        """Evaluate the (best) trained model on the test HDF5 file.

        Uses the test file from Format (same encoding and max_model_states
        lookup as the training file), keeps only trees of the classes the
        model was trained on, and applies the training normalisation.
        Writes <trn_dir>/test_predictions.csv and test_confusion.csv and
        returns a dict of test metrics (None if there is no test file).
        """
        try:
            test_fn = find_hdf5_fn(self.fmt_dir, self.fmt_prefix, 'test',
                                   self.encoding_type, self.max_model_states)
        except FileNotFoundError as e:
            print(f'▪ No test file, skipping test evaluation ({e})')
            return None

        with h5py.File(test_fn, 'r') as f:
            n_rows = f['phy_data'].shape[0]
            rows = np.arange(n_rows)
            if self.class_label_name is not None:
                names = [x.decode('utf-8') for x in f['label_names'][:].ravel()]
                vals = np.rint(f['labels'][:, names.index(self.class_label_name)]).astype(np.int64)
                rows = rows[np.isin(vals, self.class_values)]
        if len(rows) == 0:
            print(f'▪ No test trees of classes {self.class_values} in {test_fn}')
            return None
        blocks = [rows[i:i+self.block_size] for i in range(0, len(rows), self.block_size)]
        test_dataset = HDF5BlockDataset(test_fn, blocks, trainer=self, n_parts=self.n_parts)
        print(f'▪ Testing on {len(rows)} trees from {test_fn}')

        loss_func = network.CrossEntropyLoss()
        total_loss = 0.0
        true_list, pred_list, prob_list, idx_list = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for j in tqdm(range(len(test_dataset)), desc='Testing'):
                graph_list = test_dataset[j]
                loader = DataListLoader(graph_list, batch_size=self.trn_batch_size)
                for batch in loader:
                    labels = torch.cat([g.lbl_cat for g in batch], dim=0).to(self.TORCH_DEVICE).flatten()
                    idx_list.append(np.array([int(g.idx) for g in batch]))
                    for g in batch:
                        g.x = util.normalize(g.x, self.attr_msd)
                        g.lbl_num = self.norm_lbl_num(g.lbl_num)
                        g.aux_dat = util.normalize(g.aux_dat, self.aux_msd)
                    preds = self.model(batch)[3]
                    total_loss += loss_func(preds, labels).item() * len(labels)
                    prob_list.append(torch.softmax(preds, dim=1).cpu().numpy())
                    pred_list.append(preds.argmax(dim=1).cpu().numpy())
                    true_list.append(labels.cpu().numpy())

        y_true = np.concatenate(true_list)
        y_pred = np.concatenate(pred_list)
        probs = np.concatenate(prob_list)
        idx = np.concatenate(idx_list)
        K = self.num_classes
        names = [model_type_name(v) for v in self.class_values]

        conf = np.zeros((K, K), dtype=np.int64)
        np.add.at(conf, (y_true, y_pred), 1)
        support = conf.sum(axis=1)
        recall = np.where(support > 0, np.diag(conf) / np.maximum(support, 1), np.nan)
        pred_tot = conf.sum(axis=0)
        precision = np.where(pred_tot > 0, np.diag(conf) / np.maximum(pred_tot, 1), np.nan)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)

        metrics = {
            'test_file': test_fn,
            'test_n': int(len(y_true)),
            'test_loss': total_loss / len(y_true),
            'test_acc': float((y_true == y_pred).mean()),
            'test_balanced_acc': float(np.nanmean(recall)),
            'test_macro_f1': float(np.nanmean(np.where(support > 0, f1, np.nan))),
            'test_per_class': {n: {'n': int(support[i]),
                                   'recall': None if np.isnan(recall[i]) else float(recall[i]),
                                   'precision': None if np.isnan(precision[i]) else float(precision[i])}
                               for i, n in enumerate(names)},
            'test_confusion': conf.tolist(),
            'class_names': names,
        }

        # per-tree predictions (model_type values) and confusion matrix
        df = pd.DataFrame({'idx': idx,
                           'true_model_type': self.to_model_type(y_true),
                           'pred_model_type': self.to_model_type(y_pred)})
        for i, n in enumerate(names):
            df[f'prob_{n}'] = probs[:, i]
        df.to_csv(f'{self.trn_dir}/test_predictions.csv', index=False)
        pd.DataFrame(conf, index=[f'true_{n}' for n in names],
                     columns=[f'pred_{n}' for n in names]).to_csv(f'{self.trn_dir}/test_confusion.csv')

        print(f"▪ Test loss {metrics['test_loss']:.4f}  acc {metrics['test_acc']:.4f}  "
              f"balanced acc {metrics['test_balanced_acc']:.4f}")
        return metrics

    def write_metrics(self, test_metrics=None, run_seconds=None):
        """Write <trn_dir>/metrics.json: settings, best-epoch train and
        validation metrics, early stopping and (if run) test metrics."""
        h = self.train_history
        def at(epoch, dataset, metric):
            r = h[(h['epoch'] == epoch) & (h['dataset'] == dataset) & (h['metric'] == metric)]
            return float(r['value'].iloc[-1]) if len(r) else None

        last_epoch = int(h['epoch'].max()) if len(h) else -1
        out = {
            'trn_dir': self.trn_dir,
            'max_model_states': self.max_model_states,
            'num_states': self.num_states,
            'n_parts': self.n_parts,
            'subtree_pool': bool(self.subtree_pool),
            'phylo_pool': bool(self.phylo_pool),
            'num_epochs': self.num_epochs,
            'epochs_run': last_epoch + 1,
            'stopped_early': bool(getattr(self, 'stopped_early', False)),
            'stop_reason': getattr(self, 'stop_reason', ''),
            'num_classes': self.num_classes,
            'class_values': list(self.class_values),
            'class_names': [model_type_name(v) for v in self.class_values],
            'n_train': int(sum(len(b) for b in self.train_blocks)),
            'n_val': int(sum(len(b) for b in self.val_blocks)),
            'best_epoch': int(self.best_epoch) + 1,
            'best_val_loss': at(self.best_epoch, 'validation', 'loss_combined'),
            'best_val_acc': at(self.best_epoch, 'validation', 'accuracy_combined'),
            'train_loss_at_best': at(self.best_epoch, 'train', 'loss_combined'),
            'train_acc_at_best': at(self.best_epoch, 'train', 'accuracy_combined'),
            'final_val_loss': at(last_epoch, 'validation', 'loss_combined'),
            'final_val_acc': at(last_epoch, 'validation', 'accuracy_combined'),
            'run_seconds': run_seconds,
        }
        if test_metrics:
            out.update(test_metrics)
        tmp = f'{self.trn_dir}/metrics.json.tmp'
        with open(tmp, 'w') as f:
            json.dump(_to_jsonable(out), f, indent=2)
        os.replace(tmp, f'{self.trn_dir}/metrics.json')
        print(f'▪ Metrics written to {self.trn_dir}/metrics.json')

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

    
    def make_results(self):
        """Makes all results from the Train step.

        Runs the trained network over the training set (in eval mode, with the
        same input normalisation as training) and collects true labels and
        estimates. True numerical labels are kept on their original scale,
        with NaN where a label is not defined for a tree's model.

        """

        
        # get uncalibrated estimates
        # training label estimates
        first_batch = True
        pbar = tqdm(total=self.total_batches)
        all_true_res = []
        all_est_res = []
        all_idx = []
        self.model.eval()
        with torch.no_grad():
            for j in range(len(self.train_dataset)):
                graph_list = self.train_dataset[j]
                loader = DataListLoader(graph_list, batch_size=self.trn_batch_size)
                for batch in loader:
                    pbar.update(1)
                    # batch = batch.to(self.TORCH_DEVICE)
                    trn_labels_cat = torch.cat([g.lbl_cat for g in batch], dim=0).to(self.TORCH_DEVICE)
                    # original-scale numerical labels (NaN where not defined)
                    trn_labels_num = torch.stack([g.lbl_num.clone() for g in batch], dim=0)
                    idx_batch = np.array([g.idx for g in batch])

                    # same normalisation as in training
                    for g in batch:
                        g.x = util.normalize(g.x, self.attr_msd)
                        g.lbl_num = self.norm_lbl_num(g.lbl_num)
                        g.aux_dat = util.normalize(g.aux_dat, self.aux_msd)

                    label_est = self.model(batch)
                    labels_num_est = label_est[3]
                    # labels_num_est = torch.stack(labels_num_est).cpu().detach()#.numpy()
                    labels_cat_est = label_est[3]
                    all_idx.append(idx_batch)



                    if self.has_label_num:
                        all_true_res.append(trn_labels_num.cpu().detach())
                        all_est_res.append(labels_num_est.cpu().detach())


                    # generate calibration factors

                    if self.has_label_cat:
                        trn_label_cat = trn_labels_cat.cpu().detach().numpy().astype('int')
                        if j == 0 and first_batch:
                            # print("j =", j, "cat true is none")
                            self.train_label_cat_true = trn_label_cat.copy()
                        else:
                            self.train_label_cat_true = np.append(self.train_label_cat_true, trn_label_cat)
                        if not self.regression:
                            if j == 0 and first_batch:
                                self.train_label_cat_est = labels_cat_est.argmax(dim=1).cpu().detach().numpy()
                            else:
                                self.train_label_cat_est = np.append(self.train_label_cat_est, labels_cat_est.argmax(dim=1).cpu().detach().numpy())
                            total_correct = (self.train_label_cat_true == self.train_label_cat_est).sum()
                    first_batch = False
        if self.has_label_num:
            self.train_label_num_true = torch.cat(all_true_res, dim=0).numpy()
            self.train_label_num_est = torch.cat(all_est_res, dim=0).numpy()
        self.train_label_index = np.concatenate(all_idx)
        pbar.close()   

        # reformat categorical estimates, if they exist
        return

    def format_label_cat(self, x):
        """Formats categorical labels.

        Formats categorical labels for training and validation datasets.

        """
        
        df_list = list()
        for k,v in x.items():
            v = torch.softmax(v, dim=1).cpu().detach().numpy()
            col_names = [ f'{k}_{i}' for i in range(v.shape[1]) ]
            df = pd.DataFrame(v, columns=col_names)
            df_list.append(df)
            
        return pd.concat(df_list, axis=1)

    def _file_prefix(self):
        """{trn_dir}/{trn_prefix}.{run_str}.{n_layers}.{lr}.{dropout}.{act}.{reg}
        - the same prefix save_results uses for the norm and model files."""
        eff_batch = self.accumulation_steps * self.trn_batch_size
        run_str = (f'{self.num_classes}.{self.dataset_size}.{eff_batch}.{self.optimizer}.'
                   f'{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.'
                   f'{self.subtree_pool}.{self.extra_layers}')
        return (f'{self.trn_dir}/{self.trn_prefix}.{run_str}.{self.n_layers}.'
                f'{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}')

    def model_settings(self):
        """Everything needed to rebuild the network from a checkpoint."""
        net = self.model.module if hasattr(self.model, "module") else self.model
        return {
            'num_node_features': int(self.num_node_features),
            'hidden_size': int(self.hidden_size),
            'num_classes': int(self.num_classes),
            'class_values': [int(v) for v in self.class_values],
            'phylo_pool': bool(self.phylo_pool),
            'subtree_pool': bool(self.subtree_pool),
            'graph_conv': bool(self.graph_conv),
            'extra_layers': bool(self.extra_layers),
            'n_layers': int(self.n_layers),
            # n_parts the network actually uses (GCN_PhyloPool may not get the setting)
            'n_parts': int(getattr(net.phy_std, 'n_parts', self.n_parts)),
            'n_parts_setting': int(self.n_parts),
            'dropout': float(self.dropout),
            'activation_func': str(self.activation_func),
        }

    def norm_state(self):
        """Training mean / sd of node features, aux data and numerical labels (CPU)."""
        def cpu(msd):
            return None if msd is None else tuple(t.detach().cpu().clone() for t in msd)
        return {'attr_msd': cpu(self.attr_msd),
                'aux_msd': cpu(getattr(self, 'aux_msd', None)),
                'num_msd': cpu(getattr(self, 'num_msd', None))}

    def save_norm_files(self):
        """Write the trn_norm.*.csv files and run_settings.json (atomically).
        save_results writes the same norm files again at the end of training."""
        prefix = self._file_prefix()

        def write_csv(df, fn):
            tmp = fn + '.tmp'
            df.to_csv(tmp, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
            os.replace(tmp, fn)

        write_csv(pd.DataFrame({'mean': self.attr_msd[0].cpu().numpy(),
                                'sd': self.attr_msd[1].cpu().numpy()}),
                  f'{prefix}.trn_norm.attr_data.csv')
        write_csv(pd.DataFrame({'mean': self.aux_msd[0].cpu().numpy(),
                                'sd': self.aux_msd[1].cpu().numpy()}),
                  f'{prefix}.trn_norm.aux_data.csv')
        if self.has_label_num and self.regression:
            write_csv(pd.DataFrame({'name': list(self.num_label_out_names),
                                    'mean': self.num_msd[0].cpu().numpy(),
                                    'sd': self.num_msd[1].cpu().numpy()}),
                      f'{prefix}.trn_norm.labels_num.csv')

        settings = self.model_settings()
        settings['file_prefix'] = os.path.basename(prefix)
        tmp = f'{self.trn_dir}/run_settings.json.tmp'
        with open(tmp, 'w') as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, f'{self.trn_dir}/run_settings.json')
        print(f'▪ Normalisation written to {prefix}.trn_norm.attr_data.csv')

    def save_results(self):
        """Save training results.

        Saves all results from training procedure. Saved results include the
        trained network, the normalization parameters for training/calibration
        and the training history. Categorical labels are saved as model_type
        values (0 = BD, 1 = BiSSE, 2..4 = MuSSE_3..5), and a class map file
        records which model_type each network output corresponds to.

        """
        # max_idx = 1000

        path_prefix = f'{self.trn_dir}/{self.trn_prefix}'
        run_str = f'{self.num_classes}.{self.dataset_size}.{self.effective_batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.subtree_pool}.{self.extra_layers}'
        
        # output network model info
        model_arch_fn                = f'{path_prefix}.{run_str}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trained_model.pkl'
        model_history_fn             = f'{path_prefix}.{run_str}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.train_history.csv'
        class_map_fn                 = f'{path_prefix}.{run_str}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.class_map.csv'
        # model_weights_fn           = f'{path_prefix}.trn_weights.hdf5'
        
        # output scaling terms
        trn_labels_num_norm_fn    = f'{path_prefix}.{run_str}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trn_norm.labels_num.csv'
        trn_aux_data_norm_fn       = f'{path_prefix}.{run_str}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trn_norm.aux_data.csv'
        trn_attr_data_norm_fn       = f'{path_prefix}.{run_str}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trn_norm.attr_data.csv'


        # output training labels
        trn_label_num_true_fn     = f'{path_prefix}.{run_str}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trn_true.labels_num.csv'
        trn_label_num_est_fn      = f'{path_prefix}.{run_str}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trn_est.labels_num.csv'
        trn_label_est_nocalib_fn   = f'{path_prefix}.{run_str}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trn_est.labels_num_nocalib.csv'
        trn_label_cat_true_fn      = f'{path_prefix}.{run_str}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trn_true.labels_cat.csv'
        trn_label_cat_est_fn       = f'{path_prefix}.{run_str}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.trn_est.labels_cat.csv'
        trn_norm_fn       = f'{path_prefix}.{run_str}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.normilzation.csv'

        
        # save model to file
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save, model_arch_fn)
        write_mode = 'w'

        # which model_type each network output (class index) corresponds to
        if len(self.class_values) > 0:
            pd.DataFrame({'class_index': list(range(len(self.class_values))),
                          'model_type': self.class_values,
                          'model': [model_type_name(v) for v in self.class_values]}
                         ).to_csv(class_map_fn, index=False, sep=',', mode=write_mode)


        # save json history from running MASTER
        self.train_history.to_csv(model_history_fn, index=False, sep=',',
                                  float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)

        # save aux_data names, means, sd for new test dataset normalization
        self.trn_aux_data_mean_sd = (self.aux_msd[0].cpu().numpy(), self.aux_msd[1].cpu().numpy())
        df_aux_data = pd.DataFrame({
                                    'mean':self.trn_aux_data_mean_sd[0],
                                    'sd':self.trn_aux_data_mean_sd[1]})
        df_aux_data.to_csv(trn_aux_data_norm_fn, index=False, sep=',',
                           float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)

        self.trn_attr_data_mean_sd = (self.attr_msd[0].cpu().numpy(), self.attr_msd[1].cpu().numpy())
        df_aux_data = pd.DataFrame({
                                    'mean':self.trn_attr_data_mean_sd[0],
                                    'sd':self.trn_attr_data_mean_sd[1]})
        df_aux_data.to_csv(trn_attr_data_norm_fn, index=False, sep=',',
                           float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)                   
        
        # training example index
        df_trn_label_idx = pd.DataFrame(self.train_label_index, columns=['idx'])
        # print("df_trn_label_idx")
        # print(df_trn_label_idx)
 
        if self.has_label_num and self.regression:
            # save label names, means, sd for new test dataset normalization
            num_names = list(self.num_label_out_names)
            print(num_names)
            self.num_msd = (self.num_msd[0].cpu().numpy(), self.num_msd[1].cpu().numpy())
            print(self.num_msd[0])
            print(self.num_msd[1])
            df_labels = pd.DataFrame({'name':num_names,
                                      'mean':self.num_msd[0],
                                      'sd':self.num_msd[1]})
            df_labels.to_csv(trn_labels_num_norm_fn, index=False, sep=',',
                             float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)

            # true values for train numerical labels (original scale; NaN
            # where a label is not defined for the tree's model)
            df_trn_label_num_true = pd.DataFrame(self.train_label_num_true,
                                                 columns=num_names)
            df_trn_label_num_true = pd.concat([df_trn_label_idx,
                                                 df_trn_label_num_true], axis=1 )
            
            df_trn_label_num_true.to_csv(trn_label_num_true_fn,
                                            index=False, sep=',',
                                            float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
            
            # save train numerical label estimates (denormalised when the
            # network output has one column per numerical label)
            est = self.train_label_num_est
            if est.ndim == 2 and est.shape[1] == len(num_names):
                est = util.denormalize(est, self.num_msd)
                df_est = pd.DataFrame(est, columns=num_names)
            else:
                util.print_warn(f'Network output has {est.shape[-1]} columns but there '
                                f'are {len(num_names)} numerical labels; saving the '
                                f'output without denormalising')
                df_est = pd.DataFrame(est)
            self.train_label_num_est = est

            # convert to csv and save
            pd.concat([df_trn_label_idx, df_est], axis=1).to_csv(trn_label_est_nocalib_fn,
                                                    index=False, sep=',',
                                                    float_format=util.PANDAS_FLOAT_FMT_STR, mode=write_mode)
    
        if self.has_label_cat:
            # save true values for train categ. labels, as model_type values
            print(self.param_cat_names)
            df_trn_label_cat_true = pd.DataFrame(self.to_model_type(self.train_label_cat_true),
                                                   columns=self.param_cat_names )
            df_trn_label_cat_true = pd.concat([df_trn_label_idx, df_trn_label_cat_true], axis=1 )
            df_trn_label_cat_true.to_csv(trn_label_cat_true_fn,
                                           index=False, sep=',', mode=write_mode)
    
            # save train categorical label estimates, as model_type values
            if not self.regression:
                df_trn_label_cat_est = pd.DataFrame(self.to_model_type(self.train_label_cat_est),
                                                    columns=self.param_cat_names)
                df_trn_label_cat_est = pd.concat([df_trn_label_idx, df_trn_label_cat_est], axis=1 )
                df_trn_label_cat_est.to_csv(trn_label_cat_est_fn,
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