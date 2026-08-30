#!/usr/bin/env python
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm

# standard imports
import os
import sys
import threading
import time
sys.stdout.reconfigure(line_buffering=True)

# external imports
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import torch
import pickle

from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
r = robjects.r


# phyddle imports
from phyddle import utilities as util
from phyddle import network

# Re-use the block-wise HDF5 reader defined in train.py instead of loading
# the entire HDF5 file into memory. Blocks of graphs are read from disk
# on-demand, one block at a time, which keeps peak RAM usage bounded by
# block_size rather than by the size of the whole dataset.
from phyddle.train import HDF5BlockDataset


def load(args):
    """Load an Estimator object.

    This function creates an instance of the Estimator class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.

    """

    # load object
    est_method = 'default'
    if est_method == 'default':
        return GraphEmbeddingFinder(args)
    else:
        return NotImplementedError


class GraphEmbeddingFinder:
    """
    Class for extracting learned graph embeddings from a trained network
    (i.e. an intermediate-layer representation) for new (e.g. empirical or
    held-out test) phylogenetic datasets. This class requires a trained
    network from Train and input processed by Format. Test data is streamed
    from the HDF5 file in blocks (via HDF5BlockDataset), so datasets that are
    too large to fit in RAM all at once can still be processed.
    """

    def __init__(self, args):
        """Initializes a new GraphEmbeddingFinder object.

        Args:
            args (dict): Contains phyddle settings.

        """

        # settings
        self.verbose            = bool(args['verbose'])
        self.no_sim             = bool(args['no_sim'])
        self.no_emp             = bool(args['no_emp'])

        # filesystem
        self.trn_prefix         = str(args['trn_prefix'])
        self.fmt_prefix          = str(args['fmt_prefix'])
        self.est_prefix          = str(args['est_prefix'])
        self.trn_dir             = str(args['trn_dir'])
        self.fmt_dir             = str(args['fmt_dir'])
        self.est_dir             = str(args['est_dir'])
        self.sim_dir             = str(args['sim_dir'])
        self.log_dir             = str(args['log_dir'])

        self.calc_MLE            = bool(args['calc_MLE'])

        # dimensions
        self.tree_encode         = str(args['tree_encode'])
        self.char_encode         = str(args['char_encode'])
        self.brlen_encode        = str(args['brlen_encode'])
        self.tensor_format       = str(args['tensor_format'])
        self.num_char             = int(args['num_char'])
        self.num_states           = int(args['num_states'])
        self.param_est            = dict(args['param_est'])
        self.param_cat           = dict()
        self.param_cat_names     = list()
        self.param_num_names     = list()
        self.param_num            = dict()
        self.log_offset          = float(args['log_offset'])
        self.use_cuda             = bool(args['use_cuda'])

        self.regularisation      = str(args['regularisation'])
        self.dataset_size = int(args['dataset_size'])
        self.accumulation_steps = int(args['accumulation_steps'])
        self.trn_batch_size     = int(args['trn_batch_size'])
        self.effective_batch_size = self.accumulation_steps * self.trn_batch_size

        # block-wise reading settings (mirrors Trainer's block_size, so the
        # same value used at Train time can be reused at Estimate time)
        self.block_size = int(args.get('block_size', 100))
        # batch size for the DataListLoader used within each block; falls
        # back to the training batch size, then to block_size
        self.est_batch_size = int(args.get('est_batch_size',
                                            args.get('trn_batch_size', self.block_size)))

        # error checking
        self.warn_aux_outlier    = float(args['warn_aux_outlier'])
        self.warn_lbl_outlier    = float(args['warn_lbl_outlier'])
        self.est_aux_data_raw    = None
        self.est_labels_num_raw  = None

        self.phylo_pool          = bool(args['phylo_pool'])
        self.graph_conv          = bool(args['graph_conv'])
        self.extra_layers       =bool(args['extra_layers'])
        self.dropout            =float(args['dropout'])
        self.activation_func    =str(args['activation_func'])
        self.phylo_n_layers     =int(args['n_layers_phylo'])
        self.avg_n_layers = int(args['n_layers_avg'])
        self.n_layers = self.avg_n_layers
        if self.phylo_pool:     
            self.n_layers = self.phylo_n_layers
        self.phy_hidden_size = int(args['phy_hidden_size'])
        self.optimizer            = str(args['optimizer'])
        self.num_classes          = int(args['num_classes'])
        self.scheduler = "manual"

        self.learning_rate       = float(args['learning_rate'])

        # get size of CPV+S tensors
        self.num_tree_col = util.get_num_tree_col(self.tree_encode,
                                                  self.brlen_encode)
        self.num_char_col = util.get_num_char_col(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        self.num_data_col = self.num_tree_col + self.num_char_col

        # set CUDA stuff
        self.TORCH_DEVICE_STR = (
            "cuda"
            if torch.cuda.is_available() and self.use_cuda
            # else "mps"
            # if torch.backends.mps.is_available()
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)

        # cat vs. real parameter names
        self.label_num_names = [ k for k,v in self.param_est.items() if v == 'num' ]
        self.label_cat_names = [ k for k,v in self.param_est.items() if v == 'cat' ]
        self.aux_names = ["num_taxa", "age_var"]
        self.has_aux = len(self.aux_names) > 0

        self.has_label_num = len(self.label_num_names) > 0
        self.has_label_cat = len(self.label_cat_names) > 0

        # HDF5BlockDataset expects a "trainer-like" object with these
        # attributes/methods available (see separate_labels() below), so we
        # mirror Trainer's bookkeeping fields here.
        self.first_block         = True
        self.idx_num             = list()
        self.idx_cat             = list()
        self.ignore_label_num    = False

        # create logger to track runtime info
        self.logger = util.Logger(args)

        # initialized later
        self.train_aux_data_mean_sd     = None       # init in load_train_norm_stats()
        self.train_attr_data_mean_sd    = None       # init in load_train_norm_stats()
        self.train_labels_num_mean_sd   = None       # init in load_train_norm_stats()
        self.cpi_adjustments            = None       # init in load_train_input()
        self.label_names                 = None       # init in load_format_input()
        self.aux_data_names             = None       # init in load_format_input()
        self.true_labels_num             = None       # init in load_format_input()
        self.true_labels_cat             = None       # init in load_format_input()
        self.est_labels_num              = None       # init in make_results()
        self.mymodel                     = None       # init in make_results()
        self.estimate_dataset             = None       # init in load_format_input()
        self.blocks                      = None       # init in load_format_input()

        self.num_sample = -1

        # done
        return

    def separate_labels(self, input_labels, label_names=None):
        """Separates labels for categorical/numerical param_est targets.

        This mirrors Trainer.separate_labels() so that HDF5BlockDataset (which
        calls `self.trainer.separate_labels(labels, self.label_names)` once per
        block) can be reused unchanged for estimation. On the first block it
        works out which label columns are categorical/numerical, then reuses
        that column mapping for every subsequent block.

        Args:
            input_labels (numpy.ndarray): The labels for the current block.
            label_names (list): Names of the label columns, in hdf5 column order.

        Returns:
            labels_num (torch.Tensor): The numerical-valued labels for this block.
            labels_cat (torch.Tensor): The categorical labels for this block.

        """
        if label_names is None:
            return

        self.label_names = label_names
        labels = torch.from_numpy(input_labels).clone()

        if self.first_block:
            for k, v in self.param_est.items():
                if v == 'cat':
                    self.has_label_cat = True
                    idx = self.label_names.index(k)
                    self.param_cat[k] = self.num_classes
                    self.idx_cat.append(idx)
                    if k not in self.param_cat_names:
                        self.param_cat_names.append(k)

                elif v == 'num' and self.ignore_label_num == False:
                    self.has_label_num = True
                    self.idx_num.append(self.label_names.index(k))
                    if k not in self.param_num_names:
                        self.param_num_names.append(k)

            if not self.has_label_num and not self.has_label_cat:
                util.print_err(f"No training labels found.", exit=True)
            self.first_block = False

        # get data subsets
        labels_num = labels[:, self.idx_num].clone()
        labels_cat = labels[:, self.idx_cat].clone()

        # done
        return labels_num, labels_cat

    def run(self):
        """Executes all estimation tasks.

        This method prints status updates, creates the target directory for new
        estimates, then runs all estimation jobs.

        Estimation tasks are performed against all entries in the test
        dataset and against a single dataset (typically assumed to be the
        empirical dataset).

        Estimation will load the trained network, predict point estimates
        and calibrated prediction intervals (CPIs), and save results to file.

        """
        verbose = self.verbose

        # print header
        util.print_step_header('est', [self.fmt_dir, self.trn_dir], self.est_dir,
                               [self.fmt_prefix, self.trn_prefix], self.est_prefix,
                               verbose)

        # prepare workspace
        os.makedirs(self.est_dir, exist_ok=True)

        # load normalization stats (mean/sd) written out during Training, so
        # test/estimate features are standardized the same way training
        # features were
        self.load_train_norm_stats()

        # set up the block-wise HDF5 reader for the test/estimate dataset
        self.load_format_input()

        # make estimates
        util.print_str('▪ Making results', verbose)
        self.make_results()

        # done
        util.print_str('... done!', verbose)
        return

    def load_train_norm_stats(self):
        """Load the mean/sd normalization stats saved by Train.

        Train saves per-column mean/sd for node attributes (attr) and aux
        data (aux) so that new data can be standardized in exactly the same
        way training data was. This looks for the CSVs Train writes out,
        using the same filename convention as train.py's save_results().
        If they can't be found, embeddings will be computed on raw,
        unnormalized inputs, and a warning is printed.

        """
        path_prefix = (f'{self.trn_dir}/{self.trn_prefix}.{self.num_classes}.'
                        f'{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.'
                        f'{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.'
                        f'{self.regularisation}')

        trn_aux_data_norm_fn  = f'{path_prefix}.trn_norm.aux_data.csv'
        trn_attr_data_norm_fn = f'{path_prefix}.trn_norm.attr_data.csv'

        try:
            df_aux = pd.read_csv(trn_aux_data_norm_fn)
            self.train_aux_data_mean_sd = (
                torch.tensor(df_aux['mean'].to_numpy(), dtype=torch.float32),
                torch.tensor(df_aux['sd'].to_numpy(), dtype=torch.float32)
            )
        except FileNotFoundError:
            util.print_warn(f'Could not find {trn_aux_data_norm_fn}; '
                             f'aux data will not be normalized.')

        try:
            df_attr = pd.read_csv(trn_attr_data_norm_fn)
            self.train_attr_data_mean_sd = (
                torch.tensor(df_attr['mean'].to_numpy(), dtype=torch.float32),
                torch.tensor(df_attr['sd'].to_numpy(), dtype=torch.float32)
            )
        except FileNotFoundError:
            util.print_warn(f'Could not find {trn_attr_data_norm_fn}; '
                             f'node attributes will not be normalized.')

        return

    def load_format_input(self):
        """Set up block-wise loading of input data for estimation.

        Rather than reading the entire test/estimate HDF5 file into memory at
        once, this builds an HDF5BlockDataset (the same class Train uses)
        over the file, split into contiguous blocks of `self.block_size`
        graphs. Each block is only read from disk when it's requested (see
        make_results()), so peak memory use is bounded by block_size rather
        than the full dataset size.

        """

        short_path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.test'
        hdf5_fn = f'{short_path_prefix}.hdf5'

        # only need the total number of graphs up front, to build blocks;
        # everything else is read lazily by HDF5BlockDataset
        with h5py.File(hdf5_fn, 'r') as f:
            N = f['phy_data'].shape[0]

        self.num_sample = N

        # build contiguous blocks covering every graph in the dataset (no
        # shuffling needed here, unlike training)
        self.blocks = [np.arange(i, min(i + self.block_size, N))
                        for i in range(0, N, self.block_size)]

        util.print_str(f'  ▪ {N} test examples split into '
                       f'{len(self.blocks)} block(s) of up to {self.block_size}',
                       self.verbose)

        self.estimate_dataset = HDF5BlockDataset(hdf5_fn, self.blocks, self)

        # small metadata that HDF5BlockDataset already reads fully into RAM
        self.label_names = self.estimate_dataset.label_names
        self.aux_data_names = self.estimate_dataset.aux_data_names

        # done
        return

    def hook_fn(self, module, input, output):
        self.embeddings.append(output.detach().cpu())

    def make_results(self):
        """Runs the trained model over the test/estimate dataset one block at
        a time, capturing intermediate-layer graph embeddings via a forward
        hook, and writes the collected embeddings to file.

        """

        path_prefix = (f'{self.trn_dir}/{self.trn_prefix}.{self.num_classes}.{self.dataset_size}.{self.effective_batch_size}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.extra_layers}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}')
        model_arch_fn = f'{path_prefix}.trained_model.pkl'

        self.embeddings = []
        graph_idx_out = []

        self.mymodel = torch.load(model_arch_fn, map_location="cpu", weights_only=False)
        self.mymodel.to(self.TORCH_DEVICE)
        self.mymodel.eval()

        handle = self.mymodel.phy_std.gconv4.register_forward_hook(self.hook_fn)

        n_blocks = len(self.estimate_dataset)

        with torch.no_grad():
            for j in tqdm(range(n_blocks), total=n_blocks, desc='Blocks'):

                # read one block of graphs from disk
                graph_list = self.estimate_dataset[j]

                # further split the block into forward-pass-sized batches
                loader = DataListLoader(graph_list, batch_size=self.est_batch_size)

                for batch in loader:

                    # standardize inputs using the training set's mean/sd,
                    # exactly as done during training
                    for g in batch:
                        if self.train_attr_data_mean_sd is not None:
                            g.x = util.normalize(g.x, self.train_attr_data_mean_sd)
                        if self.train_aux_data_mean_sd is not None:
                            g.aux_dat = util.normalize(g.aux_dat, self.train_aux_data_mean_sd)

                    n_before = len(self.embeddings)

                    _ = self.mymodel(batch)

                    if len(self.embeddings) > n_before:
                        util.print_str(
                            f'Block {j}, embedding shape: {self.embeddings[-1].shape}',
                            self.verbose)

                    graph_idx_out.extend([g.idx.item() if torch.is_tensor(g.idx) else g.idx
                                           for g in batch])

        handle.remove()

        # save the collected embeddings alongside the graph indices they
        # correspond to
        out_fn = f'{self.est_dir}/{self.est_prefix}.graph_embeddings.pkl'
        with open(out_fn, 'wb') as f:
            pickle.dump({'idx': graph_idx_out, 'embeddings': self.embeddings}, f)
        util.print_str(f'  ▪ Wrote graph embeddings to {out_fn}', self.verbose)

        # done
        return
