#!/usr/bin/env python
"""
get_graph_embeddings
======
Extracts embeddings for inputted data by processing them on trained networks from the first layer until the final graph convolutional layer.

The trained network and the dataset(s) to embed no longer need to come from
the same phyddle project. By default the script behaves exactly as before
(model resolved from trn_dir/trn_prefix + hyperparameters, data read from
fmt_dir/fmt_prefix.{train,test}.hdf5), but any of the following optional
settings can be supplied to point at a different model and/or dataset:

    trained_model      (existing registry setting) Path to a trained model
                       file (*.trained_model.pkl). Overrides the
                       trn_dir/trn_prefix/hyperparameter lookup.
    emb_norm_prefix    Path prefix of the Train normalization CSVs
                       ({prefix}.trn_norm.aux_data.csv / .attr_data.csv).
                       Defaults to the model's own prefix.
    test_hdf5          (existing registry setting) HDF5 file to embed, or
                       several as a comma-separated string. When set, only
                       these files are read (not fmt_prefix.train/test).
    emb_fmt_dir        Format directory of the dataset to embed
                       (default: fmt_dir).
    emb_fmt_prefix     Format prefix of the dataset to embed
                       (default: fmt_prefix).
    emb_data_splits    Which {prefix}.{split}.hdf5 files to read from
                       emb_fmt_dir (default: "train,test").
    emb_out_tag        Tag added to the output filenames to identify the
                       dataset (default: derived from the dataset source).
    emb_require_labels If True, exit when a param_est label is missing from
                       the dataset. If False (default), missing labels are
                       ignored, since embeddings don't need labels (useful
                       for empirical data).

Author:  Kate Truman, and AI tools (e.g. Claude and ChatGPT). Based on existing Phyddle code by Michael Landis and Ammon Thompson
"""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm

# standard imports
import os
import glob
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
import csv

from torch_geometric.data import Data as GeoData, Batch as GeoBatch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.nn import GCNConv


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


MODEL_SUFFIX = '.trained_model.pkl'


class _StopForward(Exception):
    """Raised by the forward hook once the final GCN layer has run.

    Embeddings only need the network up to its final graph convolutional
    layer, so the forward pass is stopped there. This means the pooling and
    dense layers after it are never executed, which avoids shape errors when
    those layers differ between the network.py used for training and the
    current one.
    """
    pass


def _load_network_module(network_fn, module_name='phyddle.network'):
    """Import a specific network.py file under the name phyddle.network.

    torch.load() of a pickled model restores its saved weights and
    submodules, but takes the class definitions (including forward()) from
    whatever module the class name points to at load time. If network.py has
    changed since training, the current forward() may not fit the saved
    layers. Registering the old file under the same module name before
    loading makes pickle use the old class definitions.

    Returns:
        The previously registered module (or None), so it can be restored.
    """
    import importlib.util
    if not os.path.exists(network_fn):
        util.print_err(f'trained_network file {network_fn} does not exist.',
                       exit=True)
    spec = importlib.util.spec_from_file_location(module_name, network_fn)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name, None)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return previous


def _as_list(value):
    """Turn None / a comma-separated string / an iterable into a list of str."""
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(',') if s.strip()]
    return [str(s) for s in value]


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
    network from Train and input processed by Format. The trained network and
    the input data may come from different phyddle projects (see the emb_*
    settings in the module docstring). Data is streamed from the HDF5 file(s)
    in blocks (via HDF5BlockDataset), so datasets that are too large to fit in
    RAM all at once can still be processed.
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

        # decoupled model / dataset sources (all optional; see module
        # docstring). When none are given, behaviour matches the original
        # script exactly.
        # 'trained_model' is the registered setting; 'emb_model_fn' is
        # accepted too, but only reaches here if it is also registered
        self.emb_model_fn        = (args.get('trained_model', None) or
                                    args.get('emb_model_fn', None))
        self.emb_norm_prefix     = args.get('emb_norm_prefix', None)
        self.emb_data_fn         = _as_list(args.get('test_hdf5', None))
        self.emb_fmt_dir         = str(args.get('emb_fmt_dir', None) or self.fmt_dir)
        self.emb_fmt_prefix      = str(args.get('emb_fmt_prefix', None) or self.fmt_prefix)
        self.emb_data_splits     = _as_list(args.get('emb_data_splits', None)) or ['train', 'test']
        self.emb_out_tag         = args.get('emb_out_tag', None)
        self.emb_require_labels  = bool(args.get('emb_require_labels', False))
        # network.py the model was trained with, if it has changed since
        self.trained_network     = args.get('trained_network', None)

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
        # label column indices, cached per distinct label_names layout, so
        # datasets from different sources (possibly with different label
        # columns/orders) are each handled correctly
        self._label_idx_cache    = dict()

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
        self._trn_path_prefix            = None       # cached by get_trn_path_prefix()
        self.estimate_datasets            = None       # init in load_format_input()
        self.data_tag                    = None       # init in load_format_input()

        self.num_sample = -1

        # done
        return

    def separate_labels(self, input_labels, label_names=None):
        """Separates labels for categorical/numerical param_est targets.

        This mirrors Trainer.separate_labels() so that HDF5BlockDataset (which
        calls `self.trainer.separate_labels(labels, self.label_names)` once per
        block) can be reused unchanged for estimation. Column indices are
        worked out once per distinct label_names layout and cached, so
        datasets coming from different Format runs can be mixed safely.

        Labels aren't needed to compute embeddings, so param_est labels that
        are missing from the dataset (e.g. empirical data) are skipped with a
        warning, unless emb_require_labels is set.

        Args:
            input_labels (numpy.ndarray): The labels for the current block.
            label_names (list): Names of the label columns, in hdf5 column order.

        Returns:
            labels_num (torch.Tensor): The numerical-valued labels for this block.
            labels_cat (torch.Tensor): The categorical labels for this block.

        """
        if label_names is None:
            return

        label_names = list(label_names)
        self.label_names = label_names
        key = tuple(label_names)

        if key not in self._label_idx_cache:
            idx_num, idx_cat, missing = [], [], []
            for k, v in self.param_est.items():
                if k not in label_names:
                    missing.append(k)
                    continue
                if v == 'cat':
                    self.has_label_cat = True
                    self.param_cat[k] = self.num_classes
                    idx_cat.append(label_names.index(k))
                    if k not in self.param_cat_names:
                        self.param_cat_names.append(k)
                elif v == 'num' and self.ignore_label_num == False:
                    self.has_label_num = True
                    idx_num.append(label_names.index(k))
                    if k not in self.param_num_names:
                        self.param_num_names.append(k)

            if len(missing) > 0:
                msg = (f'Labels {missing} from param_est not found in dataset '
                       f'labels {label_names}.')
                if self.emb_require_labels:
                    util.print_err(msg, exit=True)
                else:
                    util.print_warn(msg + ' Ignoring them (labels are not '
                                    'needed to compute embeddings).')

            self._label_idx_cache[key] = (idx_num, idx_cat)
            self.first_block = False

        idx_num, idx_cat = self._label_idx_cache[key]
        self.idx_num = idx_num
        self.idx_cat = idx_cat

        # get data subsets
        labels = torch.from_numpy(input_labels).clone()
        labels_num = labels[:, idx_num].clone()
        labels_cat = labels[:, idx_cat].clone()

        # done
        return labels_num, labels_cat

    def run(self):
        """Executes all embedding-extraction tasks.

        This method prints status updates, creates the target directory for
        outputs, loads the trained network's normalization stats, sets up the
        dataset(s) to embed, then extracts and saves embeddings.

        """
        verbose = self.verbose

        # print header
        util.print_step_header('est', [self.fmt_dir, self.trn_dir], self.est_dir,
                               [self.fmt_prefix, self.trn_prefix], self.est_prefix,
                               verbose)

        # prepare workspace
        os.makedirs(self.est_dir, exist_ok=True)

        util.print_str(f'  ▪ Trained model: {self.get_model_fn()}', verbose)

        # load normalization stats (mean/sd) written out during Training, so
        # new features are standardized the same way training features were
        self.load_train_norm_stats()

        # set up the block-wise HDF5 reader(s) for the dataset(s) to embed
        self.load_format_input()

        # make estimates
        util.print_str('▪ Making results', verbose)
        self.make_results()

        # done
        util.print_str('... done!', verbose)
        return

    def get_trn_path_prefix(self):
        """Resolve the Train output path_prefix.

        If emb_model_fn is given, the prefix is that path with the
        '.trained_model.pkl' suffix removed, so the trained network can live
        anywhere (e.g. in a different phyddle project to the data).

        Otherwise, Train's saved filenames normally include num_classes. This
        tries that version first; if the corresponding trained model file
        isn't found (e.g. num_classes wasn't part of the filename when this
        model was trained), it falls back to a path_prefix built without
        num_classes. The result is cached so every caller (norm stats, model
        loading, output naming) agrees on the same prefix.

        Returns:
            str: The path_prefix to use for this run's Train outputs.

        """
        if self._trn_path_prefix is not None:
            return self._trn_path_prefix

        # explicit model file
        if self.emb_model_fn:
            model_fn = str(self.emb_model_fn)
            if not os.path.exists(model_fn):
                util.print_err(f'Trained model file emb_model_fn={model_fn} '
                               f'does not exist.', exit=True)
            if model_fn.endswith(MODEL_SUFFIX):
                self._trn_path_prefix = model_fn[:-len(MODEL_SUFFIX)]
            else:
                self._trn_path_prefix = os.path.splitext(model_fn)[0]
            return self._trn_path_prefix

        prefix_with_classes = (
            f'{self.trn_dir}/{self.trn_prefix}.{self.num_classes}.{self.dataset_size}.'
            f'{self.effective_batch_size}.{self.optimizer}.{self.scheduler}.'
            f'{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.'
            f'{self.extra_layers}.{self.n_layers}.{self.learning_rate}.'
            f'{self.dropout}.{self.activation_func}.{self.regularisation}')
        if os.path.exists(f'{prefix_with_classes}{MODEL_SUFFIX}'):
            self._trn_path_prefix = prefix_with_classes
            return self._trn_path_prefix

        prefix_without_classes = (
            f'{self.trn_dir}/{self.trn_prefix}.{self.dataset_size}.'
            f'{self.effective_batch_size}.{self.optimizer}.{self.scheduler}.'
            f'{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.'
            f'{self.extra_layers}.{self.n_layers}.{self.learning_rate}.'
            f'{self.dropout}.{self.activation_func}.{self.regularisation}')
        if os.path.exists(f'{prefix_without_classes}{MODEL_SUFFIX}'):
            util.print_warn(
                f'Trained model not found at {prefix_with_classes}{MODEL_SUFFIX}; '
                f'using path_prefix without num_classes instead.')
            self._trn_path_prefix = prefix_without_classes
            return self._trn_path_prefix

        # neither exists; return the num_classes version so downstream
        # file reads raise a clear, informative error
        self._trn_path_prefix = prefix_with_classes
        return self._trn_path_prefix

    def get_model_fn(self):
        """Path to the trained model file to load."""
        if self.emb_model_fn:
            return str(self.emb_model_fn)
        return f'{self.get_trn_path_prefix()}{MODEL_SUFFIX}'

    def get_norm_prefix(self):
        """Path prefix for the Train normalization CSVs.

        Defaults to the trained model's prefix, since Train writes the norm
        stats alongside the model. Can be overridden with emb_norm_prefix if
        the model file has been moved/renamed separately from its stats.
        """
        if self.emb_norm_prefix:
            return str(self.emb_norm_prefix)
        return self.get_trn_path_prefix()

    def find_norm_file(self, kind):
        """Find the Train normalization CSV for 'aux' or 'attr' data.

        Looks in the same folder as the trained model (or emb_norm_prefix,
        if set). First tries exact names built from the model's prefix, then
        falls back to searching that folder for a CSV whose name contains
        'norm' and the data kind, preferring files that share the model's
        filename prefix.

        Args:
            kind (str): 'aux' or 'attr'.

        Returns:
            str or None: Path to the file, or None if not found.

        """
        path_prefix = self.get_norm_prefix()
        norm_dir = os.path.dirname(path_prefix) or '.'
        base = os.path.basename(path_prefix)
        trn_prefix = base.split('.')[0]

        # exact candidates, most specific first
        candidates = [
            f'{path_prefix}.trn_norm.{kind}_data.csv',
            f'{path_prefix}.train_norm.{kind}_data.csv',
            f'{path_prefix}.train_{kind}_data_norm.csv',
            f'{norm_dir}/{trn_prefix}.trn_norm.{kind}_data.csv',
            f'{norm_dir}/{trn_prefix}.train_{kind}_data_norm.csv',
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn

        # fall back to searching the folder
        matches = sorted(set(
            glob.glob(f'{norm_dir}/*norm*{kind}*.csv') +
            glob.glob(f'{norm_dir}/*{kind}*norm*.csv')))
        if len(matches) == 0:
            return None

        # prefer files from this exact model, then this trn_prefix
        for pref in [base, trn_prefix]:
            same = [m for m in matches if os.path.basename(m).startswith(pref + '.')]
            if len(same) == 1:
                return same[0]
            if len(same) > 1:
                matches = same
                break

        if len(matches) > 1:
            util.print_warn(f'Several {kind} normalization files found in '
                            f'{norm_dir}: {[os.path.basename(m) for m in matches]}. '
                            f'Using {os.path.basename(matches[0])}; set '
                            f'emb_norm_prefix to choose a different one.')
        return matches[0]

    def load_train_norm_stats(self):
        """Load the mean/sd normalization stats saved by Train.

        Train saves per-column mean/sd for node attributes (attr) and aux
        data (aux) so that new data can be standardized in exactly the same
        way training data was. These always come from the *trained model's*
        Train run (not the dataset being embedded), since the network expects
        inputs on its own training scale, and are looked for in the same
        folder as the trained model (see find_norm_file()). If they can't be
        found, embeddings will be computed on raw, unnormalized inputs, and a
        warning is printed.

        """
        norm_dir = os.path.dirname(self.get_norm_prefix()) or '.'

        for kind, attr_name in [('aux', 'train_aux_data_mean_sd'),
                                ('attr', 'train_attr_data_mean_sd')]:
            fn = self.find_norm_file(kind)
            if fn is None:
                util.print_warn(f'No {kind} normalization file found in '
                                f'{norm_dir}; {kind} data will not be '
                                f'normalized.')
                continue
            df = pd.read_csv(fn)
            if 'mean' not in df.columns or 'sd' not in df.columns:
                util.print_warn(f'{fn} has no mean/sd columns '
                                f'(columns: {list(df.columns)}); {kind} data '
                                f'will not be normalized.')
                continue
            setattr(self, attr_name, (
                torch.tensor(df['mean'].to_numpy(), dtype=torch.float32),
                torch.tensor(df['sd'].to_numpy(), dtype=torch.float32)))
            util.print_str(f'  ▪ Loaded {kind} normalization stats from {fn}',
                           self.verbose)

        return

    def get_data_sources(self):
        """List the (source_name, hdf5_fn, required) entries to embed.

        Uses emb_data_fn if given; otherwise
        {emb_fmt_dir}/{emb_fmt_prefix}.{split}.hdf5 for each split in
        emb_data_splits (default: train, test, under fmt_dir/fmt_prefix).

        Returns:
            list: (source_name, hdf5_fn, required) tuples. Missing files are
                an error if required, otherwise skipped with a warning.

        """
        sources = []
        if len(self.emb_data_fn) > 0:
            seen = dict()
            for fn in self.emb_data_fn:
                name = os.path.basename(fn)
                if name.endswith('.hdf5'):
                    name = name[:-len('.hdf5')]
                # keep source names unique if two files share a basename
                if name in seen:
                    seen[name] += 1
                    name = f'{name}_{seen[name]}'
                else:
                    seen[name] = 0
                sources.append((name, fn, True))
        else:
            for split in self.emb_data_splits:
                fn = f'{self.emb_fmt_dir}/{self.emb_fmt_prefix}.{split}.hdf5'
                sources.append((split, fn, False))
        return sources

    def get_data_tag(self):
        """Tag identifying the embedded dataset in output filenames.

        Returns None (no tag, i.e. the original filenames) when the default
        fmt_dir/fmt_prefix data is used, so existing workflows are unchanged.
        """
        if self.emb_out_tag:
            return str(self.emb_out_tag)
        if len(self.emb_data_fn) > 0:
            return '+'.join(name for name, _, _ in self.get_data_sources())
        if (self.emb_fmt_dir != self.fmt_dir or
                self.emb_fmt_prefix != self.fmt_prefix):
            return self.emb_fmt_prefix
        return None

    def load_format_input(self):
        """Set up block-wise loading of input data for embedding.

        By default, reads the train and test HDF5 files written by Format
        under fmt_dir/fmt_prefix. Alternatively, any HDF5 file(s) produced
        by Format (possibly from a different project than the one the
        network was trained in) can be given via emb_data_fn or
        emb_fmt_dir/emb_fmt_prefix.

        Each file gets its own HDF5BlockDataset (the same class Train uses),
        split into contiguous blocks of `self.block_size` graphs. Each block
        is only read from disk when it's requested (see make_results()), so
        peak memory use is bounded by block_size rather than the full
        dataset size.

        """

        self.estimate_datasets = []
        self.num_sample = 0

        for data_str, hdf5_fn, required in self.get_data_sources():

            if not os.path.exists(hdf5_fn):
                if required:
                    util.print_err(f'Could not find {hdf5_fn}.', exit=True)
                util.print_warn(f'Could not find {hdf5_fn}; skipping.')
                continue

            # only need the total number of graphs up front, to build
            # blocks; everything else is read lazily by HDF5BlockDataset
            with h5py.File(hdf5_fn, 'r') as f:
                N = f['phy_data'].shape[0]

            # build contiguous blocks covering every graph in this file (no
            # shuffling needed here, unlike training)
            blocks = [np.arange(i, min(i + self.block_size, N))
                      for i in range(0, N, self.block_size)]

            util.print_str(f'  ▪ {hdf5_fn}: {N} {data_str} examples split '
                           f'into {len(blocks)} block(s) of up to '
                           f'{self.block_size}', self.verbose)

            dataset = HDF5BlockDataset(hdf5_fn, blocks, self)
            self.estimate_datasets.append((data_str, dataset))
            self.num_sample += N

        if len(self.estimate_datasets) == 0:
            util.print_err('No HDF5 files found to compute embeddings for. '
                           'Checked: ' +
                           ', '.join(fn for _, fn, _ in self.get_data_sources()),
                           exit=True)

        util.print_str(f'  ▪ {self.num_sample} total examples across '
                       f'{len(self.estimate_datasets)} file(s)', self.verbose)

        # small metadata that HDF5BlockDataset already reads fully into RAM;
        # taken from whichever dataset was built first, with a warning if
        # files from different sources disagree
        first_name, first_dataset = self.estimate_datasets[0]
        self.label_names = first_dataset.label_names
        self.aux_data_names = first_dataset.aux_data_names
        for name, ds in self.estimate_datasets[1:]:
            if list(ds.aux_data_names) != list(self.aux_data_names):
                util.print_warn(f'aux_data_names differ between {first_name} '
                                f'({list(self.aux_data_names)}) and {name} '
                                f'({list(ds.aux_data_names)}).')

        self.data_tag = self.get_data_tag()

        # done
        return

    def check_compatibility(self, gconv_layers):
        """Check each dataset's input dimensions match the trained network.

        When the model and data come from different projects, a mismatch in
        encodings (tree/char/brlen encode, num_char, num_states) would
        otherwise only show up as a cryptic shape error mid-run. This reads
        the first graph of each dataset and compares its node-attribute and
        aux-data widths to the training normalization stats and to the first
        graph conv layer.

        """
        first_gconv = gconv_layers[0][1] if len(gconv_layers) > 0 else None
        gconv_in = getattr(first_gconv, 'in_channels', -1)

        for data_str, dataset in self.estimate_datasets:
            if len(dataset) == 0:
                continue
            graph_list = dataset[0]
            if len(graph_list) == 0:
                continue
            g = graph_list[0]

            n_attr = g.x.shape[-1]
            if self.train_attr_data_mean_sd is not None:
                n_expect = self.train_attr_data_mean_sd[0].shape[0]
                if n_attr != n_expect:
                    util.print_err(
                        f'Dataset "{data_str}" has {n_attr} node attribute '
                        f'columns but the trained network expects {n_expect}. '
                        f'Check tree_encode/char_encode/brlen_encode/num_char/'
                        f'num_states match the settings used for Train.',
                        exit=True)
            elif gconv_in is not None and gconv_in > 0 and n_attr != gconv_in:
                # without norm stats we can only compare to the first GCN
                # layer, which may not be the network's input layer
                util.print_warn(
                    f'Dataset "{data_str}" has {n_attr} node attribute '
                    f'columns; first GCNConv layer expects {gconv_in}.')

            if self.train_aux_data_mean_sd is not None and hasattr(g, 'aux_dat'):
                n_aux = g.aux_dat.shape[-1]
                n_expect = self.train_aux_data_mean_sd[0].shape[0]
                if n_aux != n_expect:
                    util.print_err(
                        f'Dataset "{data_str}" has {n_aux} aux data columns '
                        f'but the trained network expects {n_expect}.',
                        exit=True)
        return

    def hook_fn(self, module, input, output):
        self.embeddings.append(output.detach().cpu())
        # nothing after the final GCN layer is needed, so stop here
        raise _StopForward()

    def make_results(self):
        """Runs the trained model over the dataset(s) one block at a time,
        capturing intermediate-layer graph embeddings via a forward hook, and
        writes the collected embeddings to file.

        """

        model_arch_fn = self.get_model_fn()

        self.embeddings = []
        graph_idx_out = []
        graph_src_out = []

        # (source, idx, flat_embedding) for every graph, built up
        # incrementally as each block is processed
        graph_records = []

        if not os.path.exists(model_arch_fn):
            util.print_err(f'Trained model file {model_arch_fn} not found.',
                           exit=True)

        # if the model was trained with an older network.py, load the
        # pickled model against that file's class definitions
        if self.trained_network:
            util.print_str(f'  ▪ Using network definition from '
                           f'{self.trained_network}', self.verbose)
            previous = _load_network_module(self.trained_network)
            try:
                self.mymodel = torch.load(model_arch_fn, map_location="cpu",
                                          weights_only=False)
            finally:
                # restore the current phyddle.network for everything else;
                # the loaded model keeps its reference to the old classes
                if previous is not None:
                    sys.modules['phyddle.network'] = previous
        else:
            self.mymodel = torch.load(model_arch_fn, map_location="cpu",
                                      weights_only=False)
        self.mymodel.to(self.TORCH_DEVICE)
        self.mymodel.eval()

        if self.verbose:
            print(self.mymodel)

        # Get all top-level layers in phy_std
        layers = list(self.mymodel.phy_std.named_children())

        # Find all GCNConv layers
        gconv_layers = [
            (name, module)
            for name, module in layers
            if isinstance(module, GCNConv)
        ]
        if len(gconv_layers) == 0:
            util.print_err(f'No GCNConv layers found in phy_std of '
                           f'{model_arch_fn}.', exit=True)

        # Final GCN layer
        name, final_gconv = gconv_layers[-1]

        util.print_str(f'  ▪ Extracting embeddings from final GCN layer: '
                       f'{name} ({final_gconv})', self.verbose)

        # make sure the data fits the network before the long run
        self.check_compatibility(gconv_layers)

        handle = final_gconv.register_forward_hook(self.hook_fn)

        total_blocks = sum(len(dataset) for _, dataset in self.estimate_datasets)

        with torch.no_grad():
            pbar = tqdm(total=total_blocks, desc='Blocks')
            for data_str, dataset in self.estimate_datasets:
                for j in range(len(dataset)):

                    # read one block of graphs from disk
                    graph_list = dataset[j]

                    # further split the block into forward-pass-sized batches
                    loader = DataListLoader(graph_list, batch_size=self.est_batch_size)

                    for batch in loader:

                        # standardize inputs using the *trained model's*
                        # training-set mean/sd, exactly as done during training
                        for g in batch:
                            if self.train_attr_data_mean_sd is not None:
                                g.x = util.normalize(g.x, self.train_attr_data_mean_sd)
                            if self.train_aux_data_mean_sd is not None:
                                g.aux_dat = util.normalize(g.aux_dat, self.train_aux_data_mean_sd)

                        n_before = len(self.embeddings)

                        batched = GeoBatch.from_data_list(batch).to(self.TORCH_DEVICE)

                        try:
                            _ = self.mymodel(batched)
                        except _StopForward:
                            pass
                        except RuntimeError as e:
                            if 'shapes cannot be multiplied' in str(e):
                                util.print_err(
                                    f'{e}\nThe network failed before reaching '
                                    f'its final GCN layer. This usually means '
                                    f'network.py has changed since the model '
                                    f'was trained: set trained_network to the '
                                    f'network.py used for training, and check '
                                    f'the dataset encoding matches.', exit=True)
                            raise

                        batch_idx = [g.idx.item() if torch.is_tensor(g.idx) else g.idx
                                     for g in batch]

                        if len(self.embeddings) > n_before:
                            # The last graph convolutional layer's output is one row per NODE (not per
                            # graph), stacked across every graph in this batch.
                            # batched.batch tells us which graph each row
                            # belongs to (0-indexed, in the same order as
                            # `batch`/`batch_idx`), so split the node-level
                            # output back out per graph.
                            node_emb = self.embeddings[-1]
                            node_batch_assign = batched.batch.cpu()
                            node_counts = torch.bincount(
                                node_batch_assign, minlength=len(batch_idx)
                            ).tolist()
                            per_graph_node_emb = torch.split(node_emb, node_counts, dim=0)

                            for g_idx, g_emb in zip(batch_idx, per_graph_node_emb):
                                # flatten this graph's [num_nodes, hidden_dim]
                                # embedding into a single 1-D vector
                                graph_records.append((data_str, g_idx, g_emb.reshape(-1)))

                        graph_idx_out.extend(batch_idx)
                        graph_src_out.extend([data_str] * len(batch_idx))

                    pbar.update(1)
            pbar.close()

        handle.remove()

        if len(graph_records) == 0:
            util.print_err('No embeddings were captured; nothing to write.',
                           exit=True)

        # tag output filenames with the Train path_prefix (model config) and,
        # when non-default data is used, the dataset tag, so embeddings from
        # different model/dataset combinations don't overwrite each other
        out_base = os.path.basename(self.get_trn_path_prefix())
        if self.data_tag:
            out_base = f'{out_base}.{self.data_tag}'

        # save the raw, un-padded embeddings (one tensor per batch) as well,
        # for anyone who wants the unprocessed node-level output
        raw_out_fn = f'{self.est_dir}/{out_base}.graph_embeddings.pkl'
        with open(raw_out_fn, 'wb') as f:
            pickle.dump({'idx': graph_idx_out,
                         'dataset': graph_src_out,
                         'model_fn': model_arch_fn,
                         'embeddings': self.embeddings}, f)
        util.print_str(f'  ▪ Wrote raw graph embeddings to {raw_out_fn}', self.verbose)

        # write one row per graph: flattened node embeddings, zero-padded to
        # the length of the largest graph. The 'dataset' column records which
        # file each graph came from, since graph_idx can repeat across files.
        df_out_fn = f'{self.est_dir}/{out_base}.graph_embeddings.csv'
        max_len = max(len(g_emb) for _, _, g_emb in graph_records)
        n_graphs = len(graph_records)
        report_every = max(1, n_graphs // 20)  # ~20 updates total

        with open(df_out_fn, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['graph_idx', 'dataset'] + [f'f{i}' for i in range(max_len)])
            for i, (src, g_idx, g_emb) in enumerate(graph_records):
                row = np.zeros(max_len, dtype=np.float32)
                flat = np.asarray(g_emb, dtype=np.float32)
                row[:flat.shape[0]] = flat
                writer.writerow([g_idx, src] + row.tolist())
                if (i + 1) % report_every == 0 or (i + 1) == n_graphs:
                    util.print_str(f'    ... wrote {i + 1}/{n_graphs} graphs', self.verbose)

        util.print_str(f'  ▪ Wrote padded per-graph embedding dataframe '
                    f'({n_graphs} graphs x {max_len} features) '
                    f'to {df_out_fn}', self.verbose)
        # done
        return

    def build_embedding_dataframe(self, graph_records):
        """Assemble a padded, one-row-per-graph embedding dataframe.

        Args:
            graph_records (list): (dataset, graph_idx, flat_embedding) tuples,
                one per graph, where flat_embedding is a 1-D torch.Tensor.
                Graphs have different numbers of nodes, so these vectors can
                be different lengths.

        Returns:
            pandas.DataFrame: One row per graph. The first columns, 'idx' and
                'dataset', hold the graph index and its source file. The
                remaining columns hold the flattened, zero-padded node
                embeddings (padded on the right up to the longest embedding
                in the dataset).

        """
        if len(graph_records) == 0:
            return pd.DataFrame(columns=['idx', 'dataset'])

        max_len = max(flat.shape[0] for _, _, flat in graph_records)

        idx_col = []
        src_col = []
        padded_rows = []
        for src, g_idx, flat in graph_records:
            pad_len = max_len - flat.shape[0]
            if pad_len > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_len), value=0.0)
            idx_col.append(g_idx)
            src_col.append(src)
            padded_rows.append(flat.numpy())

        df = pd.DataFrame(padded_rows)
        df.insert(0, 'dataset', src_col)
        df.insert(0, 'idx', idx_col)

        return df