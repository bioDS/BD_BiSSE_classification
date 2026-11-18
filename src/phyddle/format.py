#!/usr/bin/env python
"""
format
======
Defines classes and methods for the Format step. This step converts raw data
into tensor data that can be used by the Train step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2025, Michael Landis and Ammon Thompson
License:   MIT

# Edits by Kate Truman, including supervision of AI tools to edit code and comments. 
# Changes made in order to adapt phyddle for training graph neural networks on binary classification of birth-death models.
# Packed-simulation support. Expected layout:
#     <packed_in_dir>/BD_div/train.nwk, train.txt, test.nwk, test.txt
#     <packed_in_dir>/BiSSE/ train.nwk, train.txt, test.nwk, test.txt
# and optionally
#     <packed_in_dir>/MuSSE_3/ train.nwk, train.txt, test.nwk, test.txt
#     <packed_in_dir>/MuSSE_4/ train.nwk, train.txt, test.nwk, test.txt
#     <packed_in_dir>/MuSSE_5/ train.nwk, train.txt, test.nwk, test.txt
# (.nwk = trees, .txt = parameters). These are unpacked into
# out.N.tre / out.N.labels.csv / out.N.dat.csv before encoding.
# model_type labels: BD_div = 0, BiSSE = 1, MuSSE_3 = 2, MuSSE_4 = 3, MuSSE_5 = 4.
# A model folder is only used if its model_type is in select_classes and the
# folder exists. num_states must be at least the largest number of states
# among the selected models (e.g. 5 when MuSSE_5 is selected).
# Every selected model_type must have a folder, unless single_class = K is
# set: then only model_type K is unpacked and formatted (e.g. single_class = 1
# for a BiSSE-only test set) and the other folders may be missing.
# Set skip_unpack = True to reuse previously unpacked out.N.* files.
#
# Graph node encodings (encoding_type): 'NC', '1N', '2N', any other name
# (R's fallback 3-feature set), or 'all' to format every encoding in one
# pass with the same trees and the same train/test split.
# HDF5 output is saved as <fmt_prefix>.<split>.<encoding>.hdf5, or with
# max_model_states = K as <fmt_prefix>.<split>.<encoding>.statesK.hdf5
# (e.g. out.train.2N.states3.hdf5); an existing file of that name is first
# moved to <...>.vN.hdf5.
"""
# standard imports
# import copy
import os
import re
import sys

from collections import defaultdict

import awkward as ak

# external imports
import dendropy as dp
import h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool, set_start_method, cpu_count
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Want helper functions for converting phylogenies to graphs from https://github.com/ismael-lajaaiti/phylo-inference-ml
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
r = robjects.r
r['source']('~/AIphylo/phylo-inference-ml/R/convert-phylo-to-cblv.R')
r['source']('~/AIphylo/phylo-inference-ml/R/convert-phylo-to-graph.R')
ape = importr('ape')
castor = importr('castor')
phangorn = importr('phangorn')
MLmetrics = importr('MLmetrics')
dplyr = importr('dplyr')
diversitree = importr('diversitree')
RPANDA = importr('RPANDA')
latex2exp = importr('latex2exp')
svMisc = importr('svMisc')
scales = importr('scales')
rlist = importr('rlist')
phytools = importr('phytools')
gen_graph = r['generate_phylogeny_graph']
offset = -1
graph_id = 0

# phyddle imports
from phyddle import utilities as util

# Uncomment to debug multiprocessing
# import multiprocessing.util as mp_util
# mp_util.log_to_stderr(mp_util.SUBDEBUG)

# Allows multiprocessing fork (not spawn) new processes on Unix-based OS.
# However, import phyddle.simulate also calls set_start_method('fork'), and
# the function throws RuntimeError when called the 2nd+ time within a single
# Python. We handle this with a try-except block.
try:
    set_start_method('fork')
except RuntimeError:
    pass

##################################################
# Packed-simulation helpers
##################################################

# subfolder of <packed_in_dir> : model_type (matches labels.csv convention)
# MuSSE folders are optional: they are only read if their model_type is in
# select_classes and the folder exists.
PACKED_MODELS = {
    'BD_div':  0,
    'BiSSE':   1,
    'MuSSE_3': 2,
    'MuSSE_4': 3,
    'MuSSE_5': 4,
}

# 'MuSSE_5', 'MuSSE5', 'musse_5' ... -> number of states
_MUSSE_RE = re.compile(r'^musse_?(\d+)$', re.IGNORECASE)


def _model_num_states(sub):
    """Number of character states for a packed model folder (None if unknown)."""
    if sub == 'BD_div':
        return 1
    if sub == 'BiSSE':
        return 2
    m = _MUSSE_RE.match(sub)
    return int(m.group(1)) if m else None


def _parse_max_model_states(v):
    """Config value of max_model_states -> int, or None if not set.
    None, '', 'none', 'all' or -1 mean 'not set' (use select_classes)."""
    if v is None or str(v).strip().lower() in ('', 'none', 'na', 'all', '-1'):
        return None
    k = int(v)
    if k < 1:
        raise ValueError(f'max_model_states must be 1 or more (got {v})')
    return k


def _parse_single_class(v):
    """Config value of single_class -> int model_type, or None if not set.
    None, '', 'none', 'na', 'F', 'false' or -1 mean 'not set'."""
    if v is None or str(v).strip().lower() in ('', 'none', 'na', 'f', 'false', '-1'):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f'single_class must be a model_type number, e.g. 1 for '
                         f'BiSSE (got {v})') from None

# default file names inside each model subfolder
PACKED_DEFAULT_FILES = {
    'packed_tre_fn_train': 'train.nwk',
    'packed_par_fn_train': 'train.txt',
    'packed_tre_fn_test':  'test.nwk',
    'packed_par_fn_test':  'test.txt',
}

##################################################
# Graph node encodings
##################################################

# Node features returned by get_node_df() for each encoding.
# Must match the 'keep' lists in convert-phylo-to-graph.R.
# encoding_type = 'all' formats every encoding listed here.
ENCODING_FEATURES = {
    'NC': ['dist_to_present', 'dist_to_root', 'dist_to_parent', 'leaves'],
    '1N': ['dist_to_present', 'dist_to_root', 'dist_to_parent', 'leaves',
           'dist_to_grandparent', 'l_dist', 'r_dist', 'l_leaves', 'r_leaves'],
    '2N': ['dist_to_present', 'dist_to_root', 'dist_to_parent', 'leaves',
           'dist_to_grandparent', 'l_dist', 'r_dist', 'l_leaves', 'r_leaves',
           'dist_to_great_grandparent', 'll_dist', 'lr_dist', 'rl_dist', 'rr_dist',
           'll_leaves', 'lr_leaves', 'rl_leaves', 'rr_leaves'],
    'default': ['dist_to_root', 'dist_to_parent', 'desc_ratio'],
}


def _encoding_features(enc):
    """Feature columns for one encoding (unknown names use R's fallback set)."""
    return ENCODING_FEATURES.get(enc, ENCODING_FEATURES['default'])

##################################################

# strings treated as a missing tree / value
_NA_STRINGS = ('', 'na', 'nan', 'null', 'none')

# strips [...] comments (e.g. [&&NHX-t_s=1]) from newick strings
_COMMENT_RE = re.compile(r'\[[^\]]*\]')
# a tip label = a name that directly follows '(' or ',' (internal names follow ')')
_TIP_RE = re.compile(r'(?<=[(,])([^,():;\[\]\s]+)((?:\[[^\]]*\])*)')


def _arg(args, key, default):
    """args[key], or default if the key is missing or None.
    An explicit empty string is kept (used to skip a train/test set)."""
    v = args.get(key, None) if hasattr(args, 'get') else None
    return default if v is None else v


def _log10(x, floor=1e-12):
    """log10 that tolerates zeros (e.g. mu = 0 in BiSSE)."""
    return float(np.log10(max(float(x), floor)))


def _tip_states_nhx(newick, key='t_s', offset=1, required=True, n_states=2):
    """Return {tip_label: state} read from NHX comments on the tips.

    State is (value of key) - offset, so t_s = 1..K becomes 0..K-1 by default.
    If required is False, tips without the annotation get state 0.
    """
    key_re = re.compile(rf'{re.escape(key)}=(\d+)')
    out = {}
    for m in _TIP_RE.finditer(newick):
        label, comment = m.group(1), m.group(2)
        k = key_re.search(comment)
        if k is None:
            if required:
                raise ValueError(f'Tip "{label}" has no {key} annotation')
            out[label] = 0
            continue
        s = int(k.group(1)) - offset
        if not 0 <= s < max(n_states, 1):
            raise ValueError(f'Tip "{label}": state {s} not in 0..{n_states - 1} '
                             f'(check packed_state_offset)')
        out[label] = s
    return out


def _extant_tip_labels(clean_newick, tol):
    """Labels of tips whose root-distance is within tol (relative) of the max."""
    tr = dp.Tree.get(data=clean_newick, schema='newick',
                     preserve_underscores=True,
                     suppress_internal_node_taxa=True)
    tr.calc_node_root_distances(return_leaf_distances_only=False)
    leaves = tr.leaf_nodes()
    dmax = max(nd.root_distance for nd in leaves)
    cut = dmax * (1.0 - max(tol, 1e-6))
    return [nd.taxon.label for nd in leaves if nd.root_distance >= cut]

##################################################

def load(args):
    """Load a Formatter object.

    This function creates an instance of the Formatter class, initialized
    using phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.
    
    Returns:
        Simulator: A new Simulator object.
    """
    # settings
    sys.setrecursionlimit(10000)

    # load object
    format_method = 'default'
    if format_method == 'default':
        return Formatter(args)
    else:
        return NotImplementedError

##################################################


class Formatter:
    """
    Class for formatting phylogenetic datasets and converting them into
    tensors to be used by the Train step.
    """
    def __init__(self, args):
        """Initializes a new Formatter object.

        Args:
            args (dict): Contains phyddle settings.
        """
        
        # filesystem
        self.sim_prefix         = str(args['sim_prefix'])
        self.emp_prefix         = str(args['emp_prefix'])
        self.fmt_prefix         = str(args['fmt_prefix'])
        self.sim_dir            = str(args['sim_dir'])
        self.emp_dir            = str(args['emp_dir'])
        self.fmt_dir            = str(args['fmt_dir'])
        self.log_dir            = str(args['log_dir'])
        
        # analysis settings
        self.verbose            = bool(args['verbose'])
        self.num_proc           = int(args['num_proc'])
        self.use_parallel       = bool(args['use_parallel'])
        self.no_sim             = bool(args['no_sim'])
        self.no_emp             = bool(args['no_emp'])
        
        # dataset dimensions
        self.num_char           = int(args['num_char'])
        self.num_states         = int(args['num_states'])
        self.num_trees          = int(args['num_trees'])
        self.min_num_taxa       = int(args['min_num_taxa'])
        self.max_num_taxa       = int(args['max_num_taxa'])
        self.tree_width         = int(args['tree_width'])
        
        # dataset processing
        self.encode_all_sim     = bool(args['encode_all_sim'])
        self.start_idx          = int(args['start_idx'])
        self.end_idx            = int(args['end_idx'])
        self.downsample_taxa    = str(args['downsample_taxa'])
        self.rel_extant_age_tol = float(args['rel_extant_age_tol'])
        self.tree_encode        = str(args['tree_encode'])
        self.char_encode        = str(args['char_encode'])
        self.brlen_encode       = str(args['brlen_encode'])
        self.char_format        = str(args['char_format'])
        self.tensor_format      = str(args['tensor_format'])
        self.param_est          = list(args['param_est'])
        self.param_data         = list(args['param_data'])
        self.prop_test          = float(args['prop_test'])
        self.log_offset         = float(args['log_offset'])
        self.save_phyenc_csv    = bool(args['save_phyenc_csv'])
        self.save_graph_csv = bool(args['save_graph_csv'])
        self.select_classes = list(args['select_classes'])

        # packed simulations. Layout:
        #   <packed_in_dir>/BD_div/{train,test}.nwk + {train,test}.txt
        #   <packed_in_dir>/BiSSE/ {train,test}.nwk + {train,test}.txt
        #   optional: <packed_in_dir>/MuSSE_3/, MuSSE_4/, MuSSE_5/ (same files)
        self.packed_in_dir        = str(args['packed_in_dir'])    # parent folder holding the model folders
        self.packed_par_dir       = str(_arg(args, 'packed_par_dir', ''))  # optional; only used if parameters are not next to the trees
        self.packed_out_dir       = str(args['packed_out_dir'])   # where unpacked out.N.* files go (e.g. 'simulate')
        # all known models, plus / overridden by packed_models from the config.
        # (Merged rather than replaced: a packed_models default of only BD_div
        # and BiSSE would otherwise silently drop the MuSSE models.)
        self.packed_models        = dict(PACKED_MODELS)
        self.packed_models.update(dict(_arg(args, 'packed_models', {}) or {}))
        self.packed_state_key     = str(_arg(args, 'packed_state_key', 't_s'))
        self.packed_state_offset  = int(_arg(args, 'packed_state_offset', 1))

        # tree / parameter file names inside each model subfolder
        # (set a pair to '' to skip that set)
        self.packed_tre_fn_train  = str(_arg(args, 'packed_tre_fn_train', PACKED_DEFAULT_FILES['packed_tre_fn_train']))
        self.packed_par_fn_train  = str(_arg(args, 'packed_par_fn_train', PACKED_DEFAULT_FILES['packed_par_fn_train']))
        self.packed_tre_fn_test   = str(_arg(args, 'packed_tre_fn_test',  PACKED_DEFAULT_FILES['packed_tre_fn_test']))
        self.packed_par_fn_test   = str(_arg(args, 'packed_par_fn_test',  PACKED_DEFAULT_FILES['packed_par_fn_test']))

        # fixed train/test split taken from the train/test files
        self.fixed_split          = bool(args['fixed_split'])
        self.packed_split         = {}                                # {'train': [NUM...], 'test': [NUM...]}

        # True = use the out.N.* files already in packed_out_dir and skip unpacking
        self.skip_unpack          = bool(_arg(args, 'skip_unpack', False))

        # True = always unpack again, even if the packed files look already unpacked
        self.force_unpack         = str(_arg(args, 'force_unpack', False)).strip().upper() in ('T', 'TRUE', '1', 'YES')

        # max_model_states = K: use every model with at most K character states
        # (BD = 1, BiSSE = 2, MuSSE_3..5 = 3..5) and ignore the rest, e.g.
        # K = 3 -> BD, BiSSE and MuSSE_3. This replaces select_classes, and
        # num_states is raised to K if it is smaller.
        self.max_model_states = _parse_max_model_states(_arg(args, 'max_model_states', None))
        if self.max_model_states is not None:
            k = self.max_model_states
            self.select_classes = sorted(
                mt for sub, mt in self.packed_models.items()
                if _model_num_states(sub) is not None and _model_num_states(sub) <= k)
            used = [sub for sub, mt in self.packed_models.items() if mt in self.select_classes]
            util.print_str(f'▪ max_model_states = {k}: using {", ".join(used)} '
                           f'(model_type {self.select_classes})', self.verbose)
            if self.num_states < k:
                util.print_str(f'▪ num_states raised from {self.num_states} to {k}',
                               self.verbose)
                self.num_states = k
            elif self.num_states > k:
                util.print_warn(f'num_states = {self.num_states} is larger than '
                                f'max_model_states = {k}; states {k}..{self.num_states - 1} '
                                f'will always be empty')

        # single_class = <model_type>: unpack and format one model class only,
        # e.g. single_class = 1 for a test set that holds only BiSSE trees.
        # Only the folder of that class is required; the other selected
        # classes may be missing. select_classes is left unchanged, so the
        # model_type labels mean the same as in a full run.
        # Not set (default) = every selected model_type must have a folder.
        self.single_class = _parse_single_class(_arg(args, 'single_class', None))
        if self.single_class is not None:
            if self.single_class not in self.select_classes:
                raise ValueError(f'single_class = {self.single_class} is not among the '
                                 f'selected model_types {self.select_classes}')
            names = [sub for sub, mt in self.packed_models.items() if mt == self.single_class]
            util.print_str(f'▪ single_class = {self.single_class}: converting '
                           f'{", ".join(names)} only (model_type {self.single_class})',
                           self.verbose)

        # graph node encoding(s)
        # 'all' formats every encoding in one pass (same trees, same split)
        self.encoding_type = str(args['encoding_type'])
        if self.encoding_type == 'all':
            self.encodings = list(ENCODING_FEATURES)
        else:
            self.encodings = [self.encoding_type]
        # number of node features per encoding: {encoding: num_attr}
        self.num_attr = {e: len(_encoding_features(e)) for e in self.encodings}

        # features of interest (one f_dat_i / n_dat_i pair per character state,
        # in the same order make_summ_stat() creates them)
        self.aux_data = (["log10_tree_length", "log10_root_age", "log10_brlen_mean",
                          "log10_age_mean", "log10_B1", "colless", "age_var",
                          "brlen_var", "treeness", "N_bar"] +
                         [f'{p}_dat_{i}' for i in range(self.num_states) for p in ('f', 'n')] +
                         ["num_taxa", "prop_taxa"])
        self.block_size = int(args['block_size'])

        # set number of processors
        if self.num_proc <= 0:
            self.num_proc = cpu_count() + self.num_proc

        # are input characters, trees, or labels used?
        self.use_input_dat = (self.num_char > 0)
        self.use_input_tree = (self.num_trees > 0)
        self.use_input_lbl = (len(self.param_data) > 0)
        assert(self.use_input_dat or self.use_input_tree or self.use_input_lbl)

        # get size of CPV+S tensors
        num_tree_col = util.get_num_tree_col(self.tree_encode,
                                             self.brlen_encode)
        num_char_col = util.get_num_char_col(self.char_encode,
                                             self.num_char,
                                             self.num_states)
        self.num_data_col = num_tree_col + num_char_col
        
        # create logger to track runtime info
        self.logger = util.Logger(args)

        # initialized later
        self.rep_data           = dict()   # init with encode_all()
        self.split_idx          = dict()   # init with split_examples()
        self.rep_idx            = list()   # init with get_rep_idx()

        # done
        return

    ##########################################################
    # HDF5 file naming / versioning
    ##########################################################

    def _find_hdf5_fn(self, data_str, enc=None):
        """Path of an existing HDF5 file for one split and encoding.

        Tries, in order:
          <fmt_prefix>.<data_str>.<encoding>.states<K>.hdf5   (K = max_model_states)
          the same with a larger K (K+1..5; a superset of the models)
          <fmt_prefix>.<data_str>.<encoding>.hdf5             (no states tag)
          <fmt_prefix>.<data_str>.hdf5                         (no encoding)
        Without max_model_states only the last two are tried.
        Raises FileNotFoundError if none exists.
        """
        tried = [self._hdf5_fn(data_str, enc)]
        if self.max_model_states is not None:
            tried += [self._hdf5_fn(data_str, enc, k)
                      for k in range(self.max_model_states + 1, 6)]
            tried.append(self._hdf5_fn(data_str, enc, states=False))
        tried.append(f'{self.fmt_dir}/{self.fmt_prefix}.{data_str}.hdf5')
        tried = list(dict.fromkeys(tried))

        for i, fn in enumerate(tried):
            if os.path.exists(fn):
                if i > 0:
                    util.print_str(f'▪ {os.path.basename(tried[0])} not found; using '
                                   f'{os.path.basename(fn)}', self.verbose)
                return fn

        raise FileNotFoundError(f'No HDF5 file for {data_str}. Tried: '
                                + ', '.join(tried))

    def _hdf5_fn(self, data_str, enc=None, states=None):
        """Path of the HDF5 file for one split and encoding,
        e.g. <fmt_dir>/<fmt_prefix>.train.2N.hdf5, or with max_model_states = 3
        <fmt_dir>/<fmt_prefix>.train.2N.states3.hdf5.

        states: None = use max_model_states (no tag if it is not set),
                False = no tag, an int = that tag.
        """
        enc = self.encodings[0] if enc is None else enc
        enc = re.sub(r'[^A-Za-z0-9_\-]+', '_', enc)  # filename-safe
        if states is None:
            states = self.max_model_states
        tag = f'.states{int(states)}' if states not in (None, False) else ''
        return f'{self.fmt_dir}/{self.fmt_prefix}.{data_str}.{enc}{tag}.hdf5'

    def _archive_existing_hdf5(self, fn):
        """If fn exists, rename it to the next free version,
        e.g. out.train.2N.hdf5 -> out.train.2N.v3.hdf5 when v1 and v2
        already exist. Returns the archived path, or None if nothing was moved."""
        if not os.path.exists(fn):
            return None
        out_dir = os.path.dirname(fn) or '.'
        base = os.path.basename(fn)[:-len('.hdf5')]
        ver_re = re.compile(rf'^{re.escape(base)}\.v(\d+)\.hdf5$')
        versions = []
        for f in os.listdir(out_dir):
            m = ver_re.match(f)
            if m:
                versions.append(int(m.group(1)))
        new_v = max(versions, default=0) + 1
        archived = os.path.join(out_dir, f'{base}.v{new_v}.hdf5')
        os.rename(fn, archived)
        util.print_str(f'▪ Existing {os.path.basename(fn)} moved to '
                       f'{os.path.basename(archived)}', self.verbose)
        return archived

    ##########################################################
    # Packed-simulation unpacking
    ##########################################################

    def _find_packed_files(self, sub, tre_fn, par_fn):
        """Return (trees_path, params_path) for one packed model (e.g. 'BD_div').

        Trees:      <packed_in_dir>/<sub>/<tre_fn>           (e.g. BiSSE/train.nwk)
        Parameters: <packed_in_dir>/<sub>/<par_fn>           (e.g. BiSSE/train.txt)
                    else <packed_par_dir>/<sub>/<par_fn>
                    else <packed_par_dir>/<par_fn>
        The folder holding the trees is always searched first, so
        packed_par_dir only matters if parameters live somewhere else.
        """
        tre_path = os.path.join(self.packed_in_dir, sub, tre_fn)
        if not os.path.exists(tre_path):
            raise FileNotFoundError(f'Cannot find trees file for {sub}: {tre_path}')

        # first look next to the trees (e.g. BD_div/train.txt beside BD_div/train.nwk),
        # then in packed_par_dir if one is set
        tried = [os.path.join(os.path.dirname(tre_path), par_fn)]
        if self.packed_par_dir:
            tried += [os.path.join(self.packed_par_dir, sub, par_fn),
                      os.path.join(self.packed_par_dir, par_fn)]
        tried = list(dict.fromkeys(tried))   # drop duplicates, keep order
        par_path = next((p for p in tried if os.path.exists(p)), None)
        if par_path is None:
            raise FileNotFoundError(f'Cannot find parameters file for {sub}. Tried: '
                                    + ', '.join(tried))
        return tre_path, par_path

    def _labels_bd(self, row):
        # BD columns: index, turnover_rate, birth_rate, extinction_rate,
        #             sampling_frac, tree_size
        return {'log10_birth_rate': _log10(row['birth_rate']),
                'log10_death_rate': _log10(row['extinction_rate'])}

    def _labels_bisse(self, row):
        # BiSSE columns: index, lambda1, lambda2, turnover, sampling_frac,
        #   tree_size, mu1, mu2, net_rate1, net_rate2, q01, q10,
        #   lambda2_ratio, q01_ratio
        # q01 always equals q10 in these simulations, so the shared rate is
        # also written as log10_trans_rate / trans_rate (same names as MuSSE)
        q01, q10 = float(row['q01']), float(row['q10'])
        if not np.isclose(q01, q10, rtol=1e-6, atol=0.0):
            raise ValueError(f'BiSSE row index {row.get("index", "?")}: q01 = {q01} '
                             f'and q10 = {q10} differ, but a single shared '
                             f'transition rate is expected')
        return {'log10_birth_rate_t1': _log10(row['lambda1']),
                'log10_birth_rate_t2': _log10(row['lambda2']),
                'log10_death_rate_t1': _log10(row['mu1']),
                'log10_death_rate_t2': _log10(row['mu2']),
                'log10_trans_rate':    _log10(q01),
                'trans_rate':          q01,
                'log10_trans_01':      _log10(q01),          # kept for older configs
                'log10_trans_10':      _log10(q10),
                'trans_10':            q10}                  # raw, as in old header

    def _labels_musse(self, row, n_states):
        # MuSSE_K columns (K = 3, 4 or 5):
        #   index, n_states, lambda1..lambdaK, mu1..muK, net_rate1..K,
        #   turnover1..K, q, sampling_frac, tree_size, lambda_ratio2..K,
        #   q_ratio_lambda1, q_ratio_mean
        # q is one transition rate shared by all state pairs.
        # Labels:
        #   lambdaI -> log10_birth_rate_tI, muI -> log10_death_rate_tI,
        #   q -> log10_trans_rate (and raw trans_rate)
        # Per-pair columns (q12, q1_2, ...) are also read if a file has them,
        # giving log10_trans_12 etc.
        if 'n_states' in row.index and str(row['n_states']).strip().lower() not in _NA_STRINGS:
            n_file = int(float(row['n_states']))
            if n_file != n_states:
                raise ValueError(f'Parameter row index {row.get("index", "?")} has '
                                 f'n_states = {n_file} but the folder is MuSSE_{n_states}')

        lbl = {}
        for i in range(1, n_states + 1):
            lbl[f'log10_birth_rate_t{i}'] = _log10(row[f'lambda{i}'])   # KeyError if missing
        for i in range(1, n_states + 1):
            lbl[f'log10_death_rate_t{i}'] = _log10(row[f'mu{i}'])

        if 'q' in row.index:
            lbl['log10_trans_rate'] = _log10(row['q'])
            lbl['trans_rate'] = float(row['q'])                         # raw, like BiSSE trans_10

        for name in row.index:
            m = re.match(r'^q[._]?(\d)[._]?(\d)$', name)
            if m and str(row[name]).strip().lower() not in _NA_STRINGS:
                lbl[f'log10_trans_{m.group(1)}{m.group(2)}'] = _log10(row[name])

        if 'log10_trans_rate' not in lbl and not any(k.startswith('log10_trans_') for k in lbl):
            raise KeyError('q')
        return lbl

    def _label_maker(self, sub):
        """Label function for a packed model folder, or None if unknown."""
        if sub == 'BD_div':
            return self._labels_bd
        if sub == 'BiSSE':
            return self._labels_bisse
        if _MUSSE_RE.match(sub):
            k = _model_num_states(sub)
            return lambda row: self._labels_musse(row, k)
        return None

    def _write_dat_csv(self, path, tree_str, model_type, n_states=2):
        """Write 'taxa,data' csv of extant-tip states for one tree.

        data = (NHX t_s value) - packed_state_offset, so t_s 1..K -> 0..K-1.
        Only extant tips are written. BD trees without t_s get data = 0.
        """
        clean = _COMMENT_RE.sub('', tree_str)
        extant = _extant_tip_labels(clean, self.rel_extant_age_tol)
        states = _tip_states_nhx(tree_str,
                                 key=self.packed_state_key,
                                 offset=self.packed_state_offset,
                                 required=(model_type != 0),
                                 n_states=n_states)
        with open(path, 'w') as f:
            f.write('taxa,data\n')
            for label in extant:
                f.write(f'{label},{states[label]}\n')

    def _read_tree_table(self, tre_path):
        """Read a .nwk trees file into a list of (index or None, newick or None).

        Accepts either
          - one bare newick tree per line, or
          - '<index><TAB or space><newick>' per line, optionally with a
            header row (e.g. 'index<TAB>tree').
        Missing trees (NA, empty) are kept as None so row positions are
        preserved for position-based matching.
        """
        rows = []
        first = True
        with open(tre_path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                if ln.startswith('('):                       # bare newick
                    key, tree = None, ln
                else:
                    parts = ln.split(None, 1)
                    tok = parts[0].strip('"\'')
                    rest = parts[1].strip().strip('"\'') if len(parts) > 1 else ''
                    num = pd.to_numeric(tok, errors='coerce')
                    if pd.isna(num):
                        if first and tok.lower() not in _NA_STRINGS:
                            first = False                    # header row
                            continue
                        key, tree = None, ln                 # e.g. a bare 'NA' line
                    else:
                        key, tree = float(num), rest
                first = False
                if tree.strip('"\'').lower() in _NA_STRINGS:
                    tree = None
                rows.append((key, tree))
        return rows

    def _read_param_table(self, par_path):
        """Read a parameters .txt/.csv file (tab, comma, semicolon or
        whitespace separated, delimiter detected from the header line).

        Column names are stripped, unquoted and lower-cased. R-style files
        with row names (header one field shorter than the data) are handled.
        """
        with open(par_path) as f:
            header = next((ln for ln in f if ln.strip()), '')
        if '\t' in header:
            sep = '\t'
        elif ',' in header:
            sep = ','
        elif ';' in header:
            sep = ';'
        else:
            sep = r'\s+'
        params = pd.read_csv(par_path, sep=sep, dtype=str, keep_default_na=False,
                             engine='python', skipinitialspace=True)
        params = params.reset_index(drop=True)   # drops R row names if present
        params.columns = [str(c).strip().strip('"\'').lower() for c in params.columns]
        params = params.apply(lambda s: s.astype(str).str.strip().str.strip('"\''))
        return params

    def _load_split_map(self, map_fn):
        """Reload the fixed train/test NUM lists written by unpack_packed_sims."""
        m = pd.read_csv(map_fn)
        self.packed_split = {s: [int(x) for x in m.loc[m['split'] == s, 'num']]
                             for s in ['train', 'test']}

    def _use_existing_unpacked(self):
        """skip_unpack = True: reuse out.N.* files already in packed_out_dir.

        Does not read the .nwk/.txt files. The fixed train/test split is
        reloaded from <packed_out_dir>/split_index_map.csv.
        """
        out_dir = self.packed_out_dir
        map_fn = os.path.join(out_dir, 'split_index_map.csv')
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(f'skip_unpack is True but {out_dir} does not exist. '
                                    f'Run once with skip_unpack = False to unpack.')
        if os.path.exists(map_fn):
            self._load_split_map(map_fn)
        elif self.fixed_split:
            raise RuntimeError(f'skip_unpack and fixed_split are True but {map_fn} '
                               f'is missing, so the train/test split is unknown. '
                               f'Run once with skip_unpack = False.')
        self.sim_dir = out_dir   # rest of Format reads from here
        util.print_str(f'▪ Skipping unpacking; using existing files in {out_dir}',
                       self.verbose)
        return True

    def unpack_packed_sims(self):
        """Split packed tree/parameter files into per-replicate files.

        Expected layout:
            <packed_in_dir>/BD_div/train.nwk, train.txt, test.nwk, test.txt
            <packed_in_dir>/BiSSE/ train.nwk, train.txt, test.nwk, test.txt
        and optionally
            <packed_in_dir>/MuSSE_3/, MuSSE_4/, MuSSE_5/ (same file names)

        A model folder is used only if its model_type is in select_classes.

        Trees (.nwk) and parameters (.txt) are matched by their 'index'
        column when both files have one; otherwise they are matched by row
        position. Rows with a missing / NA tree are skipped.

        Writes <packed_out_dir>/<sim_prefix>.N.tre, .N.labels.csv and .N.dat.csv
        with a global counter N across models and splits.

        If fixed_split is True the train/test split is exactly the train/test
        files; if False they are pooled and split_examples() splits randomly.
        The N values for each split are saved in
        <packed_out_dir>/split_index_map.csv (num, model, split, orig_index).

        If skip_unpack is True, nothing is unpacked and the existing files in
        packed_out_dir are used as they are.
        Returns True if packed data was used.
        """
        # reuse previously unpacked files without touching the packed inputs
        if self.skip_unpack:
            return self._use_existing_unpacked()

        root = self.packed_in_dir
        if not os.path.isdir(root):
            # no packed data: fall back to ordinary out.N.* files in sim_dir
            util.print_warn(f'Packed tree folder not found: {root}. '
                            f'Using existing files in {self.sim_dir} instead.')
            return False

        out_dir = self.packed_out_dir
        os.makedirs(out_dir, exist_ok=True)
        map_fn = os.path.join(out_dir, 'split_index_map.csv')
        sentinel = os.path.join(out_dir, '.unpacked')

        # label function per model name (None = unknown model)
        makers = {sub: self._label_maker(sub) for sub in self.packed_models}

        # model folders present, matched ignoring case, '_' and '-'
        # (MuSSE_3 matches MuSSE3, musse_3, MUSSE-3, ...)
        def norm_name(x):
            return re.sub(r'[^a-z0-9]', '', x.lower())
        present = {norm_name(f): f for f in os.listdir(root)
                   if os.path.isdir(os.path.join(root, f))}

        # one folder per selected model_type; a selected model without a
        # folder is an error, so a K-state run never silently drops a model.
        # With single_class set, only that model_type is required and used.
        wanted = ([self.single_class] if self.single_class is not None
                  else list(self.select_classes))
        chosen = {}                 # model_type -> (model name, folder)
        for sub, model_type in self.packed_models.items():
            if model_type not in wanted or model_type in chosen:
                continue
            if makers.get(sub) is None:
                continue
            folder = present.get(norm_name(sub))
            if folder is not None:
                chosen[model_type] = (sub, folder)
        missing = [mt for mt in wanted if mt not in chosen]
        if missing:
            names = {mt: [s for s, m in self.packed_models.items() if m == mt] for mt in missing}
            hint = ('' if self.single_class is not None or len(chosen) == 0 else
                    f'. To convert only the classes that are present, set '
                    f'single_class (e.g. single_class = {min(chosen)})')
            raise FileNotFoundError(
                f'No packed folder in {root} for selected model_type(s) '
                + ', '.join(f'{mt} (expected one of {names[mt]})' for mt in missing)
                + f'. Folders present: {sorted(present.values())}' + hint)
        util.print_str('▪ Model folders: ' + ', '.join(
            f'{sub} = {folder}/ (model_type {mt})' for mt, (sub, folder) in sorted(chosen.items())),
            self.verbose)

        # list of (model name, model_type, split name, trees path, params path)
        jobs = []
        for model_type, (sub, folder) in sorted(chosen.items()):
            n_st = _model_num_states(sub)
            if n_st is not None and n_st > self.num_states:
                raise ValueError(f'{sub} has {n_st} character states but num_states '
                                 f'= {self.num_states} in the config. Set num_states '
                                 f'to at least {n_st}.')
            for split, tfn, pfn in [('train', self.packed_tre_fn_train, self.packed_par_fn_train),
                                    ('test',  self.packed_tre_fn_test,  self.packed_par_fn_test)]:
                if not tfn or not pfn:      # this set not provided
                    continue
                tre_path, par_path = self._find_packed_files(folder, tfn, pfn)
                jobs.append((sub, model_type, split, tre_path, par_path))

        if not jobs:
            util.print_warn(f'No packed tree/parameter files found under {root}. '
                            f'Using existing files in {self.sim_dir} instead.')
            return False

        # skip if already unpacked from the same inputs (files, timestamps,
        # class selection and split mode)
        sel_key = '|'.join(
            ['classes=' + ','.join(str(c) for c in sorted(self.select_classes)),
             f'single={self.single_class}',
             f'fixed={int(self.fixed_split)}'] +
            [f'{s}:{sp}:{os.path.abspath(t)}:{os.path.getmtime(t):.0f}:'
             f'{os.path.abspath(p)}:{os.path.getmtime(p):.0f}'
             for s, _, sp, t, p in jobs])
        if self.force_unpack:
            util.print_str('▪ force_unpack is set; unpacking again', self.verbose)
        elif os.path.exists(sentinel) and os.path.exists(map_fn):
            with open(sentinel) as f:
                same_inputs = f.read().strip() == sel_key
            if same_inputs:
                # the inputs match; also check every unpacked tree is still there
                m = pd.read_csv(map_fn)
                missing = [int(n) for n in m['num']
                           if not all(os.path.exists(f'{out_dir}/{self.sim_prefix}.{int(n)}.{ext}')
                                      for ext in ('tre', 'labels.csv', 'dat.csv'))]
                if not missing:
                    self._load_split_map(map_fn)
                    self.sim_dir = out_dir   # rest of Format reads from here
                    counts = m.groupby(['model', 'split']).size()
                    util.print_str('▪ Packed simulations already unpacked; reusing '
                                   f'{len(m)} trees in {out_dir}: ' +
                                   ', '.join(f'{mod} {sp} {c}' for (mod, sp), c in counts.items()),
                                   self.verbose)
                    return True
                util.print_warn(f'{len(missing)} of {len(m)} unpacked trees are missing from '
                                f'{out_dir}; unpacking again')

        # remove out.N.* files left by a previous unpack (only if that unpack
        # wrote a split map here, so user files are never touched)
        if os.path.exists(map_fn):
            stale = re.compile(rf'^{re.escape(self.sim_prefix)}\.\d+\.')
            for fn in os.listdir(out_dir):
                if stale.match(fn):
                    os.remove(os.path.join(out_dir, fn))
            os.remove(map_fn)
        if os.path.exists(sentinel):
            os.remove(sentinel)

        n = 0
        split_nums = {'train': [], 'test': []}
        map_rows = []
        for sub, model_type, split, tre_path, par_path in jobs:
            tree_rows = self._read_tree_table(tre_path)
            params = self._read_param_table(par_path)
            label = f'{sub} ({split})'
            n_states_sub = _model_num_states(sub) or self.num_states

            has_idx_col = 'index' in params.columns
            keyed = len(tree_rows) > 0 and all(k is not None for k, _ in tree_rows)

            if keyed and has_idx_col:
                # match by index
                trees = {k: t for k, t in tree_rows if t}
                p_key = pd.to_numeric(params['index'], errors='coerce').tolist()
            else:
                # match by row position
                if len(tree_rows) != len(params):
                    util.print_warn(f'{label}: {len(tree_rows)} trees in {tre_path} but '
                                    f'{len(params)} parameter rows in {par_path}; '
                                    f'matching by row position')
                trees = {float(i): t for i, (_, t) in enumerate(tree_rows) if t}
                p_key = [float(i) for i in range(len(params))]

            n_skip = 0
            n_done = 0
            for i, (pk, (_, row)) in enumerate(tqdm(list(zip(p_key, params.iterrows())),
                                                     desc=f'Unpacking {label}')):
                tree_str = trees.get(float(pk)) if pd.notna(pk) else None
                if tree_str is None:        # missing or NA tree
                    n_skip += 1
                    continue
                if not tree_str.endswith(';'):
                    tree_str += ';'
                try:
                    lbl = makers[sub](row)
                except KeyError as e:
                    raise ValueError(f'Column {e} not found in {par_path}. '
                                     f'Columns are: {list(params.columns)}') from None
                lbl['model_type'] = model_type

                base = f'{out_dir}/{self.sim_prefix}.{n}'
                with open(base + '.tre', 'w') as f:
                    f.write(_COMMENT_RE.sub('', tree_str) + '\n')
                pd.DataFrame([lbl]).to_csv(base + '.labels.csv', index=False)
                self._write_dat_csv(base + '.dat.csv', tree_str, model_type,
                                    n_states=n_states_sub)

                orig_index = row['index'] if has_idx_col else i + 1
                split_nums[split].append(n)
                map_rows.append((n, sub, split, orig_index))
                n += 1
                n_done += 1
            util.print_str(f'▪ Unpacked {label}: {n_done} trees, '
                           f'{n_skip} skipped (NA or missing tree)', self.verbose)

        pd.DataFrame(map_rows, columns=['num', 'model', 'split', 'orig_index']
                     ).to_csv(map_fn, index=False)
        with open(sentinel, 'w') as f:
            f.write(sel_key)
        self.packed_split = split_nums
        if n > 0:
            self.sim_dir = out_dir   # rest of Format reads from here
        return n > 0

    ##########################################################

    def run(self):
        """Formats all raw datasets into tensors.

        This method prints status updates, creates the target directory for new
        tensors, then formats each simulated raw dataset into an individual
        tensor-formatted dataset, then combines all individual tensors into a
        single tensor that contains all training examples.
        """
        
        verbose = self.verbose

        # convert packed model-folder train/test files into per-replicate files
        # (or reuse existing ones if skip_unpack is True)
        if not self.no_sim:
            self.unpack_packed_sims()

        valid_emp = self.has_valid_dataset(mode='emp')
        valid_sim = self.has_valid_dataset(mode='sim')
        input_print_dir = list()
        input_prefix = list()
        if valid_emp and not self.no_emp:
            input_print_dir.append(self.emp_dir)
            input_prefix.append(self.emp_prefix)
        if valid_sim and not self.no_sim:
            input_print_dir.append(self.sim_dir)
            input_prefix.append(self.sim_prefix)
            
        # print header
        util.print_step_header('fmt', input_print_dir, self.fmt_dir,
                               input_prefix, self.fmt_prefix, verbose)
        # prepare workspace
        os.makedirs(self.fmt_dir, exist_ok=True)

        # start time
        start_time, start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        found_sim = False
        if self.no_sim:
            util.print_str('▪ Skipping simulated data', verbose)
        elif self.has_valid_dataset(mode='sim'):
            # run() attempts to generate one simulation per value in rep_idx,
            # where rep_idx is list of unique ints to identify simulated datasets
            util.print_str('▪ Collecting simulated data', verbose)
            self.rep_idx = self.get_rep_idx(mode='sim')
            util.print_str('▪ Splitting into train and test datasets', verbose)
            self.split_idx = self.split_examples(self.rep_idx)
            if self.tensor_format == 'hdf5':
                self.create_hdf5('test')
                self.create_hdf5('train')
            
            strings = ["train", "test"]
            print("encoding: ", self.encoding_type, "->", ", ".join(self.encodings))

            for string in strings:
               
                rep_idx = self.split_idx[string]
                rep_pos = 0
                addition = self.block_size
                if (addition > len(rep_idx)):
                    addition = len(rep_idx)
                while (rep_pos < len(rep_idx)):
                    util.print_str('▪ Encoding simulated data as tensors', verbose, ' at pos', rep_pos)
                    self.encode_all(rep_idx[(rep_pos):(rep_pos+addition)], mode='sim')
                    util.print_str('▪ Combining and writing simulated data as tensors', verbose, ' at pos ', rep_pos)      
                    # split examples into training and test datasets
                    self.write_tensor(rep_pos, rep_pos+addition, mode=string)
                    rep_pos = rep_pos + addition
                    self.rep_data = None

    


            # clear out object data (improves multiprocessing.Pool speed)
            self.rep_data = None
            
            # working on PCA pre-check
            # self.make_pca()
            
            # done
            found_sim = True

        found_emp = False
        if self.no_emp:
            util.print_str('▪ Skipping empirical data', verbose)
        elif self.has_valid_dataset(mode='emp'):
            # collecting empirical files
            util.print_str('▪ Collecting empirical data', verbose)
            self.rep_idx = self.get_rep_idx(mode='emp')

            self.split_idx = { 'empirical' : np.array(self.rep_idx) }
            if self.tensor_format == 'hdf5':
                self.create_hdf5('empirical')
            
            rep_pos = 0
            # We process the data in blocks to avoid using too much memory.
            addition = self.block_size
            if (addition + rep_pos > len(self.rep_idx)):
                addition = len(self.rep_idx) - rep_pos
            while (rep_pos < len(self.rep_idx)):
                util.print_str('▪ Encoding empirical data as tensors', verbose, ' at pos', rep_pos)
                self.encode_all(self.rep_idx[(rep_pos):(rep_pos+addition)], mode='emp')
                util.print_str('▪ Combining and writing empirical data as tensors', verbose, ' at pos ', rep_pos)
                self.write_tensor(min_index=rep_pos, max_index=rep_pos+addition, mode='emp')
                rep_pos = rep_pos + addition
                self.rep_data = None

            
            # clear out object data (improves multiprocessing.Pool speed)
            self.rep_data = None
            
            # done
            found_emp = True

        # notify user if no work done
        if self.no_emp and self.no_sim:
            util.print_warn('Format has no work to do when no_sim '
                               'and no_emp are used together.')
        elif not found_sim and not found_emp:
            util.print_warn('No simulated or empirical datasets found. '
                            'Check config settings.', verbose)

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        # done
        util.print_str('... done!', verbose)
    
    def has_valid_dataset(self, mode='sim'):
        """Determines if empirical analysis is being performed.
        
        Args:
            mode (str): 'sim' or 'emp' for simulated or empirical analysis.
        
        Returns:
            bool: True if empirical analysis is being performed.
        """
        
        assert mode in ['sim', 'emp']
        dat_dir = ''
        dat_prefix = ''
        if mode == 'sim':
            dat_dir = self.sim_dir
            dat_prefix = self.sim_prefix
        elif mode == 'emp':
            dat_dir = self.emp_dir
            dat_prefix = self.emp_prefix
            
        # check if empirical directory exists
        if not os.path.exists(dat_dir):
            return False
        
        # check if empirical directory contains files
        files = set()
        for f in os.listdir(dat_dir):
            f = '.'.join( f.split('.')[0:2])
            if dat_prefix in f:
                files.add(f)
        
        if len(files) == 0:
            return False
        
        # check that at least one dataset is complete
        for f in files:
            has_dat = os.path.exists(f'{dat_dir}/{f}.dat.csv') or \
                      os.path.exists(f'{dat_dir}/{f}.dat.nex')
            has_tre = os.path.exists(f'{dat_dir}/{f}.tre')
            has_lbl = os.path.exists(f'{dat_dir}/{f}.labels.csv')
            
            # no labels needed for empirical datasets with no param_data
            if mode == 'emp' and len(self.param_data) == 0:
                has_lbl = True
            elif mode == 'emp' and len(self.param_data) > 0 and not has_lbl:
                util.print_warn(f'Cannot find \'{dat_dir}/{f}.labels.csv\' but len(param_data) > 0.')
                return False
            # if the 1-3 files exist, then we have 1+ valid datasets
            if (has_dat or not self.use_input_dat) and (has_tre or not self.use_input_tree) and has_lbl:
                return True
    
        return False

    def encode_all(self, selected_idx, mode='sim'):
        """Encode all simulated replicates.
        
        Encode each simulated dataset identified by the replicate-index list
        (self.rep_idx). Each dataset is encoded by calling self.encode_one(idx)
        where idx is a unique value in self.rep_idx.

        Encoded simulations are then sorted by tree-width category and stored
        into the phy_tensors dictionary for later processing. Names and lengths
        of label and summary statistic lists are also saved.

        When self.use_parallel is True then all jobs are run in parallel via
        multiprocessing.Pool. When self.use_parallel is false, jobs are run
        serially with one CPU.

        Args:
            mode (str): specifies 'sim' or 'emp' dataset
        """

        # construct list of encoding arguments
        assert mode in ['sim', 'emp']
        encode_path = ''
        if mode == 'sim':
            encode_path = f'{self.sim_dir}/{self.sim_prefix}'
        elif mode == 'emp':
            encode_path = f'{self.emp_dir}/{self.emp_prefix}'
        
        args = []
        for idx in selected_idx:
            args.append((f'{encode_path}.{idx}', idx, mode))
        # visit each replicate, encode it, and return result
        if self.use_parallel:
            # parallel jobs
            with Pool(processes=self.num_proc) as pool:
                # Note, it's critical to call this as list(tqdm(pool.imap(...)))
                # - pool.imap runs the parallelization
                # - tqdm generates progress bar
                # - list acts as a finalizer for the pool.imap work
                # Have not benchmarked imap/imap_unordered, chunksize, etc.
                res = list(tqdm(pool.imap(self.encode_one_star,
                                          args, chunksize=5),
                                total=len(args),
                                desc='Encoding',
                                smoothing=0))
                
        else:
            # serial jobs
            res = [ self.encode_one_star(a) for a in tqdm(args,
                                                          total=len(args),
                                                          desc='Encoding',
                                                          smoothing=0) ]

        num_total = len(res)
        num_valid = len([x for x in res if x is not None])
        num_invalid = len([x for x in res if x is None])
        util.print_str(f'Encoding found {num_invalid} of {num_total} invalid examples.')

        util.print_str(f'Encoding found {num_valid} of {num_total} valid examples.')
        if num_valid == 0:
            # exits
            util.print_err('Format cannot proceed without valid examples.\n'
                           '       Verify the simulation script generates \n'
                           '       valid examples that are compatible with \n'
                           '       the configuration.\n')
            sys.exit()
            
        # save all replicate output by index
        self.rep_data = {}
        for i in res:
            if i is not None:
                # Get graph information, such as node attributes and number of edges and nodes, with the phylogeny and label.
                # node_attr is a dict {encoding: DataFrame of node features}
                self.rep_data[i[0]] = { 'phy':i[1].flatten(),
                                        'node_1':i[3],
                                        'node_2':i[4],
                                        'node_attr':i[2],
                                        'num_edges':i[5],
                                        'num_nodes':i[6],
                                        'aux': i[7],
                                        'lbl': i[8]}
                # Check for a bug in obtaining individual graphs from all results.
                n_rows = {e: a.shape[0] for e, a in i[2].items()}
                if any(n != i[6] for n in n_rows.values()):
                    print(i[6])
                    print(n_rows)
                    print("MISSMATCH!")
                    quit()
        return 
    
    def get_rep_idx(self, mode='sim'):
        """Determines replicate indices to use.

        This function finds all replicate indices within sim_proj_dir.

        Returns:
            int[]: List of replicate indices.
        """

        assert mode in ['sim', 'emp']
        
        # procedure assumes the simulate directory only contains the target set
        # of files which is an unsafe assumption, in general
        # find all rep index
        all_idx = set()
        files = []
        if mode == 'sim':
            files = os.listdir(f'{self.sim_dir}')
        elif mode == 'emp':
            files = os.listdir(f'{self.emp_dir}')
        
        if self.use_input_dat:
            files = [ f for f in files if '.dat.' in f ]
        elif self.use_input_tree:
            files = [ f for f in files if '.tre' in f ]

        for f in files:
            s_idx = f.split('.')[1]
            try:
                all_idx.add(int(s_idx))
            except ValueError:
                print(f'Skipping invalid filename {f}.')
                pass
            
        all_idx = sorted(list(all_idx))
        return all_idx

    def split_examples(self, rep_idx):
        """Split examples into training and test datasets."""
        
        split_idx = {}
        rep_idx = np.array(rep_idx)

        # fixed split: use the train/test indices from the train/test files
        if self.fixed_split:
            if not self.packed_split:
                raise RuntimeError('fixed_split is True but no packed train/test '
                                   'data was found. Check packed_in_dir and the '
                                   'packed_*_fn_train / packed_*_fn_test file names.')
            avail = set(int(i) for i in rep_idx)
            split_idx['train'] = np.array([i for i in self.packed_split['train'] if i in avail])
            split_idx['test'] = np.array([i for i in self.packed_split['test'] if i in avail])
            util.print_str(f"▪ Fixed split: {len(split_idx['train'])} train, "
                           f"{len(split_idx['test'])} test", self.verbose)
            return split_idx
        num_samples = len(rep_idx)
        
        # shuffle examples
        np.random.shuffle(rep_idx)
        
        # split examples
        num_test = int(num_samples * self.prop_test)
        split_idx['test'] = rep_idx[:num_test]
        split_idx['train'] = rep_idx[num_test:]
            
        # return split indices
        return split_idx

    def write_tensor(self, min_index=0, max_index=10000, mode='train'):
        """Write tensors to file.
        
        This function writes the train and test tensors as files in 
        hdf5 format based on the tensor_format setting. Actual writing
        is delegated to the write_tensor_hdf5() function.
        """
        if mode == 'emp':
            self.write_tensor_hdf5('empirical',  min_index, max_index)
        elif mode == 'sim':
            self.write_tensor_hdf5('train',  min_index, max_index)
            self.write_tensor_hdf5('test', min_index, max_index)
        elif mode == 'train':
            self.write_tensor_hdf5('train',  min_index, max_index)
        elif mode == 'test':
            self.write_tensor_hdf5('test', min_index, max_index)
        return
    
    def create_csv(self, data_str):
        """"Create empty csv files """
        assert data_str in ['test', 'train', 'empirical']

        # dimensions
        num_samples           = len(rep_idx)
        tree_width            = self.tree_width

        # info
        print(f'Making {data_str} csv dataset: {num_samples} examples for tree width = {tree_width}')
        
        # output csv filepaths
        out_prefix    = f'{self.fmt_dir}/{self.fmt_prefix}.{data_str}'
        out_idx_fn    = f'{out_prefix}.index.csv'
        out_phy_fn    = f'{out_prefix}.phy_data.csv'
        out_edges_fn = f'{out_prefix}.graph_edges.csv'
        out_nodes_fn = f'{out_prefix}.graph_nodes.csv'
        out_aux_fn    = f'{out_prefix}.aux_data.csv'
        out_lbl_fn    = f'{out_prefix}.labels.csv'
        for file in [out_idx_fn, out_phy_fn, out_edges_fn, out_nodes_fn, out_aux_fn, out_lbl_fn]:
            with open(file, 'w') as f:
                pass
        
    
    def create_hdf5(self, data_str):
        """Initialise one empty HDF5 file per encoding for this split.

        Files are named <fmt_prefix>.<data_str>.<encoding>.hdf5. An existing
        file of that name is first moved to <...>.<encoding>.vN.hdf5.
        """
        assert data_str in ['test', 'train', 'empirical']
        rep_idx         = self.split_idx[data_str]
        aux_data_names  = np.array(self.aux_data)
        par_est_names   = np.array(self.param_est)
        num_samples     = len(rep_idx)
        tree_width      = self.tree_width
        num_data_length = tree_width * self.num_data_col
        num_aux_data    = len(aux_data_names)
        num_par_est     = len(par_est_names)

        for enc in self.encodings:
            num_attr = self.num_attr[enc]
            print(f'Making {data_str} hdf5 dataset ({enc}, {num_attr} node features): '
                  f'{num_samples} examples for tree width = {tree_width}')

            out_hdf5_fn = self._hdf5_fn(data_str, enc)
            self._archive_existing_hdf5(out_hdf5_fn)

            with h5py.File(out_hdf5_fn, 'w') as hdf5_file:
                # fixed name arrays
                hdf5_file.create_dataset('aux_data_names', (1, num_aux_data),
                                         'S64', aux_data_names.astype('S64'), compression='gzip')
                hdf5_file.create_dataset('label_names', (1, num_par_est),
                                         'S64', par_est_names.astype('S64'), compression='gzip')
                node_attr_names = np.array(_encoding_features(enc))
                hdf5_file.create_dataset('node_attr_names', (1, num_attr),
                                         'S64', node_attr_names.astype('S64'), compression='gzip')

                # growable arrays, appended to by write_tensor_hdf5().
                # Code to work with variable length arrays is based on examples from Stack Overflow.
                def growable(name, width=None):
                    shape    = (0,) if width is None else (0, width)
                    maxshape = (None,) if width is None else (None, width)
                    hdf5_file.create_dataset(name, shape, dtype='f', compression='gzip',
                                             chunks=True, maxshape=maxshape)

                growable('idx', 1)
                growable('phy_data', num_data_length)
                growable('aux_data', num_aux_data)
                growable('labels', num_par_est)
                growable('num_edges', 1)
                growable('num_nodes', 1)
                growable('node_1')
                growable('node_2')
                growable('node_attr', num_attr)
                growable('graph_id')
        return


    def write_tensor_hdf5(self, data_str, min_ind, max_ind):
        """Writes formatted tensors to HDF5 file(s).

        Appends the current block of encoded examples to the HDF5 file of each
        encoding. Each input data point includes the phylogenetic state
        vector, the graph (edges and node features), the auxiliary data
        vector and the training label vector. Everything except node_attr is
        the same across encodings, so it is built once and written to every
        file.

        Args:
            data_str (str): specifies 'test', 'train' or 'empirical' dataset
        """

        assert data_str in ['test', 'train', 'empirical']

        res = list(self.rep_data.values())
        if len(res) == 0:
            return

        # This code for appending data is based on https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
        shared = {
            'phy_data':  np.vstack([x['phy'] for x in res]),
            'labels':    np.vstack([x['lbl'] for x in res]),
            'aux_data':  np.vstack([x['aux'] for x in res]),
            'idx':       np.vstack([k for k in self.rep_data.keys()]),
            'node_1':    np.concatenate([np.array(x['node_1']) for x in res], axis=None),
            'node_2':    np.concatenate([np.array(x['node_2']) for x in res], axis=None),
            'num_edges': np.vstack([x['num_edges'] for x in res]),
            'num_nodes': np.vstack([x['num_nodes'] for x in res]),
            'graph_id':  np.arange(len(res)),
        }

        for enc in self.encodings:
            # NOTE: graphs have different numbers of nodes, so node_attr
            # cannot be flattened across all graphs and reshaped with a
            # single global order='F' (that assumes one uniform matrix and
            # scrambles feature columns across graph boundaries). Stack
            # each graph's own (num_nodes_i, num_attr) block instead.
            node_attr_cat = np.vstack([
                np.asarray(x['node_attr'][enc], dtype=np.float64).reshape(-1, self.num_attr[enc])
                for x in res
            ])
            with h5py.File(self._hdf5_fn(data_str, enc), 'a') as hdf5_file:
                for name, new_dat in list(shared.items()) + [('node_attr', node_attr_cat)]:
                    ds = hdf5_file[name]
                    ds.resize(ds.shape[0] + new_dat.shape[0], axis=0)
                    ds[-new_dat.shape[0]:] = new_dat
        return
    
    def write_tensor_csv(self, data_str):
        """ Writes formatted tensor to CSV files.
        
        This function reads in all formatted input data the test or train
        datasets. Each input data point includes the phylogenetic state
        vector, the summary statistic vector, and the training label vector.
        Known parameters and summary statistics are converted into the
        auxiliary data vector. Data are then compiled into a different tensor
        types (phylo-state, aux. data, labels) and stored as separate CSV
        files.

        Args:
            data_str (str): specifies 'test' or 'train' dataset
        """
        if data_str in ["train", "test"]:
            rep_idx = self.split_idx[data_str]
        else:
            rep_idx = self.rep_idx
        first_aux_data_values = list(self.rep_data.values())[0]['aux']
        first_par_est_values  = list(self.rep_data.values())[0]['lbl']
        aux_data_names        = first_aux_data_values.columns.to_list()
        par_est_names         = first_par_est_values.columns.to_list()

        # dimensions
        num_samples           = len(rep_idx)
        tree_width            = self.tree_width
        num_data_length       = tree_width * self.num_data_col
        num_aux_data          = len(aux_data_names)
        num_par_est           = len(par_est_names)
        
        # info
        print(f'Making {data_str} csv dataset: {num_samples} examples for tree width = {tree_width}')
        
        # output csv filepaths
        out_prefix    = f'{self.fmt_dir}/{self.fmt_prefix}.{data_str}'
        in_prefix     = f'{self.sim_dir}/{self.sim_prefix}'
        out_idx_fn    = f'{out_prefix}.index.csv'
        out_phy_fn    = f'{out_prefix}.phy_data.csv'
        out_edges_fn = f'{out_prefix}.graph_edges.csv'
        out_nodes_fn = f'{out_prefix}.graph_nodes.csv'
        out_aux_fn    = f'{out_prefix}.aux_data.csv'
        out_lbl_fn    = f'{out_prefix}.labels.csv'
        
        # phylo-state data tensor
        with open(out_phy_fn, 'a') as outfile:
            for idx in rep_idx:
                x = self.rep_data[idx]['phy']
                s = util.ndarray_to_flat_str(x) + '\n'
                outfile.write(s)

        with open(out_edges_fn, 'a') as outfile:
            for idx in rep_idx:
                x = self.rep_data[idx]['graph_edges']
                outfile.write(x)
        
        with open(out_nodes_fn, 'a') as outfile:
            for idx in rep_idx:
                x = self.rep_data[idx]['graph_nodes']
                outfile.write(x)

        # aux. data tensor
        with open(out_aux_fn, 'a') as outfile:
            is_first = True
            for idx in rep_idx:
                x = self.rep_data[idx]['aux']
                s = ''
                if is_first:
                    s += ','.join(aux_data_names) + '\n'
                    is_first = False
                s += util.ndarray_to_flat_str(x.to_numpy()) + '\n'
                outfile.write(s)
                    
        # labels tensor
        with open(out_lbl_fn, 'a') as outfile:
            is_first = True
            for idx in rep_idx:
                x = self.rep_data[idx]['lbl']
                s = ''
                if is_first:
                    s += ','.join(par_est_names) + '\n'
                    is_first = False
                s += util.ndarray_to_flat_str(x.to_numpy()) + '\n'
                outfile.write(s)
            
        # replicate index
        df_idx = pd.DataFrame(rep_idx, columns=['idx'])
        df_idx.to_csv(out_idx_fn, index=False, mode='a')

        return

    def encode_one_star(self, args):
        """Wrapper for encode_one w/ unpacked args"""
        return self.encode_one(*args)

    def encode_one(self, tmp_fn, idx, mode, save_phyenc_csv=False):
        """
        Encode a single simulated raw dataset into tensor format.
         
        This function transforms raw input into tensor outputs. The inputs are
        read from a tree file and data matrix file. The tree is filtered to
        target ranges of taxon counts and pruned of non-extant taxa.
        
        Trees are then binned into tree-width categories (number of columns for
        compact vector representation). Next, trees and character data are
        encoded into extended CBLV or CDV formats that contain additional state
        and branch length information.

        Arguments:
            tmp_fn (str):            prefix for individual dataset
            idx (int):               replicate index for simulation
            mode (str):              specifies 'sim' or 'emp' dataset
            save_phyenc_csv (bool):  save phylogenetic state tensor as csv?

        Returns:
            cpvs (numpy.array): compact phylo vector + states (CPVS)
        """
        np.set_printoptions(formatter={'float': lambda x: format(x, '8.6E')},
                            precision=util.OUTPUT_PRECISION)
        
        # make filenames
        dat_ext = ''
        if self.char_format == 'nexus':
            dat_ext = '.nex'
        elif self.char_format == 'csv':
            dat_ext = '.csv'
        
        dat_fn     = tmp_fn + '.dat' + dat_ext
        tre_fn     = tmp_fn + '.tre'
        prune_fn   = tmp_fn + '.extant.tre'
        down_fn    = tmp_fn + '.downsampled.tre'
        cpsv_fn    = tmp_fn + '.phy_data.csv'
        graph_edges_fn   = tmp_fn + '.graph_edges.csv'
        graph_nodes_fn = tmp_fn + '.graph_nodes.csv'
        aux_fn     = tmp_fn + '.aux_data.csv'
        lbl_fn     = tmp_fn + '.labels.csv'
        par_est_fn = tmp_fn + '.param_est.csv'
       
        # check if key files exist
        if not os.path.exists(dat_fn):
            self.logger.write_log('fmt', f'Cannot find {dat_fn}')
            return
        if not os.path.exists(tre_fn):
            self.logger.write_log('fmt', f'Cannot find {tre_fn}')
            return
        if not os.path.exists(lbl_fn) and mode == 'sim':
            self.logger.write_log('fmt', f'Cannot find {lbl_fn}')
            return
       
        # read in nexus data file as numpy array
        dat = None
        if self.char_format == 'nexus':
            dat = util.convert_nexus_to_array(dat_fn,
                                                   self.char_encode,
                                                   self.num_states)
            
        elif self.char_format == 'csv':
            dat = util.convert_csv_to_array(dat_fn,
                                                 self.char_encode,
                                                 self.num_states)
        
        # verify data dimensions match what phyddle expects
        expected_char_dim = util.get_num_char_col(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        
        # get tree file
        phy = util.read_tree(tre_fn)

        # abort if simulation is invalid
        if dat is None:
            self.logger.write_log('fmt', f'Cannot process {dat_fn}')
            return
        if phy is None:
            self.logger.write_log('fmt', f'Cannot process {tre_fn}')
            return

        if dat.shape[0] != expected_char_dim:
            # improve message with more details later
            msg = (f'The numbers of characters and states in the character data'
                   f' file {dat_fn} do not match the config file settings for'
                   f' num_char, num_states, and char_encode.')
            util.print_err(msg)
            # program quits

        # prune tree, if needed
        num_taxa_before_prune = len(phy.leaf_nodes())
        if self.tree_encode == 'extant':
            phy = util.make_prune_phy(phy, prune_fn, self.rel_extant_age_tol)
            if phy is None:
                # abort, no valid pruned tree
                self.logger.write_log('fmt', f'Invalid pruned tree for {tre_fn}')
                return

        # downsample taxa
        num_taxa_orig = len(phy.leaf_nodes())
        phy = util.make_downsample_phy(
            phy, down_fn,
            max_taxa=self.tree_width,
            strategy=self.downsample_taxa)

        # get tree size
        num_taxa = len(phy.leaf_nodes())

        # tree size error checking
        if num_taxa < num_taxa_before_prune and mode == 'emp' and self.tree_encode == 'extant':
            tol = self.rel_extant_age_tol
            util.print_warn(f'Difficulty processing {tre_fn}.\n'
                            f'         Original empirical tree should contain {num_taxa_before_prune} extant taxa,\n'
                            f'         but Format detected {(num_taxa_before_prune-num_taxa)} non-extant taxa.\n' 
                            f'         Node ages might not be close enough to the value 0.\n'
                            f'         Try using larger value for --rel_extant_age_tol to convert\n'
                            f'         nodes with near-zero ages to zero (curr. val {tol})\n')
        if num_taxa > self.tree_width:
            # abort, too many taxa
            self.logger.write_log('fmt', f'Too many taxa {tre_fn} has too many taxa')
            return
        if num_taxa < self.min_num_taxa or num_taxa < 0:
            # report special warning for empirical + extant analyses
            # always log warning when too few taxa
            self.logger.write_log('fmt', f'Too few taxa (<{self.min_num_taxa}) for {tre_fn}')
            return



        phy_str = phy.as_string(schema="newick")
        global offset
        # one R call per tree: with several encodings, ask R for every
        # feature ('all') and split the columns per encoding below
        r_encoding = 'all' if len(self.encodings) > 1 else self.encoding_type
        graph_data_base = gen_graph(ape.read_tree(text=phy_str), offset=0, encoding=r_encoding)

        graph_data = gen_graph(ape.read_tree(text=phy_str), offset=0)
        n1 = pd.Series(graph_data[1].rx2('node1'))
        n2 = pd.Series(graph_data[1].rx2('node2'))

        node_1 = pd.concat([n1,n2], ignore_index=True)
        node_2 = pd.concat([n2,n1], ignore_index=True)
        edge_data = pd.DataFrame({
            'from': node_1,
            'to': node_2
            })

        # all node-feature columns R returned, then each encoding's subset
        # (in R's order). Reading fixed column names fails for encodings
        # where a column is absent (rx2() then returns R NULL).
        node_df_r = graph_data_base[0]
        node_df = pd.DataFrame({
            str(name): np.asarray(node_df_r.rx2(name), dtype=np.float64)
            for name in node_df_r.names
        })
        node_attr = {}
        for enc in self.encodings:
            cols = _encoding_features(enc)
            missing = [c for c in cols if c not in node_df.columns]
            if missing:
                raise ValueError(f'{tre_fn}: encoding "{enc}" needs node features '
                                 f'{missing}, which R did not return '
                                 f'(got {list(node_df.columns)})')
            node_attr[enc] = node_df[cols]
        nodes_dist = node_df['dist_to_root'].to_numpy()

        num_edges = max(len(node_1), len(node_2))
        num_nodes = len(nodes_dist)
        offset = offset + num_edges

                
        if (len(nodes_dist) != max(node_1.max(), node_2.max())):
            print("nodes_dist:", len(nodes_dist))
            print("max edge node:", max(node_1.max(), node_2.max()))
            print("unique edge nodes:", len(np.unique(np.concatenate([node_1,node_2]))))
            print(graph_data)
            quit()

        save_graph_csv_ = self.save_graph_csv # or save_graph_csv
        if save_graph_csv_  and graph_data is not None:
            graph_str =  edge_data.to_string(index_names=False, index=False, header=False)#util.ndarray_to_flat_str(np.array(graph_data[1])) + "\n" #util.ndarray_to_flat_str(np.array(graph_data)) + '\n'
            util.write_to_file(graph_str, graph_edges_fn)
            util.write_to_file(util.ndarray_to_flat_str(np.array(graph_data[0])) + "\n", graph_nodes_fn)
        # create compact phylo-state vector, CPV+S = {CBLV+S, CDV+S}
        cpvs_data = util.encode_cpvs(phy, dat, tree_width=self.tree_width,
                                     tree_encode_type=self.brlen_encode,
                                     tree_type=self.tree_encode, idx=idx)

        # save CPVS
        save_phyenc_csv_ = self.save_phyenc_csv or save_phyenc_csv
        if save_phyenc_csv_ and cpvs_data is not None:
            cpsv_str = util.ndarray_to_flat_str(cpvs_data.flatten()) + '\n'
            util.write_to_file(cpsv_str, cpsv_fn)
            util.decode_cdvs_string(cpsv_str)

        # read in labels file
        labels = None
        if mode == 'sim' or len(self.param_data) > 0:
            labels = pd.read_csv(lbl_fn, header=0)
            try:
                if labels['model_type'].iloc[0] not in self.select_classes:
                    return None
            except:
                print("labels:", labels)
                print("exception")
                return None
            
        # process parameters to estimate
        param_est = pd.DataFrame()
        if mode == 'sim':
            # match param_est names to label columns ignoring case, '_' and '-'
            # (e.g. 'modeltype' in the config matches 'model_type' in labels.csv)
            def norm(s):
                return str(s).lower().replace('_', '').replace('-', '')
            label_cols = {norm(c): c for c in labels.columns}
            for header in self.param_est:
                if header in labels.columns:
                    continue
                src = label_cols.get(norm(header))
                if src is not None:
                    labels[header] = labels[src]
                else:
                    # not defined for this model (e.g. BiSSE/MuSSE rates on a BD tree)
                    labels[header] = 10000000
            param_est = labels[self.param_est]
            # check label matching

            for k in self.param_est:
                if k not in param_est.columns:
                    util.print_warn(f"Estimated parameter '{k}' missing in labels. Skipped.")
            
            # save param_est
            param_est.to_csv(par_est_fn, index=False, float_format=util.PANDAS_FLOAT_FMT_STR)
    
            # set empty param dataframes as None
            if len(param_est.columns) == 0:
                param_est = None

        # print("param_est:", param_est)

        # process parameters that are data
        param_data = pd.DataFrame()
        if mode == 'sim' or len(self.param_data) > 0:
            param_data = labels[self.param_data]
            for k in self.param_data:
                if k not in param_data.columns:
                    util.print_warn(f"Data parameter '{k}' missing in labels. Skipped.")

        # record summ stat data
        summ_stat = self.make_summ_stat(phy, dat)
        summ_stat['num_taxa'] = [ num_taxa_orig ]
        summ_stat['prop_taxa'] = [ num_taxa / num_taxa_orig ]
        
        # make aux. data from  "known" data parameters and sum. stats
        aux_data = pd.concat(objs=[summ_stat, param_data], axis=1)

        # save summ. stats.
        aux_data.to_csv(aux_fn, index=False, float_format=util.PANDAS_FLOAT_FMT_STR)

    
        # done!
        # node_attr is a dict {encoding: DataFrame of node features}
        return idx, cpvs_data, node_attr, node_1, node_2, num_edges, num_nodes, aux_data, param_est # second argument was cpvs_data
    
    def make_summ_stat(self, phy, dat):
        """
        Generate summary statistics.

        This function populates a dictionary of summary statistics. Keys
        are the names of statistics that later appear as column names in the
        summ_stat and aux_data files and containers. Values are for the
        inputted phy and dat variables.

        Arguments:
            phy (dendropy.Tree): phylogenetic tree
            dat (numpy.array): character data

        Returns:
            summ_stats (dict): summary statistics
        """
        # new dictionary to return
        summ_stats = {}

        # small dx to offset zero-valued things
        zero_offset = 1E-4
        one_offset = 1E+4
        
        # return default summ stats if phy not valid
        if phy is not None:
            
            # read basic info from phylogenetic tree
            num_taxa                  = len(phy.leaf_nodes())
            node_ages                 = phy.internal_node_ages(ultrametricity_precision=False)
            root_age                  = phy.seed_node.age
            tree_height               = root_age
            if phy.seed_node.edge.length is not None:
                tree_height           += phy.seed_node.edge.length
            branch_lengths            = [ nd.edge.length for nd in phy.nodes() if nd != phy.seed_node ]
            
            # tree statistics
            summ_stats['log10_tree_length'] = np.log10( phy.length() )
            summ_stats['log10_root_age']    = np.log10( root_age )
            if root_age != tree_height:
                summ_stats['log10_tree_height'] = np.log10( tree_height )
            summ_stats['log10_brlen_mean']  = np.log10( np.mean(branch_lengths) )
            summ_stats['log10_age_mean']    = np.log10( np.mean(node_ages) )
            summ_stats['log10_B1']          = np.log10( dp.calculate.treemeasure.B1(phy) )
            try:
                summ_stats['colless'] = np.log10( dp.calculate.treemeasure.colless_tree_imbalance(phy) + zero_offset )
            except ZeroDivisionError:
                summ_stats['colless'] = np.log10(zero_offset)
            summ_stats['age_var']     = np.log10( np.var(node_ages) )
            summ_stats['brlen_var']   = np.log10( np.var(branch_lengths) )
            summ_stats['treeness']    = np.log10( dp.calculate.treemeasure.treeness(phy) )
            summ_stats['N_bar']       = np.log10( dp.calculate.treemeasure.N_bar(phy) )

        # frequencies of character states
        if self.char_encode == 'integer':

            # initialize state freqs
            states = list(range(self.num_states))
            for i in states:
                summ_stats[f'f_dat_{i}'] = 0.0
                summ_stats[f'n_dat_{i}'] = 0.0

            # fill-in non-zero state freqs (match counts to the state value,
            # so states absent from a tree keep a count of 0).
            # dat can hold the states as strings ('0', '1', ...) together with
            # the csv header ('data'), so only whole-number values are counted.
            vals = pd.to_numeric(pd.Series(np.asarray(dat, dtype=object).ravel()),
                                 errors='coerce').dropna().to_numpy()
            vals = vals[np.isclose(vals, np.round(vals))].astype(np.int64)
            unique, counts = np.unique(vals, return_counts=True)
            for u, j in zip(unique, counts):
                i = int(u)
                if i in states:
                    summ_stats[f'f_dat_{i}'] = j / num_taxa
                    summ_stats[f'n_dat_{i}'] = j
                
            for i in states:
                f = summ_stats[f'f_dat_{i}']
                n = summ_stats[f'n_dat_{i}']
                # if f == 0.0:
                #     f = zero_offset
                # elif f == 1.0:
                #     f = one_offset
                # else:
                #     # print(f)
                #     f = np.log(f / (1.0 - f))
                if n == 0.0:
                    n = zero_offset
                else:
                    n = np.log(n)
                summ_stats[f'f_dat_{i}'] = f
                summ_stats[f'n_dat_{i}'] = n
                
        elif self.char_encode == 'one_hot':
            # one-hot-encoded states
            for i in range(dat.shape[0]):
                f = np.sum(dat.iloc[i]) / num_taxa
                n = np.sum(dat.iloc[i])
                # if f == 0.0:
                #     f = zero_offset
                # elif f == 1.0:
                #     f = one_offset
                # else:
                #     f = np.log(f / (1.0 - f))
                if n == 0.0:
                    n = zero_offset
                else:
                    n = np.log(n)
                summ_stats['f_dat_' + str(i)] = f
                summ_stats['n_dat_' + str(i)] = n

        df = pd.DataFrame(summ_stats, index=[0])

        # done
        return df
    
    def encode_cpvs(self, phy, dat, tree_width, tree_type,
                    tree_encode_type, idx, rescale=True):
        """
        Encode Compact Phylogenetic Vector + States (CPV+S) array

        This function encodes the dataset into Compact Bijective Ladderized
        Vector + States (CBLV+S) when tree_type is 'serial' or Compact
        Diversity-Reordered Vector + States (CDV+S) when tree_type is 'extant'.

        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa)
                                     in CPVS array
            tree_type (str):         type of the tree ('serial' or 'extant')
            tree_encode_type (str):  type of tree encoding ('height_only' or
                                     'height_brlen')
            idx (int):               replicate index
            rescale (bool):          set tree height to 1 then encode, if True

        Returns:
            cpvs (numpy.array):      CPV+S encoded tensor
        """
        
        cpvs = util.encode_cpvs(phy, dat, tree_width, tree_type,
                                tree_encode_type, idx, rescale)
        return cpvs
    
    def encode_cdvs(self, phy, dat, tree_width, tree_encode_type, rescale=True):
        """
        Encode Compact Diversity-reordered Vector + States (CDV+S) array

        # num columns equals tree_width, 0-padding
        # returns tensor with following rows
        # 0:  internal node root-distance
        # 1:  leaf node branch length
        # 2:  internal node branch length
        # 3+: state encoding

        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa)
                                     in CPVS array
            tree_encode_type (str):  type of tree encoding ('height_only' or
                                     'height_brlen')
            rescale:                 set tree height to 1 then encode, if True

        Returns:
            numpy.ndarray: The encoded CDV+S tensor.
        """

        cdvs = util.encode_cdvs(phy, dat, tree_width,
                                tree_encode_type, rescale)
        return cdvs

    def encode_cblvs(self, phy, dat, tree_width, tree_encode_type, rescale=True):
        """
        Encode Compact Bijective Ladderized Vector + States (CBLV+S) array

        # num columns equals tree_width, 0-padding
        # returns tensor with following rows
        # 0:  leaf node-to-last internal node distance
        # 1:  internal node root-distance
        # 2:  leaf node branch length
        # 3:  internal node branch length
        # 4+: state encoding

        Arguments:
            phy (dendropy.Tree):     phylogenetic tree
            dat (numpy.array):       character data
            tree_width (int):        number of columns (max. num. taxa)
                                     in CPVS array
            tree_encode_type (str):  type of tree encoding ('height_only' or
                                     'height_brlen')
            rescale:                 set tree height to 1 then encode, if True

        Returns:
            numpy.ndarray: The encoded CBLV+S tensor.
        """
        
        cblvs = util.encode_cblvs(phy, dat, tree_width, tree_encode_type, rescale)
        return cblvs
    

    def make_pca(self):
        """
        Make PCA of simulated dataset.
        """

        # load data
        self.sim_data = self.load_data(mode='sim')
        # emp_data = self.load_data(mode='emp')
        
        # for each dataset type
        self.scaler = {}
        self.pca = {}
        for s in ['phy_data', 'aux_data', 'labels']:
            
            # data
            sim_dat = self.sim_data[s]
            # emp_dat = emp_data[s]
            
            # scaler
            self.scaler[s] = StandardScaler()
            sim_scaler = self.scaler[s].fit_transform(sim_dat)
            # emp_scaler = scaler[s].transform(emp_dat)

            # pca
            num_comp = min(sim_dat.shape)
            # print(sim_dat.shape)
            # print(sim_scaler)
            self.pca[s] = PCA(n_components=num_comp, whiten=True)
            sim_pca = self.pca[s].fit_transform(sim_dat)
            # emp_pca = pca[s].transform(emp_dat)
        
        return
    
    def check_pca(self, mode):
        # load data
        self.emp_data = self.load_data(mode='emp')

        # for each dataset type
        self.scaler = {}
        self.pca = {}
        for s in ['phy_data', 'aux_data', 'labels']:

            # data
            sim_dat = self.sim_data[s]
            emp_dat = self.emp_data[s]

            # scaler
            self.scaler[s] = StandardScaler()
            sim_scaler = self.scaler[s].fit_transform(sim_dat)
            # emp_scaler = scaler[s].transform(emp_dat)

            # pca
            num_comp = min(sim_dat.shape)
            self.pca[s] = PCA(n_components=num_comp, whiten=True)
            sim_pca = self.pca[s].fit_transform(sim_dat)
            # emp_pca = pca[s].transform(emp_dat)

        return

    def load_data(self, mode):
        """Load in full dataset for given filename prefix.

        With several encodings, the HDF5 file of the first encoding is read
        (phy_data, aux_data, labels and idx are the same in every file)."""
        # This should probably be put into utilities and reused
        # in all other steps.

        # initialize variables
        phy_data = None
        aux_data = None
        idx_data = None
        labels = None
        label_names = None
        aux_data_names = None

        # simulated test datasets for csv or hdf5
        path_prefix = ''
        if mode == 'sim':
            path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.test'
        elif mode == 'emp':
            path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.empirical'
        phy_data_fn = f'{path_prefix}.phy_data.csv'
        aux_data_fn = f'{path_prefix}.aux_data.csv'
        idx_data_fn = f'{path_prefix}.index.csv'
        labels_fn = f'{path_prefix}.labels.csv'
        if self.tensor_format == 'hdf5':
            hdf5_fn = self._find_hdf5_fn('test' if mode == 'sim' else 'empirical')
            hdf5_file = h5py.File(hdf5_fn, 'r')
            phy_data = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
        # read data

            hdf5_file = h5py.File(hdf5_fn, 'r')
            phy_data = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
            aux_data = pd.DataFrame(hdf5_file['aux_data']).to_numpy()
            print("aux data 1580", aux_data)
            idx_data = pd.DataFrame(hdf5_file['idx'], columns=['idx'])
            labels = pd.DataFrame(hdf5_file['labels']).to_numpy()
            label_names = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            hdf5_file.close()
        
        ret = {
            'phy_data': phy_data,
            'aux_data': aux_data,
            'idx_data': idx_data,
            'labels': labels,
            'label_names': label_names,
            'aux_data_names': aux_data_names
            }
        return ret