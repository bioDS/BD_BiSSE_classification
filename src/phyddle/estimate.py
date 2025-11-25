#!/usr/bin/env python
"""
estimate
========
Defines classes and methods for the Estimate step, which loads a pre-trained
network and uses it to generate new estimates, e.g. estimate model parameters
for a new empirical dataset.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2025, Michael Landis and Ammon Thompson
License:   MIT
"""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm

# standard imports
import os
import sys
sys.stdout.reconfigure(line_buffering=True)

# external imports
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import torch
import subprocess, pickle

from torch_geometric.data import Dataset as Geoset, Data as GeoData, Batch as GeoBatch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
r = robjects.r
r['source']('~/AIphylo/phyddle/workspace/pj_phyddle/MLE/mle.R')
get_mle = r['get_mle_label']




# phyddle imports
from phyddle import utilities as util

##################################################
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
        for i in range(len(graph_ids)):
            current_edge_ind = prev_edge_ind + num_edges[i].astype(int)[0]
            current_node_ind = prev_node_ind + num_nodes[i].astype(int)[0]

            selected_nodes = node_data[prev_node_ind:current_node_ind]
            selected_edges = edges_data[:, prev_edge_ind:current_edge_ind]


        
            if (selected_nodes.shape[0] != selected_edges.shape[1] + 1):
                print("MISMATCH!!!!")
                quit()
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
        return Estimator(args)
    else:
        return NotImplementedError

# class GraphOnlyWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, data, *args, **kwargs):
#         # Ignore all non-graph arguments from DataLoader
#         return self.model(data)

##################################################





class Estimator:
    """
    Class for making neural network estimates (i.e. label predictions) from new
    (e.g. empirical) phylogenetic datasets. This class requires a trained
    network from Train and input processed by Format. Output is written to
    file and can be visualized using Plot.
    """

    def __init__(self, args):
        """Initializes a new Simulator object.

        Args:
            args (dict): Contains phyddle settings.
            
        """
        
        # settings
        self.verbose            = bool(args['verbose'])
        self.no_sim             = bool(args['no_sim'])
        self.no_emp             = bool(args['no_emp'])
        
        # filesystem
        self.trn_prefix         = str(args['trn_prefix'])
        self.fmt_prefix         = str(args['fmt_prefix'])
        self.est_prefix         = str(args['est_prefix'])
        self.trn_dir            = str(args['trn_dir'])
        self.fmt_dir            = str(args['fmt_dir'])
        self.est_dir            = str(args['est_dir'])
        self.sim_dir            = str(args['sim_dir'])
        self.log_dir            = str(args['log_dir'])

        self.calc_MLE           =bool(args['calc_MLE'])
        self.load_MLE           =bool(args['load_MLE'])
        
        # dimensions
        self.tree_encode        = str(args['tree_encode'])
        self.char_encode        = str(args['char_encode'])
        self.brlen_encode       = str(args['brlen_encode'])
        self.tensor_format      = str(args['tensor_format'])
        self.num_char           = int(args['num_char'])
        self.num_states         = int(args['num_states'])
        self.param_est          = dict(args['param_est'])        
        self.param_cat          = dict() 
        self.param_cat_names = list()  

        self.log_offset         = float(args['log_offset'])
        self.use_cuda           = bool(args['use_cuda'])
        
        # error checking
        self.warn_aux_outlier   = float(args['warn_aux_outlier'])
        self.warn_lbl_outlier   = float(args['warn_lbl_outlier'])
        self.est_aux_data_raw   = None
        self.est_labels_num_raw = None

        self.phylo_pool         =bool(args['phylo_pool'])
        self.graph_conv         =bool(args['graph_conv'])
        self.phy_hidden_size = int(args['phy_hidden_size'])
        self.optimizer          = str(args['optimizer'])
        self.num_classes        =  int(args['num_classes'])
        self.scheduler="NA"

        self.learning_rate      = float(args['learning_rate'])
        
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

        self.mle_labels = None

        self.has_label_num = len(self.label_num_names) > 0
        self.has_label_cat = len(self.label_cat_names) > 0
        
        # create logger to track runtime info
        self.logger = util.Logger(args)

        # initialized later
        self.train_aux_data_mean_sd     = None       # init in load_train_input()
        self.train_labels_num_mean_sd   = None       # init in load_train_input()
        self.cpi_adjustments            = None       # init in load_train_input()
        self.phy_data                   = None       # init in load_format_input()
        self.aux_data                   = None       # init in load_format_input()
        self.idx_data                   = None       # init in load_format_input()
        self.graph_data                 = None
        self.aux_data_names             = None       # init in load_format_input()
        self.true_labels_num            = None       # init in load_format_input()
        self.true_labels_cat            = None       # init in load_format_input()
        self.est_labels_num             = None       # init in make_results()
        self.mymodel                    = None       # init in make_results()

        self.num_sample = -1
        
        # done
        return

    def tree_generator(self, file_names):
        for f in file_names:
            with open(f, "r") as open_f:
                yield open_f.read()

    def MLE(self, idx):
        # print("idx data:")
        # print(idx.iloc[:, 0])
        # [f'{self.sim_dir}/{self.fmt_prefix}.{i}.tre' for i in idx.iloc[:, 0]]
        file_names = [f'{self.sim_dir}/{self.fmt_prefix}.{i}.tre' for i in idx.iloc[:, 0]]
        phylogenies = []
        for f in file_names:
            with open(f, "r") as open_file:
                phylogenies.append(open_file.read())
        # phylogenies = [open(f, "r").read() for f in file_names]

        # batch_size = 1
        # batches = [phylogenies[i:i+batch_size] for i in range(0, len(phylogenies), batch_size)]
        # results = thread_map(self.batch_mle, batches, max_workers=70, desc="Calculating MLE")
        mle_labels = []
        with ThreadPoolExecutor(max_workers=80) as executor:
            futures = {executor.submit(self.single_mle, phy): i for i, phy in enumerate(phylogenies)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating MLE", ncols=100):
                mle_labels.append(future.result())
        cat("mle labels:")
        print(mle_labels)        
        return mle_labels

    def batch_mle(self, batch):
        results = [self.single_mle(phy) for phy in batch]
        return results

    def single_mle(self, phy):
        # print("opening single_mle")
        p = subprocess.Popen(
        ["python", "parallel_mle.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        )
        try:
            pickle.dump(phy, p.stdin)
            p.stdin.close()
            mle_label = pickle.load(p.stdout)
        except Exception:
            p.kill()
            p.wait()
            raise
        finally:
            p.stdout.close()
            p.stderr.close()
            p.wait()
        # print(mle_label)
        return mle_label
    
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

        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        # print estimate settings
        util.print_str('▪ Estimation targets:', verbose)
        num_ljust = max([len(k) for k in self.param_est.keys()])
        for k,v in self.param_est.items():
            util.print_str(f'  ▪ {k.ljust(num_ljust)}  [type: {v}]', verbose)


        # load Train input
        util.print_str('▪ Preparing network', verbose)
        device_info = ''
        if self.TORCH_DEVICE_STR == 'cuda':
            device_info = '  ▪ using CUDA + GPU'
            device_info += '  [device: ' + torch.cuda.get_device_properties(0).name + ']'
        elif self.TORCH_DEVICE_STR == 'cpu':
            num_cpu = os.cpu_count()
            device_info = '  ▪ using CPUs  [num: ' + str(num_cpu) + ']'
        if device_info != '':
            util.print_str(device_info, verbose)

        self.load_train_input()
        
        found_sim = False
        if self.no_sim:
            # skip sim
            util.print_str('▪ Skipping simulated test input', verbose)
            
        elif self.has_valid_dataset(mode='sim'):
            # load input
            util.print_str('▪ Loading simulated test input', verbose)
            self.load_format_input(mode='sim')
    
            # make estimates
            util.print_str('▪ Making simulated test estimates', verbose)
            self.make_results(mode='sim')
            
            # done
            found_sim = True

        found_emp = False
        if self.no_emp:
            # skip emp
            util.print_str('▪ Skipping empirical test input', verbose)
            
        if self.has_valid_dataset(mode='emp'):
            # load input
            util.print_str('▪ Loading empirical input', verbose)
            self.load_format_input(mode='emp')
    
            # make estimates
            util.print_str('▪ Making empirical estimates', verbose)
            self.make_results(mode='emp')
            
            # check inputs/outputs
            self.check_empirical_results()

            # done
            found_emp = True

        # notify user if no work done
        if self.no_emp and self.no_sim:
            util.print_warn('Estimate has no work to do when no_sim '
                            'and no_emp are used together.')
        elif not found_sim and not found_emp:
            util.print_warn('No simulated test or empirical datasets found. '
                            'Check config settings.', verbose)
        
        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        # done
        util.print_str('... done!', verbose)

        if self.calc_MLE:
            self.mle_labels = self.MLE(self.idx_data)
            self.mle_labels.to_csv(out_est_labels_num_fn, index=False, sep=',',
                                      float_format=util.PANDAS_FLOAT_FMT_STR)
        elif self.load_MLE:
            self.mle_labels = pd.read_csv(out_est_labels_num_fn, sep=',', index_col=False)
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

    def has_valid_dataset(self, mode='sim'):
        """Determines if empirical analysis is being performed.
        
        Args:
            mode (str): 'sim' or 'emp' for simulated or empirical analysis.
            
        Returns:
            bool: True if empirical analysis is being performed.
        """
        assert mode in ['sim', 'emp']

        # check if empirical directory exists
        if not os.path.exists(self.fmt_dir):
            print("empirical directory not found")
            return False

        data_src = None
        if mode == 'emp':
            data_src = 'empirical'
        elif mode == 'sim':
            data_src = 'test'
        # check if empirical directory contains files
        files = ['']


        if self.tensor_format == 'hdf5':
            files = [ f'{self.fmt_dir}/{self.fmt_prefix}.{data_src}.hdf5' ] # {self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}
        elif self.tensor_format == 'csv':
            files = [ f'{self.fmt_dir}/{self.fmt_prefix}.{data_src}.phy_data.csv',
             f'{self.fmt_dir}/{self.fmt_prefix}.{data_src}.aux_data.csv' ]
        # fail if key file missing
        for fn in files:
            if not os.path.exists(fn):
                return False
        
        return True
    
    def load_train_input(self):
        """Load input data for estimation.

        This function loads input from Train and Estimate. From Train, it
        imports the trained network, scaling factors for the aux. data and
        labels. It also loads the phy. data and aux. data tensors stored in the
        Estimate job directory.
        
        The script re-normalizes the new estimation to match the scale/location
        used for simulated training examples to train the network.
            
        """
        # filesystem
        path_prefix = f'{self.trn_dir}/{self.trn_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}'
        train_norm_aux_data_fn = f'{path_prefix}.train_norm.aux_data.csv'
        train_norm_labels_num_fn = f'{path_prefix}.train_norm.labels_num.csv'
        model_cpi_fn = f'{path_prefix}.cpi_adjustments.csv'

        # denormalization factors for new aux data
        train_aux_data_norm = pd.read_csv(train_norm_aux_data_fn, sep=',', index_col=False)
        train_aux_data_means = train_aux_data_norm['mean'].T.to_numpy().flatten()
        train_aux_data_sd = train_aux_data_norm['sd'].T.to_numpy().flatten()
        self.train_aux_data_mean_sd = (train_aux_data_means, train_aux_data_sd)
        
        if self.has_label_num:
            # denormalization factors for labels
            train_norm_labels_num = pd.read_csv(train_norm_labels_num_fn, sep=',', index_col=False)
            train_num_labels_mean = train_norm_labels_num['mean'].T.to_numpy().flatten()
            train_num_labels_sd = train_norm_labels_num['sd'].T.to_numpy().flatten()
            self.train_labels_num_mean_sd = (train_num_labels_mean, train_num_labels_sd)
            
            # read in CQR interval adjustments
            self.cpi_adjustments = pd.read_csv(model_cpi_fn, sep=',', index_col=False).to_numpy()
            
        # done
        return

    def load_format_input(self, mode='sim'):
        """Load input data for estimation.

        This function loads the phy. data and aux. data tensors stored in the
        Format job directory.
        
        Args:
            mode (str): 'sim' or 'emp' for simulated or empirical analysis.
            
        """

        assert mode in ['sim', 'emp']
        
        path_prefix = ''
        if mode == 'sim':
            path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.test'
            short_path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.test'
        elif mode == 'emp':
            path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.empirical'
            short_path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.empirical'
        
        # simulated test datasets for csv or hdf5
        phy_data_fn = f'{path_prefix}.phy_data.csv'
        aux_data_fn = f'{path_prefix}.aux_data.csv'
        idx_data_fn = f'{path_prefix}.index.csv'
        labels_fn = f'{path_prefix}.labels.csv'
        hdf5_fn = f'{short_path_prefix}.hdf5'
        
        # load all the test dataset
        phy_data = None
        aux_data = None
        idx_data = None
        graph_data = None
        labels = None
        label_names = None
        nodes_dist = None
        node_1_data = None
        node_2_data = None
        if self.tensor_format == 'csv':
            phy_data = pd.read_csv(phy_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            aux_data = pd.read_csv(aux_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            idx_data = pd.read_csv(idx_data_fn,
                                        on_bad_lines='skip')
            if mode == 'sim':
                labels = pd.read_csv(labels_fn, header=None,
                                            on_bad_lines='skip').to_numpy()
                label_names = labels[0,:]
                labels = labels[1:,:].astype('float64')
            aux_data = aux_data[1:,:].astype('float64')
            aux_data_names = aux_data[0,:]

        elif self.tensor_format == 'hdf5':
            hdf5_file = h5py.File(hdf5_fn, 'r')
            phy_data = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
            aux_data = pd.DataFrame(hdf5_file['aux_data']).to_numpy()
            idx_data = pd.DataFrame(hdf5_file['idx'], columns=['idx']).to_numpy()
            node_1_data = pd.DataFrame(hdf5_file['node_1'])#.to_numpy()
            node_2_data = pd.DataFrame(hdf5_file['node_2'])#.to_numpy()
            nodes_dist = pd.DataFrame(hdf5_file['nodes_dist']).to_numpy()
            graph_ids = pd.DataFrame(hdf5_file['graph_id']).to_numpy()
            num_nodes = pd.DataFrame(hdf5_file['num_nodes']).to_numpy()
            num_edges = pd.DataFrame(hdf5_file['num_edges']).to_numpy()
            self.label_names = [s.decode() for s in hdf5_file['label_names'][0,:] ]



            # idx_data = idx_data[:,:].astype('int')
            if mode == 'sim':
                labels = pd.DataFrame(hdf5_file['labels']).to_numpy()
            label_names = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            hdf5_file.close()
        
        edges_matrix = pd.concat((node_1_data, node_2_data), axis=1).to_numpy().T
        node_attributes = nodes_dist.flatten()

        labels_num, labels_cat = self.separate_labels(labels)
        self.num_sample = phy_data.shape[0]
        phy_data.shape = (self.num_sample, -1, self.num_data_col)

        unique_ids = np.unique(graph_ids)

        self.estimate_dataset = Dataset(phy_data, node_attributes,
                                             edges_matrix, aux_data,
                                             idx_data,
                                             labels_num,
                                             labels_cat, unique_ids, num_nodes, num_edges)

        # self.phy_data    = torch.from_numpy(np.transpose(phy_data, axes=[0,2,1]).astype('float32'))
        # self.aux_data    = torch.from_numpy(aux_data.astype('float32'))
        self.idx_data    = torch.from_numpy(idx_data.astype('int'))
        # self.labels_num  = torch.from_numpy(labels_num.astype('float32'))
        # self.labels_cat  = torch.from_numpy(labels_cat.astype('int'))

        # edges_matrix = torch.from_numpy(edges_matrix.astype('int'))
        # min_index =   edges_matrix.min()
        # edges_matrix = torch.sub(edges_matrix, min_index)
        # unique = torch.unique(edges_matrix)
        # all_from_unique = torch.arange(edges_matrix.max()+1)
        # difference = all_from_unique[torch.isin(all_from_unique, unique, invert=True)]
        # reduction = torch.searchsorted(difference, all_from_unique, right=False)
        # edges_matrix =  torch.sub(edges_matrix, reduction[edges_matrix])

        # self.graph_data = []
        # graph_ids = np.unique(graph_ids)
        # self.id_list = graph_ids
        # i = 0
        # num_nodes = num_nodes.astype(int)
        # num_edges = num_edges.astype(int)
        # prev_edge_ind = 0
        # prev_node_ind = 0

        # for i in range(len(graph_ids)):

        #     current_edge_ind = prev_edge_ind + int(num_edges[i])
        #     current_node_ind = prev_node_ind + int(num_nodes[i])
        #     selected_nodes = node_attributes[prev_node_ind:current_node_ind]
        #     selected_edges = edges_matrix[:, prev_edge_ind:current_edge_ind]
        
        #     #print("i = ", i, "nodes", selected_nodes.shape, "edges", selected_edges.shape)
        #     self.graph_data.append(GeoData(x=torch.transpose(torch.from_numpy(selected_nodes).view(1,-1),0,1).float(), edge_index=selected_edges))# y=labels_cat[i])#, phy_data=self.phy_data, aux_data=self.aux_data))
        #     prev_edge_ind = current_edge_ind
        #     prev_node_ind = current_node_ind

        
        # get number of samples

        # reshape phylogenetic state tensor
        # phy_data.shape = (num_sample, -1, self.num_data_col)
        # phy_data = np.transpose(phy_data, axes=[0,2,1]).astype('float32')
        # self.phy_data = phy_data

        # test dataset normalization
        assert aux_data.shape[0] == self.num_sample
        self.aux_data = util.normalize(aux_data, self.train_aux_data_mean_sd)
        #self.aux_data_names = aux_data_names
        
        # dataset index
        self.idx_data = idx_data

        # running against test sim?
        if mode == 'sim':
            # real vs. cat labels
            label_num_idx = list()
            label_cat_idx = list()
            aux_idx = list()
            for i,p in enumerate(label_names):
                if p in self.label_num_names:
                    label_num_idx.append(i)
                if p in self.label_cat_names:
                    label_cat_idx.append(i)
            
            assert labels.shape[0] == self.num_sample
            
            for i,p in enumerate(aux_data_names):
                if p in self.aux_names:
                    aux_idx.append(i)
            self.aux_names = []
            for i in aux_idx:
                self.aux_names.append(aux_data_names[i])


            self.true_labels_num = labels[:,label_num_idx]
            self.true_labels_cat = labels[:,label_cat_idx]

            self.true_aux = self.aux_data[:,aux_idx]
            
            # recode categorical labels
            for idx in range(self.true_labels_cat.shape[1]):
                unique_cats, encoded_cats = np.unique(self.true_labels_cat[:,idx],
                                                      return_inverse=True)
                self.true_labels_cat[:,idx] = encoded_cats                    
                # num_outliers = np.sum(np.abs(self.aux_data[:, i]) > bound)
                # if num_outliers > num_expected:
                # util.print_warn(f'Outlier detected in column {i} of aux_data')
            
        # done
        return

    def make_results(self, mode='sim'):
        """Makes all results for the Estimate step.

        This function loads a trained model from the Train stem, then uses it
        to perform the estimation task. For example, the step might estimate all
        model parameter values and adjusted lower and upper CPI bounds.

        Args:
            mode (str): 'sim' or 'emp' for simulated or empirical analysis.

        """

        # filesystem
        path_prefix = ''
        if mode == 'sim':
            path_prefix = f'{self.est_dir}/{self.est_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.test'
        if mode == 'emp':
            path_prefix = f'{self.est_dir}/{self.est_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.empirical'

        model_arch_fn = f'{self.trn_dir}/{self.trn_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.trained_model.pkl'
        out_est_labels_num_fn = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}_est.labels_num.csv'
        out_true_labels_num_fn = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}_true.labels_num.csv'
        out_est_labels_cat_fn = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}_est.labels_cat.csv'
        out_true_labels_cat_fn = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}_true.labels_cat.csv'
        out_true_aux_fn = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}_true.aux.csv'
        out_aux_names_fn = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.aux_names.csv'
        out_est_mle_labels_num_fn = f'{path_prefix}.{self.num_classes}.{self.optimizer}.{self.scheduler}.{self.phy_hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}_MLE_est.labels_num.csv'


    
        # load model
        self.mymodel = torch.load(model_arch_fn, map_location=self.TORCH_DEVICE, weights_only=False)
        # if self.use_cuda:
        #    self.mymodel = DataParallel((self.mymodel)) # model_arch_fn
        self.mymodel.to(self.TORCH_DEVICE)

        # get estimates
        loader = DataLoader(self.estimate_dataset, batch_size = self.num_sample)
        for loaded_graph, loaded_idx, loaded_lbl_num, loaded_lbl_cat in loader:
            loaded_graph = loaded_graph.to(self.TORCH_DEVICE)
            # loaded_batch = loaded_batch.to("cuda")
            label_est = self.mymodel(loaded_graph)#.to(torch.device("cuda"))
        
        print("LOADED GRAPH")
        print(loaded_graph)
        # real vs. cat estimates
        labels_est_num = label_est[0:3]
        labels_est_cat = label_est[3]
        amax = labels_est_cat.argmax(dim=1)

        # # force categorical dimensionality (had problems for categ)
        # for k,v in labels_est_cat.items():
        #     labels_est_cat[k] = torch.reshape(input=labels_est_cat[k],
        #                                       shape=(self.graph_data.shape[0],-1))

        # point estimates & CPIs for test labels
        if self.has_label_num:
            
            # move Tensor from device to numpy
            labels_est_num = torch.stack(labels_est_num).cpu().detach().numpy()
            
            if labels_est_num.ndim == 2:
                labels_est_num.shape = (labels_est_num.shape[0], 1, labels_est_num.shape[1])
            labels_est_num[1,:,:] = labels_est_num[1,:,:] + self.cpi_adjustments[0,:]
            labels_est_num[2,:,:] = labels_est_num[2,:,:] + self.cpi_adjustments[1,:]
            
            # denormalize test label estimates
            denorm_est_labels_num = util.denormalize(labels_est_num,
                                                      self.train_labels_num_mean_sd,
                                                      exp=False)

            # save label real estimates
            df_est_labels_num = util.make_param_VLU_mtx(denorm_est_labels_num,
                                                         self.label_num_names)
            df_est_labels_num = pd.concat( [self.idx_data, df_est_labels_num], axis=1 )
            df_est_labels_num.to_csv(out_est_mle_labels_num_fn, index=False, sep=',',
                                      float_format=util.PANDAS_FLOAT_FMT_STR)

        self.idx_data = pd.DataFrame(self.idx_data)                              
        
        # save label cat estimates
        if self.has_label_cat:
            df_est_labels_cat = pd.DataFrame((labels_est_cat.cpu().detach().argmax(dim=1).flatten()))
            #self.format_label_cat(labels_est_cat)
            df_est_labels_cat = pd.concat( [self.idx_data, df_est_labels_cat], axis=1 )
            df_est_labels_cat.to_csv(out_est_labels_cat_fn, index=False, sep=',',
                                     float_format=util.PANDAS_FLOAT_FMT_STR)
            
            # for k,v in labels_est_cat.items():
            #     labels_est_cat[k] = labels_est_cat[k].cpu().detach().numpy()
        
        if mode == 'sim':
            if self.has_label_num:
                df_true_labels_num = pd.DataFrame(self.true_labels_num, columns=self.label_num_names)
                df_true_labels_num = pd.concat( [self.idx_data, df_true_labels_num], axis=1 )
                df_true_labels_num.to_csv(out_true_labels_num_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
            
            if self.has_label_cat:
                df_true_labels_cat = pd.DataFrame(self.true_labels_cat, columns=self.label_cat_names, dtype='int')
                df_true_labels_cat = pd.concat( [self.idx_data, df_true_labels_cat], axis=1)
                df_true_labels_cat.to_csv(out_true_labels_cat_fn, index=False, sep=',')
            if self.has_aux:
                df_true_aux = pd.DataFrame(self.true_aux, dtype='float')
                df_true_aux = pd.concat( [self.idx_data, df_true_aux], axis=1)
                
                df_true_aux.columns = ["idx"] + self.aux_names
                df_true_aux.to_csv(out_true_aux_fn, index=False, sep=',')
        
        if mode == 'emp':
            # self.est_labels_num_raw = denorm_est_labels_num
            self.est_true_aux_raw = util.denormalize(self.aux_data,
                                                     self.train_aux_data_mean_sd,
                                                     exp=False)
            if self.has_label_num:
                self.est_labels_num_raw = util.denormalize(labels_est_num,
                                                          self.train_labels_num_mean_sd,
                                                          exp=False)[0,:,:]
        
        # done
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
    
    
    def check_empirical_results(self):

        # check for outliers in aux_data
        aux_data = util.normalize(self.est_aux_data_raw, self.train_aux_data_mean_sd)
        aux_std_bound = np.round(sp.stats.norm.ppf(1.0 - self.warn_aux_outlier/2, loc=0, scale=1), 2)
        for i in range(aux_data.shape[1]):
            outlier_fail = np.abs(aux_data[:, i]) > aux_std_bound
            mu = self.train_aux_data_mean_sd[0][i]
            sd = self.train_aux_data_mean_sd[1][i]
            raw_lower = "{:.2e}".format(mu - aux_std_bound * sd)
            raw_upper = "{:.2e}".format(mu + aux_std_bound * sd)
            percent = 100*(1.0 - self.warn_aux_outlier)
            outlier_idx = np.where(outlier_fail)[0]
            outliers = self.est_aux_data_raw[outlier_fail, i]
            if outliers.shape[0] > 0:
                util.print_warn(f'Outlier(s) detected in empirical aux. data: {self.aux_data_names[i]}')
                # util.print_str(f'           Values outside {(1.0 - self.warn_aux_outlier)*100}% interval of [{raw_lower}, {raw_upper}]')
                util.print_str(f'         - Values outside ± {aux_std_bound}sd ({percent}%) interval of [{raw_lower}, {raw_upper}]')
                util.print_str(f'         - Detected outlier(s):')
                for j in range(outliers.shape[0]):
                    util.print_str(f'             index {outlier_idx[j]} : value {outliers[j]}')
        
        # check for outliers in labels
        if self.has_label_num:
            est_labels_num = util.normalize(self.est_labels_num_raw, self.train_labels_num_mean_sd)
            lbl_std_bound = np.round(sp.stats.norm.ppf(1.0 - self.warn_lbl_outlier/2, loc=0, scale=1), 2)
            for i in range(est_labels_num.shape[1]):
                outlier_fail = np.abs(est_labels_num[:, i]) > lbl_std_bound
                mu = self.train_labels_num_mean_sd[0][i]
                sd = self.train_labels_num_mean_sd[1][i]
                raw_lower = "{:.2e}".format(mu - lbl_std_bound * sd)
                raw_upper = "{:.2e}".format(mu + lbl_std_bound * sd)
                percent = 100*(1.0 - self.warn_lbl_outlier)
                outlier_idx = np.where(outlier_fail)[0]
                outliers = self.est_labels_num_raw[outlier_fail, i]
                if outliers.shape[0] > 0:
                    util.print_warn(f'Outlier(s) detected in empirical labels: {self.label_num_names[i]}')
                    # util.print_str(f'           Values outside {(1.0 - self.warn_lbl_outlier*100)}% interval of [{raw_lower}, {raw_upper}]')
                    util.print_str(f'         - Values outside ± {lbl_std_bound}sd ({percent}%) interval of [{raw_lower}, {raw_upper}]')
                    util.print_str(f'         - Detected outlier(s):')
                    for j in range(outliers.shape[0]):
                        util.print_str(f'             index {outlier_idx[j]} : value {outliers[j]}')
        
##################################################
