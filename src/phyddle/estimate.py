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
import threading
import time
import math
sys.stdout.reconfigure(line_buffering=True)

# external imports
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import torch
import subprocess, pickle

from torch_geometric.data import Dataset as Geoset, Data as GeoData, Batch
from torch_geometric.loader import DataLoader as GeoLoader, DataListLoader
from torch.utils.data import DataLoader
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from phyddle import train
# from train import HDF5BlockDataset
import torch.distributed as dist
from torch import linalg as LA

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
r = robjects.r
# r['source']('~/AIphylo/phyddle/workspace/pj_phyddle/MLE/mle.R')
# get_mle = r['get_mle_label']




# phyddle imports
from phyddle import utilities as util
from phyddle import network


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
        self.args = args
        
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

        self.regularisation = str(args['regularisation'])
        self.block_size = int(args['block_size'])


        
        # error checking
        self.warn_aux_outlier   = float(args['warn_aux_outlier'])
        self.warn_lbl_outlier   = float(args['warn_lbl_outlier'])
        self.est_aux_data_raw   = None
        self.est_labels_num_raw = None

        self.phylo_pool         =bool(args['phylo_pool'])
        self.graph_conv         =bool(args['graph_conv'])
        self.hidden_size = int(args['phy_hidden_size'])
        self.optimizer          = str(args['optimizer'])
        self.dropout = float(args['dropout'])
        self.dataset_size = int(args['dataset_size'])
        self.batch_size = int(args['accumulation_steps']) *  int(args['trn_batch_size'])
        self.activation_func    = str(args['activation_func'])
        self.extra_layers       =bool(args['extra_layers'])
        self.phylo_n_layers     =int(args['n_layers_phylo'])
        self.avg_n_layers       =int(args['n_layers_avg'])
        self.n_layers = self.avg_n_layers
        if self.phylo_pool:
            self.n_layers = self.phylo_n_layers


        self.num_classes        =  int(args['num_classes'])
        self.scheduler="manual"

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
        self.trainer = None
        self.trn_batch_size     = int(args['trn_batch_size'])


        self.num_sample = -1

        
        # done
        return

    def tree_generator(self, file_names):
        for f in file_names:
            with open(f, "r") as open_f:
                yield open_f.read()

    def MLE(self, idx):
        print("idx data:")
        print(idx.iloc[:, 0])
        file_names = [f'{self.sim_dir}/{self.fmt_prefix}.{i}.tre' for i in idx.iloc[:, 0]]
        phylogenies = []
        for f in file_names:
            with open(f, "r") as open_file:
                phylogenies.append(open_file.read())

        mle_labels = []
        with ThreadPoolExecutor(max_workers=80) as executor:
            futures = {executor.submit(self.single_mle, phy): i for i, phy in enumerate(phylogenies)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating MLE", ncols=100):
                mle_labels.append(future.result())
        print("mle labels:")
        print(mle_labels)  
        mle_labels = np.array(mle_labels, dtype=float)
        print(mle_labels)
        return mle_labels


    def single_mle(self, phy):
        p = subprocess.Popen(
        ["python", "parallel_mle.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        )

        def watchdog():
            time.sleep(300)
            if p.poll() is None:
                print("time out")
                p.kill()
                return -1

        timer = threading.Thread(target=watchdog, daemon=True)
        timer.start()
        try:
            pickle.dump(phy, p.stdin)
            p.stdin.close()
            try:
                mle_label = pickle.load(p.stdout)
                # print(mle_label)
            except Exception:
                print("time out")
                return -1
        finally:
            p.stdout.close()
            p.stderr.close()
            p.wait()
        # if (p_val < 0.1):
        #     return 1
        # else:
        #     return 0
        return float(mle_label[0])
    
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
            # self.load_format_input(mode='sim')
    
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

        path_prefix = f'{self.trn_dir}/{self.trn_prefix}.{self.num_classes}.{self.dataset_size}.{self.batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.dropout}.{self.activation_func}.{self.learning_rate}.{self.regularisation}'
        out_est_mle_labels_cat_fn = f'{path_prefix}_MLE_est.labels_cat.csv'
        out_est_mle_vals_cat_fn = f'{path_prefix}_MLE_est.p_vals_cat.csv'

        if self.calc_MLE:
            self.mle_labels = self.MLE(self.idx_data)

            print(out_est_mle_labels_cat_fn)
            pd.DataFrame(self.mle_labels).to_csv(out_est_mle_vals_cat_fn, index=False, sep=',',
                                      float_format=util.PANDAS_FLOAT_FMT_STR)
            self.mle_labels =  np.where(self.mle_labels != -1, self.mle_labels < 0.1, -1).astype(int)
            print("self.mle_labels")
            print(self.mle_labels)
            pd.DataFrame(self.mle_labels).to_csv(out_est_mle_labels_cat_fn, index=False, sep=',',
                                      float_format=util.PANDAS_FLOAT_FMT_STR)




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
            print("k:", k, "v", v)
            if v == 'cat':
                self.has_label_cat = True
                idx = self.label_names.index(k)
                unique_cats, encoded_cats = np.unique(labels[:,idx],
                                                      return_inverse=True)
                self.param_cat[k] = len(unique_cats)
                labels[:,idx] = encoded_cats
                idx_cat.append( idx )
                self.param_cat_names.append(k)

        
        if not self.has_label_num and not self.has_label_cat:
            util.print_err(f"No training labels found.", exit=True)
               
        # get data subsets
        labels_num = labels[:,idx_num].copy()
        labels_cat = labels[:,idx_cat].copy()
        print("returning from sep labels")
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
        path_prefix = f'{self.trn_dir}/{self.trn_prefix}.{self.num_classes}.{self.dataset_size}.{self.batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.extra_layers}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}'
        train_norm_aux_data_fn = f'{path_prefix}.trn_norm.aux_data.csv'
        train_norm_labels_num_fn = f'{path_prefix}.trn_norm.labels_num.csv'

        # model_cpi_fn = f'{path_prefix}.cpi_adjustments.csv'

        # denormalization factors for new aux data
        train_aux_data_norm = pd.read_csv(train_norm_aux_data_fn, sep=',', index_col=False)
        train_aux_data_means = train_aux_data_norm['mean'].T.to_numpy().flatten()
        train_aux_data_sd = train_aux_data_norm['sd'].T.to_numpy().flatten()
        self.train_aux_data_mean_sd = (train_aux_data_means, train_aux_data_sd)
        
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
            path_prefix = f'{self.est_dir}/{self.est_prefix}.{self.num_classes}.{self.dataset_size}.{self.batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.extra_layers}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.test'
            short_path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.test'
            training_prefix = f'{self.trn_dir}/{self.trn_prefix}.{self.num_classes}.{self.dataset_size}.{self.batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.extra_layers}.{self.n_layers}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}'

        if mode == 'emp':
            path_prefix = f'{self.est_dir}/{self.est_prefix}.{self.num_classes}.{self.dataset_size}.{self.batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}.empirical'
            short_path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.empirical'
            training_prefix =  f'{self.trn_dir}/{self.est_prefix}.{self.num_classes}.{self.dataset_size}.{self.batch_size}.{self.optimizer}.{self.scheduler}.{self.hidden_size}.{self.graph_conv}.{self.phylo_pool}.{self.learning_rate}.{self.dropout}.{self.activation_func}.{self.regularisation}'


        model_arch_fn = f'{training_prefix}.trained_model.pkl'
        out_est_labels_num_fn = f'{path_prefix}_est.labels_num.csv'
        out_true_labels_num_fn = f'{path_prefix}_true.labels_num.csv'
        out_est_labels_cat_fn = f'{path_prefix}_est.labels_cat.csv'
        out_true_labels_cat_fn = f'{path_prefix}_true.labels_cat.csv'
        out_true_aux_fn = f'{path_prefix}_true.aux.csv'
        out_aux_names_fn = f'{path_prefix}_aux_names.csv'

        # simulated test datasets for csv or hdf5
        hdf5_fn = f'{short_path_prefix}.hdf5'
        train_labels_num_norm_fn    = f'{training_prefix}.trn_norm.labels_num.csv'
        train_aux_data_norm_fn       = f'{training_prefix}.trn_norm.aux_data.csv'
        train_attr_data_norm_fn       = f'{training_prefix}.trn_norm.attr_data.csv'

        aux_norms = pd.read_csv(train_aux_data_norm_fn,sep=',')
        attr_norms = pd.read_csv(train_attr_data_norm_fn,sep=',')

        self.aux_msd = (torch.from_numpy(aux_norms.iloc[:, 0].to_numpy()),torch.from_numpy(aux_norms.iloc[:, 1].to_numpy()))
        self.attr_msd = (torch.from_numpy(attr_norms.iloc[:, 0].to_numpy()),torch.from_numpy(attr_norms.iloc[:, 1].to_numpy()))

    
        print("LOADING", model_arch_fn)
        # load model
        self.mymodel = torch.load(model_arch_fn, map_location=self.TORCH_DEVICE, weights_only=False)
        if self.use_cuda:
            self.mymodel = DataParallel(self.mymodel)
        self.mymodel.to(self.TORCH_DEVICE)

        with h5py.File(hdf5_fn, "r") as f:
            N = f["phy_data"].shape[0]
            self.label_names = [s.decode() for s in f['label_names'][0,:] ]
            aux_data_names = [ s.decode() for s in f['aux_data_names'][0,:] ]

            blocks = [np.arange(N)[i:i+self.block_size] for i in range(0,N,self.block_size)]
            self.trainer = train.CnnTrainer(self.args)
            self.estimate_dataset = train.HDF5BlockDataset(hdf5_fn, blocks, self.trainer)

        # get estimates
        self.total_batches = len(self.estimate_dataset)*(math.ceil(self.block_size / self.trn_batch_size))
        pbar = tqdm(total=self.total_batches)

        print("Trainer type:", type(self.trainer))
        print("Method being used:", self.trainer.separate_labels)

        
        all_idx = []
        print("out for estimated nums:, ", out_est_labels_num_fn)
        print("out for estimated cats:, ", out_est_labels_cat_fn)
        print("mode:", mode)

        for j in range(len(self.estimate_dataset)):
            graph_list = self.estimate_dataset[j]
            loader = DataListLoader(graph_list, batch_size=self.trn_batch_size)
            # print("mean, sd:", self.train_labels_num_mean_sd)
            first_batch = True
            for batch in loader:
                lbl_cat = torch.cat([g.lbl_cat for g in batch], dim=0).to(self.TORCH_DEVICE)
                lbl_num = torch.stack([g.lbl_num for g in batch], dim=0).to(self.TORCH_DEVICE)
                batch_idx = np.array([g.idx for g in batch])
                all_idx.append(batch_idx)
                aux_dat = torch.stack([g.aux_dat for g in batch], dim=0).to(self.TORCH_DEVICE)
                for g in batch:
                    old = g.x
                    g.x = util.normalize(g.x, self.attr_msd).clone().detach().float()
                    g.aux_dat = util.normalize(g.aux_dat, self.aux_msd).float().clone().detach().float()
                try:
                    label_est = self.mymodel(batch)
                    pbar.update(1)
                except Exception as e:
                    print("exception in lbls_est", e)
                    raise

                # real vs. cat estimates
                labels_est_num = label_est[3]
                # print("labels_est_num", labels_est_num)
                # print("labels_est_num:", labels_est_num)
                # amax = labels_est_cat.argmax(dim=1)

                # # force categorical dimensionality (had problems for categ)
                # for k,v in labels_est_cat.items():
                #     labels_est_cat[k] = torch.reshape(input=labels_est_cat[k],
                #                                       shape=(self.graph_data.shape[0],-1))

                csv_mode = "a"
                header = False
                if j == 0 and first_batch:
                    csv_mode = "w"
                    first_batch = False
                    # header = True
                # save label cat estimates
                # print("estimates", labels_est_num)
                df_est_labels_cat = pd.DataFrame((labels_est_num.cpu().detach().argmax(dim=1).flatten()))
                #self.format_label_cat(labels_est_cat)
                df_est_labels_cat = pd.concat( [pd.DataFrame(batch_idx), df_est_labels_cat], axis=1 )
                # print("writing estimates")
                df_est_labels_cat.to_csv(out_est_labels_cat_fn, index=False, sep=',',
                                        float_format=util.PANDAS_FLOAT_FMT_STR, header=header, mode=csv_mode)


                    
                    # for k,v in labels_est_cat.items():
                    #     labels_est_cat[k] = labels_est_cat[k].cpu().detach().numpy()
                
                if mode == 'sim':
                    df_true_labels_cat = pd.DataFrame(lbl_cat.cpu().detach(), columns=self.label_cat_names, dtype='int')
                    df_true_labels_cat = pd.concat( [pd.DataFrame(batch_idx), df_true_labels_cat], axis=1)
                    df_true_labels_cat.to_csv(out_true_labels_cat_fn, header=header, index=False, sep=',', mode=csv_mode)
                if self.has_aux:
                        df_true_aux = pd.DataFrame(aux_dat.cpu().detach().squeeze(1), dtype='float')
                        df_true_aux = pd.concat( [pd.DataFrame(batch_idx), df_true_aux], axis=1)
                        
                        # df_true_aux.columns = ["idx"] + self.aux_names
                        df_true_aux.to_csv(out_true_aux_fn, index=False, sep=',', header=header, mode=csv_mode)
                
                if mode == 'emp':
                    # self.est_labels_num_raw = denorm_est_labels_num
                    self.est_true_aux_raw = aux_dat #util.denormalize(aux_dat,
                                                #         self.train_aux_data_mean_sd,
                                                #         exp=False)
                   
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
