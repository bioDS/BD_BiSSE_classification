#==============================================================================#
# Config:       Default phyddle config file                                    #
# Authors:      Michael Landis and Ammon Thompson                              #
# Date:         230804                                                         #
# Description:  Simple birth-death and equal-rates CTMC model in R using ape   #
#==============================================================================#


args = {

    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'step'               : 'T', #EP               # step(s) to run
    'verbose'              : 'T',                   # print verbose phyddle output?
    'dir'                : '/mnt/DiversificationGraphInference/thirty_thousand_trees/',    # base directory for step directories
    'prefix'             : 'out',                 # base prefix for step output
    'output_precision'   : 12,                    # Number of digits (precision) for numbers in output files

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'T',                 # use multiprocessing to speed up jobs?
    'use_cuda'       : 'T',                 # use CUDA for training?
    'num_proc'       : -10,                  # how many CPUs to use (-2 means all but 2)

    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : f'python3 sim_BD.py BD.pj trs', # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 1,                # first simulation replicate index
    'end_idx'           : 10000,            # last simulation replicate index
    'sim_batch_size'    : 20,

    'num_classes': 2,
    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'encode_all_sim'    : 'T',
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'min_num_taxa'      : 100,               # min number of taxa for valid sim
    'max_num_taxa'      : 1000,              # max number of taxa for valid sim
    'tree_width'        : 50000,              # tree width category used to train network
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
    'param_est'         : {                 # model parameters to predict (labels)
                          'log10_birth_rate':'num', 
                          'log10_death_rate':'num',
                          'log10_birth_rate_t1':'num',
                          'log10_birth_rate_t2':'num',
                          'log10_death_rate_t1':'num',
                          'log10_death_rate_t2':'num',
                          'model_type':'cat',
                          },
    'param_data'        : {                 # model parameters that are known (aux. data)
                          },
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'char_format'       : 'csv',
    'save_phyenc_csv'   : 'T',     # save intermediate phylo-state vectors to file
    'save_graph_csv' : 'T',      # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    'load_model'   : 'F', # whether to use already trained model 
    'phylo_pool' : 'F', # if True, use GNN-PhyloPool architecture, else use GNN-AvgPool (architectures from Leroy et al. 2025)
    'graph_conv' : 'T', # Which type of graph convolutional layer to use
    'num_epochs'        : 100,               # number of training intervals (epochs)
    'n_val_blocks'          : 3,              # number of blocks to use in validation dataset
    'trn_batch_size'    : 16,            # number of samples in each training batch
    'loss_numerical'    : 'mse',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'activation_func'   :'relu',
    'phy_hidden_size'   : 50,            # parameter controlling number of neurons in hidden layers
    'dropout' : 0.01,                    
    'learning_rate'      :0.0005, 
    'num_early_stop'    : 5,            # Number of epochs of consecutive validation loss increasing that we stop training for
    'regularisation'    : 'NA',    # No, L1, L2 or both L1 and L2 regularisation
    'regression': False,            # We are performing binary classification rather than regression - we could alter code to make this parameter unnecessary.
    'block_size': 1500,            # Number of observations in each block
    'accumulation_steps': 4,        # Number of batches before we update the optimizer, effective batch size is this parameter times trn_batch_size
    'dataset_size': 28500,            # Number of observations in the training + validation dataset.


    #-------------------------------#
    # Estimate Step settings        #
    #-------------------------------#

    #-------------------------------#
    # Plot Step settings            #
    #-------------------------------#
    'plot_train_color'      : 'blue',       # plot color for training data
    'plot_test_color'       : 'purple',     # plot color for test data
    'plot_val_color'        : 'red',        # plot color for validation data
    'plot_aux_color'        : 'green',      # plot color for input auxiliary data
    'plot_label_color'      : 'orange',     # plot color for labels (params)
    'plot_emp_color'        : 'black'       # plot color for estimated data/values

}
