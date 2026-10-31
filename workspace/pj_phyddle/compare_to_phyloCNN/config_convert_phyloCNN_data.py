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
    'step'               : 'F',               # step(s) to run
    'verbose'            : 'T',                   # print verbose phyddle output?
    'dir'                : '/mnt/DiversificationGraphInference/phyloCNN/Simulations/',                  # base directory for step directories
    'prefix'             : 'out',                 # base prefix for step output
    'output_precision'   : 12,                    # Number of digits (precision) for numbers in output files

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'T',                 # use multiprocessing to speed up jobs?
    'use_cuda'       : 'T',                 # use CUDA for training?
    'num_proc'       : -10,                  # how many CPUs to use (-2 means all but 2)

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'encode_all_sim'    : 'T',
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'min_num_taxa'      : 10,               # min number of taxa for valid sim
    'max_num_taxa'      : 500,              # max number of taxa for valid sim
    'tree_width'        : 500,              # tree width category used to train network
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
    'param_est'         : {                 # model parameters to predict (labels)
                           'log10_birth_rate':'num', 
                           'log10_death_rate':'num',
                           'modeltype':'cat',
                          },
    'param_data'        : {                 # model parameters that are known (aux. data)
                          },
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'char_format'       : 'csv',
    'save_phyenc_csv'   : 'T',              # save intermediate phylo-state vectors to file

    'packed_in_dir':   '/mnt/DiversificationGraphInference/phyloCNN/generate_parameters_and_trees/',
    # 'packed_par_dir':  '/mnt/DiversificationGraphInference/phyloCNN/Simulations/parameters',
    'packed_out_dir':  '/mnt/DiversificationGraphInference/phyloCNN/generate_parameters_and_trees/simulate',
    # 'packed_models':       {'BD_div': 0, 'BiSSE': 1},
    # 'packed_tre_fn_train':       'train.nwk',
    # 'packed_par_fn_train':        'train.csv',
    # 'packed_tre_fn_test':       'test.nwk',
    # 'packed_par_fn_test':        'test.csv',
    # 'packed_state_key':    't_s',
    # 'packed_state_offset': 1,
    'fixed_split': True,
    'encoding_type': "graph",

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    'num_epochs'        : 50,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset

    'trn_batch_size'    : 4096,             # number of samples in each training batch
    'loss_numerical'    : 'mse',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters

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
