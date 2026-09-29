# Binary classification of birth-death and binary state speciation and extinction models 

This repository contains code used to differentiate between phylogenies simulated under either a constant birth-death (BD) (Feller 1939), or binary state speciation and extinction (BiSSE) model (Maddison et al. 2007).  Code has been written by hand and using AI tools, such as ChatGPT and Claude, with human supervision.

We use several different methods for our binary classification task, include graph neural networks. Code related to using graph neural networks is in the `src/phyddle` folder, while additional files for data simulation, result processing and other methods such as AIC (Akaike 1973) are in the `workspace/pj_phyddle` folder.


## Training graph neural networks
To train graph neural networks, we edit source code from  (Thompson et al. 2024, Landis & Thompson 2026). To use our version, clone the [Phyddle repository](https://github.com/mlandis/phyddle), and replace the `src/phyddle` folder with the one available here. Our `workplace/pj_phyddle` folder can be added to the default Phyddle workspace. We recommend familiarising yourself with the standard Phyddle pipeline before using this repository, particularly the different steps of the pipeline, including simulation, data formatting, network training and estimation.

To convert phylogenies to graphs, we use helper functions from the [phylo-inference-ml repository](https://github.com/ismael-lajaaiti/phylo-inference-ml) (see Lajaaiti 2023).

In addition to adapting code for simulating and formatting data, plus training the network and making estimates, we also include the `get_graph_embeddings.py` file in the source folder to extract graph embeddings by applying trained networks up to the final graph convolutional layer to input data.

We primarily rely on our own plotting code to interpret analyses, rather than `plot.py` in the Phyddle source folder, but retain it for convenience of initial analyses.

## Data simulation
Our dataset consists of thirty thousand phylogenies, half of which are generated under a BD model, and half of which are generated under a BiSSE model. We use PhyloJunction (Mendes & Landis 2024) to generate the phylogenies. [PhyloJunction repository](https://github.com/mendeslab/PhyloJunction)

Files to simulate the phylogenies are in the `data_simulation` subfolder of `workspace/pj_phyddle`. `sim_BD.py` generates indiviudual phylogenies under the BD model using PhyloJunction, while `config_BD.py` uses Phyddle to repeat this process for a large number of samples. We can change the indices specified in `config.BD.py` to specify the ID given to each sample, which allows us to gradually add to the dataset over time. Similarly, `sim_BiSSE.py` is used to generate a phylogeny under the BiSSE model, while `config_BiSSE.py` uses this to produce many samples. The config files use our version of the simulation step from Phyddle, which can be found in the `src/phyddle` folder. 

To process the phylogenies generated using either a BD or BiSSE model together, we move the outputs from the phyddle simulation step to be in the same folder. We give an example config file for processing outputs from both models stored in one folder as `workspace/pj_phyddle/training.example.py`. To run steps other than training, the step argument can be changed, such as using 'F' to conduct formatting. The config file controls parameters such as the size of hidden layers in the graph neural network and the number of epochs for training. 

## Data formatting
We use the formatting step of our edited Phyddle code to convert simulated phylogenies into graphs, calculate node features and store the resulting information in HDF5 files. We process phylogenies in blocks to avoid running out of memory. We specify how many blocks to allocate to the validation set, and create a HDF5 file for the training (and validation) data, and a separate HDF5 file for the testing data. To avoid accidentally overwriting HDF5 files, we include the substring "new" in the filenames, and must remove this in order to use the data for training.

## Graph neural network training
The training ('T') step of the Phyddle pipeline is used to train a graph neural network. Parameters such as the number of hidden layers and the number of graph convolutional layers can be specified in the config file. We can choose between two network architectures from Leroy et al. 2025 which use either average-pooling or the PhyloPool procedure.
We use two different datasets, one with 30,000 phylogenies, which is our large dataset, and a subset of this with 3,000 phylogenies which is our small dataset. We use 85% of data for training, 15% for validation and 5% for testing.

After initial testing, we train six different networks. Our trained graph neural networks are available in the `trained_model` folder:
- A network trained using the **small** dataset using **average** pooling with **six** graph convolutional layers in total (four of which have the same input and output dimensions),  `out.2.2850.75.adam.manual.50.True.False.False.4.0.0005.0.01.relu.NA.trained_model.pkl`
- A network trained using the **small** dataset using **average** pooling with **twelve** graph convolutional layers in total (ten of which have the same input and output dimensions),  `out.2.2850.75.adam.manual.50.True.False.True.10.0.0005.0.01.relu.NA.trained_model.pkl`
- A network trained using the **small** dataset using **PhyloPool** with **three** graph convolutional layers in total,  `out.2.2850.150.adam.manual.8.True.True.False.3.0.0005.0.01.relu.NA.trained_model.pkl`
- - A network trained using the **large** dataset using **average** pooling with **six** graph convolutional layers in total (four of which have the same input and output dimensions),  `out.2.28500.75.adam.manual.50.True.False.False.4.0.0005.0.01.relu.NA.trained_model.pkl`
- A network trained using the **large** dataset using **average** pooling with **twelve** graph convolutional layers in total (ten of which have the same input and output dimensions),  `out.2.28500.75.adam.manual.50.True.False.True.10.0.0005.0.01.relu.NA.trained_model.pkl`
- A network trained using the **large** dataset using **PhyloPool** with **three** graph convolutional layers in total, `out.2.28500.150.adam.manual.8.True.True.False.3.0.0005.0.01.relu.NA.trained_model.pkl`

## Estimation
The estimation ('E') step is used to obtain predictions from the trained network.

# References
> Akaike H. (1973). “Information Theory and an Extension of the Maximum Likelihood Principle”. In: Proc. 2nd Int. Symp. Inf. Theory. Ed. by B. N. Petrov, F. Csaki. Budapest:Akademiai Kiado, pp. 267–281. <br>

> Feller W. (1939). “Die Grundlagen der Volterraschen Theorie des Kampfes ums Dasein in wahrscheinlichkeitstheoretischer Behandlung”. In: Acta Biotheor. 5.1, pp. 11–40. <br>

> Maddison, W. P., Midford, P. E., & Otto, S. P. (2007). Estimating a binary character's effect on speciation and extinction. Systematic Biology, 56(5), 701-710. <br>

>Mendes F. K., Landis M. J. (2024). “PhyloJunction: A computational framework for simulating, developing, and teaching evolutionary models”. In: Syst. Biol. 73.6, pp. 1051–1060. <br>

>Neyman J., Pearson E. S. (Feb. 1933). “IX. On the problem of the most efficient tests of statistical hypotheses”. In: Philos. Trans. R. Soc. Lond. A 231.694-706, pp. 289–337. <br>

>Lajaaiti I., Lambert S., Voznica J., Morlon H., Hartig F. (2023). "A Comparison of Deep
Learning Architectures for Inferring Parameters of Diversification Models from Extant
Phylogenies." <br>

> Landis, M.J., Thompson, A. 2025. phyddle: software for exploring phylogenetic models with deep learning. Systematic Biology (in press). doi:10.1093/sysbio/syaf036.<br>

> Leroy A., Lajaaiti I., Lambert S., Voznica J., Pichler M., Morlon H., Hartig F., Jacob L. (Aug.
2025). Graph neural networks for likelihood-free inference in diversification models. <br>

> Thompson, A., Liebeskind, B., Scully, E.J., Landis, M.J.. 2024. Deep learning and likelihood approaches for viral phylogeography converge on the same answers whether the inference model is right or wrong. Systematic Biology 73:183-206. 
using Phyddle () <br>

> Wilks S. S. (1938). “The large-sample distribution of the likelihood ratio for testing composite hypotheses”. In: Ann. Math. Stat. 9.1, pp. 60–62. <br>

