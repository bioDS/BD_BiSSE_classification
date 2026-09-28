# Binary classification of birth-death and binary state speciation and extinction models 

This repository contains code used to differentiate between phylogenies simulated under either a constant birth-death (BD) (Feller 1939), or binary state speciation and extinction (BiSSE) model (Maddison et al. 2007).  Code has been written by hand and using AI tools, such as ChatGPT and Claude, with human supervision.

We use several different methods for our binary classification task, include graph neural networks. Code related to using graph neural networks is in the `src/phyddle` folder, while additional files for data simulation, result processing and other methods such as AIC (Akaike 1973) are in the `workspace/pj_phyddle` folder.


## Training graph neural networks
To train graph neural networks, we edit source code from  (Thompson et al. 2024, Landis & Thompson 2026). To use our version, clone the [Phyddle repository](https://github.com/mlandis/phyddle), and replace the `src/phyddle` folder with the one available here. We recommend familiarising yourself with the standard Phyddle pipeline before using this repository.

In addition to adapting code for simulating and formatting data, plus training the network and making estimates, we also include a 

## Data simulation
Our dataset consists of thirty thousand phylogenies, half of which are generated under a BD model, and half of which are generated under a BiSSE model. We use PhyloJunction (Mendes & Landis 2024) to generate the phylogenies. [PhyloJunction repository](https://github.com/mendeslab/PhyloJunction)

W

# References
> Akaike H. (1973). “Information Theory and an Extension of the Maximum Likelihood Principle”. In: Proc. 2nd Int. Symp. Inf. Theory. Ed. by B. N. Petrov, F. Csaki. Budapest:Akademiai Kiado, pp. 267–281. <br>

> Feller W. (1939). “Die Grundlagen der Volterraschen Theorie des Kampfes ums Dasein in wahrscheinlichkeitstheoretischer Behandlung”. In: Acta Biotheor. 5.1, pp. 11–40. <br>

> Maddison, W. P., Midford, P. E., & Otto, S. P. (2007). Estimating a binary character's effect on speciation and extinction. Systematic Biology, 56(5), 701-710. <br>

>Mendes F. K., Landis M. J. (2024). “PhyloJunction: A computational framework for simulating, developing, and teaching evolutionary models”. In: Syst. Biol. 73.6, pp. 1051–1060. <br>

>Neyman J., Pearson E. S. (Feb. 1933). “IX. On the problem of the most efficient tests of statistical hypotheses”. In: Philos. Trans. R. Soc. Lond. A 231.694-706, pp. 289–337. <br>

> Landis, M.J., Thompson, A. 2025. phyddle: software for exploring phylogenetic models with deep learning. Systematic Biology (in press). doi:10.1093/sysbio/syaf036.<br>

> Thompson, A., Liebeskind, B., Scully, E.J., Landis, M.J.. 2024. Deep learning and likelihood approaches for viral phylogeography converge on the same answers whether the inference model is right or wrong. Systematic Biology 73:183-206. 
using Phyddle () <br>

> Wilks S. S. (1938). “The large-sample distribution of the likelihood ratio for testing composite hypotheses”. In: Ann. Math. Stat. 9.1, pp. 60–62. <br>

