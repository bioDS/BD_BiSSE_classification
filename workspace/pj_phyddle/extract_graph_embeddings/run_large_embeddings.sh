#!/bin/bash

source ~/mambaforge/etc/profile.d/conda.sh

mamba activate AIphylo

python pca_and_XGB_for_bash.py -d large -p out.2.28500.75.adam.manual.50.True.False.False.4.0.0005.0.01.relu.NA.graph_embeddings
python pca_and_XGB_for_bash.py -d large -p out.2.28500.75.adam.manual.50.True.False.True.10.0.0005.0.01.relu.NA.graph_embeddings
python pca_and_XGB_for_bash.py -d large -p out.2.28500.150.adam.manual.8.True.True.False.3.0.0005.0.01.relu.NA.graph_embeddings
python pca_and_XGB_for_bash.py -d small -p out.2.2850.75.adam.manual.50.True.False.False.4.0.0005.0.01.relu.NA.graph_embeddings
python pca_and_XGB_for_bash.py -d small -p out.2.2850.75.adam.manual.50.True.False.True.10.0.0005.0.01.relu.NA.graph_embeddings
python pca_and_XGB_for_bash.py -d small -p out.2.2850.150.adam.manual.8.True.True.False.3.0.0005.0.01.relu.NA.graph_embeddings

