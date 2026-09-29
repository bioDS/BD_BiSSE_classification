# Code to apply PCA and LDA to graph embeddings.

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from scipy.stats import randint
import joblib
from sklearn.decomposition import PCA
 
 # Check which network we are processing embeddings from
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCA + LDA grid search over a range of PCA component sizes."
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=["small", "large"],
        required=True,
        help="Dataset size: 'small' or 'large'."
    )
    parser.add_argument(
        "-p", "--path",
        required=True,
        help="File path (no extension) relative to the dataset's estimate directory."
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    # Check multiple pca sizes to try to mitigate feature reduction removing important information
    pca_sizes = list(range(1, 51))  
    param_grid = {
        'solver': ['lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
 
    directory = "/mnt/DiversificationGraphInference/"
    dataset = args.dataset
    if dataset == "large":
        directory = directory + "thirty_thousand_trees/estimate/"
    elif dataset == "small":
        directory = directory + "three_thousand_trees/estimate/"
    else:
        # Unreachable due to argparse choices, but kept for safety.
        print("Error: Dataset size not recognized.")
        raise SystemExit(1)
 
    path = args.path + ".csv"
    full_path = directory + path
    columns = pd.read_csv(full_path, nrows=0).columns

    dtype = {columns[0]: np.int64,**{col: np.float32 for col in columns[1:]}}
    print("loading csv")
    df = pd.read_csv(full_path, dtype=dtype)
    print("loaded csv")
    if dataset == "large":
        train_df = pd.read_csv("../traditional_model_selection_thesis/train_ids.csv")
        test_df = pd.read_csv("../traditional_model_selection_thesis/test_ids.csv")
    else:
        train_df = pd.read_csv("../traditional_model_selection_thesis/small_train_ids.csv")
        test_df = pd.read_csv("../traditional_model_selection_thesis/small_test_ids.csv")
 
    # Get training and testing set
    train_ids = train_df["idx"]
    test_ids = test_df["idx"]
    train_rows = df.iloc[:, 0].isin(train_ids)
    test_rows = df.iloc[:, 0].isin(test_ids)
    y_train = (df.loc[train_rows].iloc[:, 0].values > 1_000_000).astype(int)
    y_test = (df.loc[test_rows].iloc[:, 0].values > 1_000_000).astype(int)
 
    best_accuracy = 0
    best_pca_size = -1
    overall_LDA = None
    results = pd.DataFrame(columns=["pca_size", "accuracy"])
 
    print(y_train.shape)
    print(y_test.shape)
 
    # Look for the best accuracy with a grid search for each PCA size
    for pca_size in pca_sizes:
        print("PCA size:", pca_size)
        pca = PCA(n_components=pca_size, random_state=42)
        X = df.iloc[:, 1:].values
        X = StandardScaler().fit_transform(X)
        X_pca = pca.fit_transform(X)
        X_test_pca = X_pca[test_rows.values]
        X_train_pca = X_pca[train_rows.values]
 
        grid_search = GridSearchCV(
            estimator=LinearDiscriminantAnalysis(),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy', verbose=0
        )
        grid_search.fit(X_train_pca, y_train)
        best_LDA = grid_search.best_estimator_
        y_pred = best_LDA.predict(X_test_pca)
        acc_score = accuracy_score(y_test, y_pred)
        results.loc[len(results)] = [pca_size, acc_score]
 
        if acc_score > best_accuracy:
            overall_LDA = best_LDA
            best_pca_size = pca_size
            best_accuracy = acc_score
            print("new top accuracy", best_accuracy)
 
    print(results)
    results.to_csv("pca_and_LDA_results_" + path + ".csv")
    joblib.dump(overall_LDA, "overall_LDA_" + path)
    print("best pca size:", best_pca_size)
 
 
if __name__ == "__main__":
    main()
