# Code to apply PCA and XGBoostClassifier to graph embeddings.

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
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import h5py
import numpy as np

 # Check which network we are processing embeddings from
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCA + XGBoost randomized search over a range of PCA component sizes."
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
    param_dist = {'n_estimators': randint(100, 500), 'max_depth': randint(3, 15)}

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

    path = args.path + ".h5"
    full_path = directory + path

    print("loading hdf5")
    with h5py.File(full_path, "r") as h5f:
        ids = h5f["ids"][:]                # int64, 1-D -- small, fine to load fully
        X = h5f["features"][:]             # float32, 2-D -- this is the big one
    print("loaded hdf5:", X.shape)
    if dataset == "large":
        train_df = pd.read_csv("../traditional_model_selection_thesis/train_ids.csv")
        test_df = pd.read_csv("../traditional_model_selection_thesis/test_ids.csv")
    else:
        train_df = pd.read_csv("../traditional_model_selection_thesis/small_train_ids.csv")
        test_df = pd.read_csv("../traditional_model_selection_thesis/small_test_ids.csv")

    # Get training and testing set
    test_ids = test_df["idx"]
    train_ids = train_df["idx"]
    train_rows = np.isin(ids, train_ids.values)
    test_rows = np.isin(ids, test_ids.values)
    y_train = (ids[train_rows] > 1_000_000).astype(int)
    y_test = (ids[test_rows] > 1_000_000).astype(int)

    best_accuracy = 0
    best_pca_size = -1
    overall_XGB = None
    results = pd.DataFrame(columns=["pca_size", "accuracy"])
    boostClassifier = XGBClassifier()

    print(y_train.shape)
    print(y_test.shape)
    X = StandardScaler().fit_transform(X)
    # Look for the best accuracy with a random search for each PCA size

    for pca_size in pca_sizes:
        print("PCA size:", pca_size)
        pca = PCA(n_components=pca_size, random_state=42)
        X_pca = pca.fit_transform(X)
        X_test_pca = X_pca[test_rows]
        X_train_pca = X_pca[train_rows]
        rand_search = RandomizedSearchCV(
            boostClassifier, param_distributions=param_dist,
            n_iter=200, cv=5, scoring='accuracy',
            n_jobs=-1, random_state=42, verbose=0)

        rand_search.fit(X_train_pca, y_train)
        # print(f"Best Parameters: {rand_search.best_params_}")
        # print(f"Best Cross-Validated Score: {rand_search.best_score_:.4f}")
        best_XGB = rand_search.best_estimator_
        y_pred = best_XGB.predict(X_test_pca)
        acc_score = accuracy_score(y_test, y_pred)
        results.loc[len(results)] = [pca_size, acc_score]

        if acc_score > best_accuracy:
            overall_XGB = best_XGB
            best_pca_size = pca_size
            best_accuracy = acc_score
            print("new top accuracy", best_accuracy)

    print(results)
    results.to_csv("pca_and_XGB_200_results_" + path + ".csv")
    joblib.dump(overall_XGB, "overall_200_XGB_" + path)
    print("best pca size:", best_pca_size)


if __name__ == "__main__":
    main()
