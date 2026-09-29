# Code to apply PCA and Random Forest to graph embeddings.

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
        description="Run PCA + RandomForest randomized search over a range of PCA component sizes."
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

    pca_sizes = list(range(1, 51))
    # We will do a random search of these parameters to reduce the number of iterations required
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
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
    test_ids = test_df["idx"]
    train_ids = train_df["idx"]
    train_rows = df.iloc[:, 0].isin(train_ids)
    test_rows = df.iloc[:, 0].isin(test_ids)
    y_train = (df.loc[train_rows].iloc[:, 0].values > 1_000_000).astype(int)
    y_test = (df.loc[test_rows].iloc[:, 0].values > 1_000_000).astype(int)

    best_accuracy = 0
    best_pca_size = -1
    overall_RF = None
    results = pd.DataFrame(columns=["pca_size", "accuracy"])

    print(y_train.shape)
    print(y_test.shape)

    # Look for the best accuracy with a random search for each PCA size
    for pca_size in pca_sizes:
        print("PCA size:", pca_size)
        pca = PCA(n_components=pca_size, random_state=42)
        X = df.iloc[:, 1:].values
        X = StandardScaler().fit_transform(X)
        X_pca = pca.fit_transform(X)
        X_test_pca = X_pca[test_rows.values]
        X_train_pca = X_pca[train_rows.values]
        rf = RandomForestClassifier(max_depth=None, random_state=42)

        rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy',
                                          n_jobs=-1, random_state=42, verbose=0)
        rand_search.fit(X_train_pca, y_train)

        # print(f"Best Parameters: {rand_search.best_params_}")
        # print(f"Best Cross-Validated Score: {rand_search.best_score_:.4f}")
        best_RF = rand_search.best_estimator_
        y_pred = best_RF.predict(X_test_pca)
        acc_score = accuracy_score(y_test, y_pred)
        results.loc[len(results)] = [pca_size, acc_score]

        if acc_score > best_accuracy:
            overall_RF = best_RF
            best_pca_size = pca_size
            best_accuracy = acc_score
            print("new top accuracy", best_accuracy)

    print(results)
    results.to_csv("pca_and_RF_results_" + path + ".csv")
    joblib.dump(overall_RF, "overall_RF_" + path)
    print("best pca size:", best_pca_size)


if __name__ == "__main__":
    main()
