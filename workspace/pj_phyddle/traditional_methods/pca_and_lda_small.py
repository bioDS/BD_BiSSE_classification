# Apply PCA and LDA to CHV encodings for the small dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from scipy.stats import randint
import joblib
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
pca_sizes = list(range(1,51))
param_grid = {
            'solver': ['lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9]
                }

train_df = pd.read_csv("chv_small_train.csv", header=None)
test_df = pd.read_csv("chv_small_test.csv", header=None)
df = pd.concat(
    [train_df.assign(_source="train"),
     test_df.assign(_source="test")],
    ignore_index=True
)
test_rows = df.index[df["_source"] == "test"]
train_rows = df.index[df["_source"] == "train"]

y_train = (train_df.iloc[:, 0].values > 1_000_000).astype(int)
y_test = (test_df.iloc[:, 0].values > 1_000_000).astype(int)
best_accuracy = 0
best_pca_size = -1
print(y_train.shape)
print(y_test.shape)
print(len(test_rows))
print(len(train_rows))
df = df.drop(columns="_source")
results = pd.DataFrame(columns=["pca_size", "accuracy"])
for pca_size in pca_sizes:
    pca = PCA(n_components=pca_size, random_state=42)
    X = df.iloc[:,1:].values
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
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validated Score: {grid_search.best_score_:.4f}")
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
results.to_csv("pca_and_lda_results.csv")
joblib.dump(overall_LDA, "overall_LDA")
print("best pca size:", best_pca_size)
