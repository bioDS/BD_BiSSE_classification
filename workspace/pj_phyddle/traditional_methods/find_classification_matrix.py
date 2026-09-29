# Code to apply best performing XGBClassifier model on the summary statistics and vector encodings for the large dataset.
# Similar scripts can be used to apply other classifiers.
# In the future they should be combined into one flexible script.
# Author: Kate Truman, with use of AI tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from scipy.stats import randint
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

model_path = '/home/ket581/AIphylo/phyddle/workspace/pj_phyddle/traditional_model_selection_thesis/XGB_summary_stats_model.joblib'
loaded_model = joblib.load(model_path)
train_df = pd.read_csv("train_ids.csv")
test_df = pd.read_csv("test_ids.csv")
df_full = pd.concat([test_df, train_df])
df = df_full.drop(columns=['tree_path', 'birth', 'death','birth_0', 'birth_1', 'death_0', 'death_1', 'root_age', 'age_mean'])
test_ids = test_df["idx"]
train_ids = train_df["idx"]
sc = StandardScaler()
X = df.iloc[:, 2:]
X_scaled = sc.fit_transform(X)
train_rows = df_full['idx'].isin(train_df['idx'])
test_rows = df_full['idx'].isin(test_df['idx'])
X_train = X_scaled[train_rows,]
X_test = X_scaled[test_rows,]
X_test_df = pd.DataFrame(X_test, columns=df.iloc[:,2:].columns)
X_test_df.insert(0, "idx", test_df["idx"].values)
X_test_df.to_csv("X_test_scaled.csv", index=False)
y_train = train_df['label']
y_test = test_df['label']
preds = loaded_model.predict(X_test)
labels = [0,1]
cm = confusion_matrix(
    y_test,
    preds,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {x}" for x in labels],
    columns=[f"Predicted {x}" for x in labels]
)
print(cm_df)
test_df["preds"] = preds
test_df.to_csv("/home/ket581/AIphylo/phyddle/workspace/pj_phyddle/data_to_download/XGB_summary_stats_preds.csv")

loaded_model = joblib.load('/home/ket581/AIphylo/phyddle/workspace/pj_phyddle/traditional_model_selection_thesis/overall_XGB')
df = pd.read_csv("chv.csv")
train_df = pd.read_csv("train_ids.csv")
train_ids = train_df["idx"]
test_df = pd.read_csv("test_ids.csv")
test_ids = test_df["idx"]
train_rows = df.iloc[:, 0].isin(train_ids)
test_rows = df.iloc[:, 0].isin(test_ids)

y_train = (df.loc[train_rows].iloc[:, 0].values > 1_000_000).astype(int)
y_test = (df.loc[test_rows].iloc[:, 0].values > 1_000_000).astype(int)
X = df.iloc[:,1:].values
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=40, random_state=42)
X_pca = pca.fit_transform(X)
X_test = X_pca[test_rows.values]
preds = loaded_model.predict(X_test)
cm = confusion_matrix(
    y_test,
    preds,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {x}" for x in labels],
    columns=[f"Predicted {x}" for x in labels]
)
print(cm_df)
test_df["preds"] = preds
test_df.to_csv("/home/ket581/AIphylo/phyddle/workspace/pj_phyddle/data_to_download/XGB_CHV_preds.csv")
