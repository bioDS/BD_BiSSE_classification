# Code to process summary statistics on the small dataset.
# Author: Kate Truman, with AI assistance.
# Code also adapted from the following sources:
# https://www.datacamp.com/tutorial/random-forests-classifier-python
# https://www.geeksforgeeks.org/machine-learning/ml-linear-discriminant-analysis/ 


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


train_df = pd.read_csv("small_train_ids.csv")
test_df = pd.read_csv("small_test_ids.csv")
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

boostClassifier = XGBClassifier()

param_dist = {
  'n_estimators': randint(100,500),
  'max_depth': randint(3,15),
}

rand_search = RandomizedSearchCV(
  boostClassifier, param_distributions=param_dist,
  n_iter=200, cv=5, scoring='accuracy',
  n_jobs=-1, random_state=42, verbose=0)

rand_search.fit(X_train, y_train)

best_x = rand_search.best_estimator_
best_x_preds = best_x.predict(X_test)
best_x_accuracy = accuracy_score(y_test, best_x_preds)
joblib.dump(best_x, 'small_XGB_summary_stats_model.joblib')
print("XGB accuracy:", best_x_accuracy)
with open('small_summary_statistics_results.txt', mode='a') as f:
    f.write(f"XGB accuracy:{best_x_accuracy}")
    for key, value in best_x.get_params().items():
        f.write(f"{key},{value}\n")
rf_without_lda = RandomForestClassifier(max_depth=None, random_state=42)

param_dist = {
            'n_estimators': randint(100,500),
            'max_depth': randint(3,15),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5)
            }

rand_search = RandomizedSearchCV(
        rf_without_lda, param_distributions=param_dist,
        n_iter=200, cv=5, scoring='accuracy',
        n_jobs=-1, random_state=42, verbose=0)

rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
best_rf_preds = best_rf.predict(X_test)
best_rf_accuracy = accuracy_score(y_test, best_rf_preds)
joblib.dump(best_rf, 'small_random_forest_summary_stats_model.joblib')
print("random forest accuracy:", best_rf_accuracy)
with open('small_summary_statistics_results.txt', mode='a') as f:
    f.write(f"random forest accuracy:{best_rf_accuracy}")
    for key, value in best_rf.get_params().items():
        f.write(f"{key},{value}\n")
param_grid = {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9] }
grid_search = GridSearchCV(estimator=LinearDiscriminantAnalysis(),
                            param_grid=param_grid, cv=5,
                            scoring='accuracy', verbose=0)
grid_search.fit(X_train, y_train)
best_LDA = grid_search.best_estimator_
best_LDA_preds = best_LDA.predict(X_test)
best_LDA_accuracy = accuracy_score(y_test, best_LDA_preds)
joblib.dump(best_LDA, 'small_LDA_summary_stats_model.joblib')
print("LDA accuracy:", best_LDA_accuracy)

with open('small_summary_statistics_results.txt', mode='a') as f:
    f.write(f"LDA accuracy:{best_LDA_accuracy}")
    for key, value in best_LDA.get_params().items():
        f.write(f"{key},{value}\n")



