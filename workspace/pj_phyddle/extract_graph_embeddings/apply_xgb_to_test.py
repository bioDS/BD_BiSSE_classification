"""
Apply a saved (joblib-dumped) XGBoost model -- trained by pca_and_XGB_for_bash.py
on PCA-reduced features -- to the test split, and write predictions to a CSV
that also carries along whatever other columns test_ids.csv has.

The original training script only saves the fitted XGBClassifier, not the
PCA/StandardScaler objects. To reproduce the exact same feature space at
inference time, this script:
  1. Infers the PCA component count from the saved model itself
     (model.n_features_in_ == the pca_size that produced the best accuracy).
  2. Refits StandardScaler + PCA on the full feature matrix exactly as
     pca_and_XGB_for_bash.py did (both are deterministic given the same data
     and random_state=42), so the transform matches what the model was
     trained on.
  3. Selects only the test rows (by id) after transforming, predicts, and
     merges predictions back onto test_ids.csv by "idx".

Usage:
    python apply_xgb_to_test.py -d large -p out.2.28500...graph_embeddings \
        [-m overall_XGB_out.2.28500...graph_embeddings.csv] \
        [-o predictions_out.2.28500...graph_embeddings.csv]
"""

import argparse

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a saved XGB model to the test set and save results to CSV."
    )
    parser.add_argument("-d", "--dataset", choices=["small", "large"], required=True,
                         help="Dataset size: 'small' or 'large'.")
    parser.add_argument("-p", "--path", required=True,
                         help="Base file path (no extension), same value used with "
                              "pca_and_XGB_for_bash.py's -p / csv_to_hdf5.py's -o.")
    parser.add_argument("-m", "--model", default=None,
                         help="Path to the saved joblib model. Defaults to the naming "
                              "convention pca_and_XGB_for_bash.py used after the HDF5 patch: "
                              "'overall_XGB_<path>.h5' in the current directory.")
    parser.add_argument("-o", "--output", default=None,
                         help="Output CSV path. Defaults to 'test_predictions_<path>.csv'.")
    parser.add_argument("--pca-size", type=int, default=None,
                         help="Number of PCA components to use. Defaults to the model's "
                              "n_features_in_ (i.e. whatever it was trained on). Override "
                              "this only if you know the model expects a different value.")
    parser.add_argument("--format", choices=["auto", "csv", "h5"], default="auto",
                         help="Feature file format to read. 'auto' (default) uses 'h5' for "
                              "the 'large' dataset and 'csv' for the 'small' dataset, "
                              "matching current practice. Override if that doesn't hold.")
    return parser.parse_args()


def main():
    args = parse_args()

    directory = "/mnt/DiversificationGraphInference/"
    if args.dataset == "large":
        directory += "thirty_thousand_trees/estimate/"
        test_ids_path = "../traditional_model_selection_thesis/test_ids.csv"
    else:
        directory += "three_thousand_trees/estimate/"
        test_ids_path = "../traditional_model_selection_thesis/small_test_ids.csv"

    fmt = args.format
    if fmt == "auto":
        fmt = "h5" if args.dataset == "large" else "csv"
    print(f"feature file format: {fmt}")

    features_path = directory + args.path + ("." + fmt)
    model_path = args.model or ("overall_XGB_" + args.path + "." + fmt)
    output_path = args.output or ("test_predictions_" + args.path + ".csv")

    print(f"loading model: {model_path}")
    model = joblib.load(model_path)
    inferred_pca_size = model.n_features_in_
    if args.pca_size is not None:
        pca_size = args.pca_size
        if pca_size != inferred_pca_size:
            print(f"WARNING: --pca-size {pca_size} differs from model.n_features_in_ "
                  f"({inferred_pca_size}). The model was trained on {inferred_pca_size} "
                  f"features -- predict() will likely raise a shape-mismatch error unless "
                  f"this is intentional (e.g. re-testing a different pca_size's model).")
    else:
        pca_size = inferred_pca_size
    print(f"using pca_size: {pca_size}")

    print(f"loading features: {features_path}")
    if fmt == "h5":
        with h5py.File(features_path, "r") as h5f:
            ids = h5f["ids"][:]
            X = h5f["features"][:]
    else:  # csv
        columns = pd.read_csv(features_path, nrows=0).columns
        dtype_map = {columns[0]: np.int64, **{c: np.float32 for c in columns[1:]}}
        df = pd.read_csv(features_path, dtype=dtype_map)
        ids = df.iloc[:, 0].to_numpy(dtype=np.int64)
        X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    print(f"loaded features: {X.shape}")

    print("loading test_ids.csv")
    test_ids_df = pd.read_csv(test_ids_path)
    test_ids = test_ids_df["idx"]

    print("re-fitting StandardScaler + PCA to match training-time transform")
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=pca_size, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    test_rows = np.isin(ids, test_ids.values)
    X_test_pca = X_pca[test_rows]
    test_ids_ordered = ids[test_rows]
    y_test = (test_ids_ordered > 1_000_000).astype(int)

    print(f"predicting on {X_test_pca.shape[0]} test rows")
    y_pred = model.predict(X_test_pca)
    y_proba = model.predict_proba(X_test_pca)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"accuracy: {acc:.4f}  precision: {prec:.4f}  recall: {rec:.4f}")

    # Build a results frame keyed by idx, then merge onto test_ids.csv so all
    # of its other columns are carried through into the output.
    results_df = pd.DataFrame({
        "idx": test_ids_ordered,
        "y_true": y_test,
        "y_pred": y_pred,
    })
    if y_proba is not None:
        results_df["y_proba"] = y_proba

    merged = test_ids_df.merge(results_df, on="idx", how="inner")
    merged.to_csv(output_path, index=False)
    print(f"wrote {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
