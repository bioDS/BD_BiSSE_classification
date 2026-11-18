#!/usr/bin/env python
"""
ROC curve + AUC for a saved phyddle GCN classifier (BD vs BiSSE, 2 classes).

Two modes
---------
1) From a saved model + the Format test HDF5 file (for older runs):

    python roc_from_saved_model.py model \
        --model  train/out.2.2850....relu.NA.trained_model.pkl \
        --test   format/out.test.<enc>.hdf5 \
        --norm   train/out.2.2850....relu.NA.trn_norm.attr_data.csv

   --norm is the node-feature mean/sd saved by Train (same name as the .pkl,
   with "trained_model.pkl" -> "trn_norm.attr_data.csv"). The test trees must
   be normalised with the TRAINING mean/sd, exactly as in Train.

2) From test_predictions.csv (written by test_model() in the current train.py):

    python roc_from_saved_model.py csv --csv train/test_predictions.csv

Run in the environment you trained in (torch, torch_geometric, phyddle).
Outputs roc_curve.png and roc_points.csv in the current folder (or --out).
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

MODEL_TYPE_NAMES = {0: 'BD', 1: 'BiSSE', 2: 'MuSSE_3', 3: 'MuSSE_4', 4: 'MuSSE_5'}


# ---------------------------------------------------------------- plotting
def save_roc(y_true, p_pos, pos_name, neg_name, out_dir):
    fpr, tpr, thr = roc_curve(y_true, p_pos)
    auc = roc_auc_score(y_true, p_pos)
    acc = accuracy_score(y_true, (p_pos >= 0.5).astype(int))
    print(f"n = {len(y_true)}  ({int((y_true == 1).sum())} {pos_name}, "
          f"{int((y_true == 0).sum())} {neg_name})")
    print(f"accuracy (threshold 0.5) = {acc:.4f}")
    print(f"AUC = {auc:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"GCN (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    plt.xlabel(f"False positive rate ({neg_name} called {pos_name})")
    plt.ylabel(f"True positive rate ({pos_name} called {pos_name})")
    plt.title(f"ROC: {pos_name} vs {neg_name} (test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=200)
    np.savetxt(os.path.join(out_dir, "roc_points.csv"), np.c_[fpr, tpr, thr],
               delimiter=",", header="fpr,tpr,threshold", comments="")
    print(f"wrote {out_dir}/roc_curve.png and {out_dir}/roc_points.csv")


def save_confident(y_true, p_pos, pos_name, neg_name, out_dir, cut=0.75):
    """Accuracy counting only confident predictions: P(class 1) > cut or
    < 1 - cut. Trees in between are left out (not counted as right or wrong)."""
    BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    AQUA, VIOLET = "#1baf7a", "#6250d6"   # right panel: different measures, different hues
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos, dtype=float)
    pred = (p_pos >= 0.5).astype(int)
    conf = (p_pos > cut) | (p_pos < 1 - cut)
    n, n_conf = len(y_true), int(conf.sum())

    def acc(mask):
        return float((pred[mask] == y_true[mask]).mean()) if mask.any() else np.nan

    groups = [("Overall", np.ones(n, bool)), (neg_name, y_true == 0), (pos_name, y_true == 1)]
    rows = []
    for name, m in groups:
        rows.append((name, acc(m), int(m.sum()), acc(m & conf), int((m & conf).sum())))

    print(f"\nconfident predictions (P({pos_name}) > {cut} or < {1-cut:.2f}): "
          f"{n_conf} of {n} trees ({100*n_conf/max(n,1):.1f}%)")
    for name, a_all, k_all, a_c, k_c in rows:
        print(f"  {name:8s} accuracy: all {a_all:.4f} (n={k_all})   "
              f"confident {a_c:.4f} (n={k_c}, {100*k_c/max(k_all,1):.1f}% kept)")

    # confusion matrix of the confident trees
    import pandas as pd
    cm = np.zeros((2, 2), int)
    np.add.at(cm, (y_true[conf], pred[conf]), 1)
    pd.DataFrame(cm, index=[f"true_{neg_name}", f"true_{pos_name}"],
                 columns=[f"pred_{neg_name}", f"pred_{pos_name}"]).to_csv(
        os.path.join(out_dir, "confident_confusion.csv"))
    pd.DataFrame(rows, columns=["group", "acc_all", "n_all", "acc_confident", "n_confident"]).to_csv(
        os.path.join(out_dir, "confident_accuracy.csv"), index=False)

    # ---- figure
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) accuracy, all vs confident only, per class
    x = np.arange(len(rows))
    w = 0.36
    for off, col, key, lab in ((-w/2 - 0.01, BLUE, 1, "All predictions"),
                               (w/2 + 0.01, ORANGE, 3, f"Confident only (>{cut} or <{1-cut:.2f})")):
        vals = [r[key] for r in rows]
        bars = ax1.bar(x + off, np.nan_to_num(vals), w, color=col, label=lab, zorder=2)
        for b, v, r in zip(bars, vals, rows):
            k = r[key + 1]
            txt = "n=0" if np.isnan(v) else f"{v:.2f}\nn={k}"
            ax1.text(b.get_x() + b.get_width()/2, (0 if np.isnan(v) else v) + 0.015, txt,
                     ha="center", va="bottom", fontsize=8, color=INK2)
    ax1.set_xticks(x, [f"{r[0]}" for r in rows])
    ax1.set_ylim(0, 1.15)
    ax1.set_yticks(np.linspace(0, 1, 6))
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Accuracy, confident predictions only\n"
                  f"{n_conf} of {n} test trees ({100*n_conf/max(n,1):.0f}%) are confident",
                  fontsize=10, color=INK, loc="left")
    ax1.legend(frameon=False, loc="upper left", fontsize=8, ncol=1, bbox_to_anchor=(0, 1.0))
    ax1.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    # (b) trade-off: stricter cutoff -> fewer trees kept, (usually) higher accuracy
    cuts = np.round(np.arange(0.50, 1.0, 0.01), 2)
    acc_c, kept = [], []
    for c in cuts:
        m = (p_pos > c) | (p_pos < 1 - c) if c > 0.5 else np.ones(n, bool)
        acc_c.append(acc(m))
        kept.append(m.mean())
    ax2.plot(cuts, acc_c, color=AQUA, lw=2, zorder=3)
    ax2.plot(cuts, kept, color=VIOLET, lw=2, zorder=3)
    ax2.axvline(cut, color=INK2, lw=1, ls="--", zorder=1)
    i = int(np.argmin(np.abs(cuts - cut)))
    for vals, col in ((acc_c, AQUA), (kept, VIOLET)):
        if not np.isnan(vals[i]):
            ax2.plot(cuts[i], vals[i], "o", ms=7, color=col, mec="#fcfcfb", mew=2, zorder=4)
    # direct labels at the right end of each line
    last = max(j for j in range(len(cuts)) if not np.isnan(acc_c[j]))
    ax2.text(cuts[last] + 0.005, acc_c[last], "  accuracy of\n  kept trees", color=INK2,
             va="center", fontsize=8)
    ax2.text(cuts[-1] + 0.005, kept[-1], "  fraction of\n  trees kept", color=INK2,
             va="center", fontsize=8)
    ax2.text(cut + 0.008, 1.05, f"cutoff {cut}", ha="left", fontsize=8, color=INK2)
    ax2.set_xlim(0.5, 1.07)
    ax2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks(np.linspace(0, 1, 6))
    ax2.set_xlabel(f"Confidence cutoff c  (keep trees with P({pos_name}) > c or < 1 − c)")
    ax2.set_ylabel("Proportion")
    ax2.set_title("Accuracy vs. how many trees are kept", fontsize=10, color=INK, loc="left")
    ax2.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fn = os.path.join(out_dir, "confident_accuracy.png")
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"wrote {fn}, confident_accuracy.csv and confident_confusion.csv")


def save_confidence_hist(y_true, p_pos, pos_name, neg_name, out_dir, cut=0.75):
    """Histogram of P(class 1) for every test tree, stacked by true class, with
    the uncertain band (1 - cut .. cut) shaded."""
    BLUE, ORANGE, INK, INK2, GRID, SURF = ("#2a78d6", "#eb6834", "#0b0b0b",
                                           "#52514e", "#e4e3df", "#fcfcfb")
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos, dtype=float)
    n = len(p_pos)
    # bin edges fall exactly on 1 - cut and cut when cut is a multiple of 0.05
    bins = np.linspace(0, 1, 21)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    ax.axvspan(1 - cut, cut, color="#ebeae6", zorder=0, lw=0)
    ax.hist([p_pos[y_true == 0], p_pos[y_true == 1]], bins=bins, stacked=True,
            color=[BLUE, ORANGE], edgecolor=SURF, linewidth=1.5, zorder=2,
            label=[f"true {neg_name} (n={int((y_true == 0).sum())})",
                   f"true {pos_name} (n={int((y_true == 1).sum())})"])
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)

    # counts in each zone
    lo, mid, hi = p_pos < 1 - cut, (p_pos >= 1 - cut) & (p_pos <= cut), p_pos > cut
    top = ax.get_ylim()[1] * 1.12
    ax.set_ylim(0, top)
    for (x0, x1), m, txt in (((0, 1 - cut), lo, f"confident {neg_name}"),
                             ((1 - cut, cut), mid, "uncertain - not counted"),
                             ((cut, 1), hi, f"confident {pos_name}")):
        k = int(m.sum())
        wrong = int(((p_pos[m] >= 0.5).astype(int) != y_true[m]).sum()) if txt.startswith("conf") else None
        sub = f"{k} trees ({100*k/max(n,1):.0f}%)" + (f", {wrong} wrong" if wrong is not None else "")
        ax.text((x0 + x1) / 2, top * 0.97, f"{txt}\n{sub}", ha="center", va="top",
                fontsize=8.5, color=INK2)
    for x in (1 - cut, cut):
        ax.axvline(x, color=INK2, lw=1, ls="--", zorder=3)

    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xlabel(f"Predicted P({pos_name})")
    ax.set_ylabel("Number of test trees")
    ax.set_title(f"Model confidence for all {n} test trees", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fn = os.path.join(out_dir, "confidence_histogram.png")
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"wrote {fn}")


# ---------------------------------------------------------------- csv mode
def run_csv(a):
    import pandas as pd
    df = pd.read_csv(a.csv)
    classes = [int(c) for c in a.classes.split(",")]
    neg, pos = classes
    pos_name, neg_name = MODEL_TYPE_NAMES.get(pos), MODEL_TYPE_NAMES.get(neg)
    y = (df["true_model_type"].values == pos).astype(int)
    save_roc(y, df[f"prob_{pos_name}"].values, pos_name, neg_name, a.out)
    save_confident(y, df[f"prob_{pos_name}"].values, pos_name, neg_name, a.out, a.cut)
    save_confidence_hist(y, df[f"prob_{pos_name}"].values, pos_name, neg_name, a.out, a.cut)


# ---------------------------------------------------------------- model mode
def read_test_graphs(fn, classes):
    """Read trees from a Format HDF5 file, rebuilding each graph the same way
    as HDF5BlockDataset.__getitem__ in train.py."""
    import h5py
    import torch
    from torch_geometric.data import Data
    with h5py.File(fn, "r") as f:
        num_edges = np.asarray(f["num_edges"][:], dtype=np.int64).ravel()
        num_nodes = np.asarray(f["num_nodes"][:], dtype=np.int64).ravel()
        names = [x.decode("utf-8") for x in np.atleast_1d(f["label_names"][:].squeeze())]
        mt = np.rint(f["labels"][:, names.index("model_type")]).astype(np.int64)
        node_1 = f["node_1"][:].astype(np.int64)
        node_2 = f["node_2"][:].astype(np.int64)
        node_attr = f["node_attr"][:].astype(np.float32)

    e_off = np.concatenate([[0], np.cumsum(num_edges)])
    n_off = np.concatenate([[0], np.cumsum(num_nodes)])
    lut = {v: i for i, v in enumerate(classes)}   # model_type -> class index

    graphs = []
    for g in np.where(np.isin(mt, classes))[0]:
        s, e = e_off[g], e_off[g + 1]
        ei = np.stack([node_1[s:e], node_2[s:e]], axis=0)
        ei = ei - ei.min()
        allv = np.arange(ei.max() + 1)
        gaps = allv[~np.isin(allv, np.unique(ei))]
        ei = ei - np.searchsorted(gaps, allv, side="left")[ei]
        x = node_attr[n_off[g]:n_off[g + 1]]
        assert 2 * (x.shape[0] - 1) == ei.shape[1], f"node/edge mismatch in tree {g}"
        graphs.append(Data(x=torch.from_numpy(x),
                           edge_index=torch.from_numpy(ei),
                           y=torch.tensor([lut[int(mt[g])]])))
    skipped = int((~np.isin(mt, classes)).sum())
    if skipped:
        print(f"skipped {skipped} test trees whose model_type is not in {classes}")
    return graphs


def gcn_avg_forward(net, x, edge_index, batch):
    """Forward pass for the saved GCN_avg layout:
    gconv1 3->50, gconv2 50->50, gconv3 50->50, gconv4 50->100,
    mean pool, fc1 100->50, fc2 50->2.
    (The current network.py has no gconv4, so model(batch) would fail.)"""
    import torch.nn.functional as F
    from torch_geometric.nn import global_mean_pool
    act = F.leaky_relu if getattr(net, "activation_func", "relu") == "leaky_relu" else F.relu
    for layer in (net.gconv1, net.gconv2, net.gconv3, net.gconv4):
        x = act(layer(x, edge_index))          # dropout is off in eval mode
    x = global_mean_pool(x, batch)
    x = net.fc1(x)
    return net.fc2(x)


def run_model(a):
    import torch
    import pandas as pd
    from torch_geometric.data import Batch

    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    try:
        model = torch.load(a.model, map_location=dev, weights_only=False)
    except TypeError:                                   # older torch
        model = torch.load(a.model, map_location=dev)
    model = getattr(model, "module", model)             # unwrap DataParallel if present
    model.to(dev).eval()
    for m in model.modules():                           # saved on cuda; follow `dev`
        if hasattr(m, "TORCH_DEVICE"):
            m.TORCH_DEVICE, m.TORCH_DEVICE_STR = dev, dev.type
    net = model.phy_std
    print("pooling network:", type(net).__name__,
          "| layers:", [n for n, _ in net.named_children()])

    norm = pd.read_csv(a.norm)
    mean = torch.tensor(norm["mean"].values, dtype=torch.float32)
    sd = torch.tensor(norm["sd"].values, dtype=torch.float32)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))

    classes = [int(c) for c in a.classes.split(",")]
    graphs = read_test_graphs(a.test, classes)
    for g in graphs:
        g.x = (g.x - mean) / sd

    probs, ys = [], []
    with torch.no_grad():
        for i in range(0, len(graphs), a.batch_size):
            b = Batch.from_data_list(graphs[i:i + a.batch_size]).to(dev)
            if hasattr(net, "gconv4"):
                logits = gcn_avg_forward(net, b.x, b.edge_index, b.batch)
            else:                                       # matches the current code
                logits = model(b)[3]
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            ys.append(b.y.view(-1).cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(probs)
    neg, pos = classes
    save_roc(y, p, MODEL_TYPE_NAMES.get(pos), MODEL_TYPE_NAMES.get(neg), a.out)
    save_confident(y, p, MODEL_TYPE_NAMES.get(pos), MODEL_TYPE_NAMES.get(neg), a.out, a.cut)
    save_confidence_hist(y, p, MODEL_TYPE_NAMES.get(pos), MODEL_TYPE_NAMES.get(neg), a.out, a.cut)
    pd.DataFrame({"true_class": y, f"prob_{MODEL_TYPE_NAMES.get(pos)}": p}).to_csv(
        os.path.join(a.out, "test_probs.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("model")
    m.add_argument("--model", required=True)
    m.add_argument("--test", required=True, help="Format test HDF5 file")
    m.add_argument("--norm", required=True, help="trn_norm.attr_data.csv from Train")
    m.add_argument("--batch-size", type=int, default=64)
    m.add_argument("--cpu", action="store_true")
    c = sub.add_parser("csv")
    c.add_argument("--csv", required=True)
    for p in (m, c):
        p.add_argument("--classes", default="0,1",
                       help="model_type of class 0 and class 1 (default 0,1 = BD,BiSSE)")
        p.add_argument("--out", default=".")
        p.add_argument("--cut", type=float, default=0.75,
                       help="count only predictions with P(class 1) > cut or < 1 - cut "
                            "(default 0.75)")
    a = ap.parse_args()
    run_model(a) if a.mode == "model" else run_csv(a)