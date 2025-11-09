#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s5_evaluate_extra_plots.py
Author: ChatGPT (with love for high‑quality, reviewer‑proof plots)
Purpose: Generate additional, model‑driven evaluation figures that complement the standard set.
Outputs: PNG, SVG, PDF for each plot, plus a JSON manifest of summary metrics.

New plot set (unique vs typical scatter/residual/BA/QQ/reliability):
1) Distribution Alignment Panel:
   - KDE density overlay of y_true vs y_pred
   - ECDF overlay and Kolmogorov–Smirnov statistic
   - Jensen–Shannon divergence (base 2, ∈[0,1]) and Wasserstein‑1 distance
2) Tolerance–Accuracy Curve (TAC):
   - Accuracy within ±δ as δ sweeps from 0 to the 95th pct of |residual|
   - Area under TAC (AUTA) and Accuracy@selected tolerances
3) Top‑k Capture Curve (ranking performance for “find the best” use‑cases):
   - Recall of true top‑T% items vs the fraction selected by predicted ranking
   - AUCC, plus Precision at k for k ∈ {1%, 5%, 10%}
4) Prediction Interval Coverage Calibration (PICC) via residual quantiles:
   - Observed coverage vs nominal (50%→99%); mean abs calibration error
5) Heteroscedasticity Profile by predicted quantiles:
   - Mean±SE(|residual|) vs predicted quantile bins
   - Spearman ρ between predicted quantile index and |residual|
"""
import os, json, argparse
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6.0, 5.0),
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed fonts
    "ps.fonttype": 42
})
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _save_all(fig, outbase):
    for ext in ("png","svg","pdf"):
        fig.savefig(f"{outbase}.{ext}")
    plt.close(fig)

def _anchored_stats(ax, text, loc="upper right"):
    box = AnchoredText(text, loc=loc, prop=dict(size=9), frameon=True, pad=0.3, borderpad=0.6)
    box.patch.set_alpha(0.9)
    ax.add_artist(box)

def _ecdf(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    xs = np.sort(x)
    n = xs.size
    ys = np.arange(1, n+1) / n
    return xs, ys

def _gauss_kde(x, grid):
    kde = stats.gaussian_kde(x)
    return kde(grid)

def _js_divergence(p, q, base=2):
    # Jensen-Shannon divergence for discrete distributions p, q (already normalized)
    m = 0.5*(p+q)
    # add small epsilon for numerical stability
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = np.clip(m, eps, None)
    kl_pm = stats.entropy(p, m, base=base)
    kl_qm = stats.entropy(q, m, base=base)
    return 0.5*(kl_pm+kl_qm)

def _se(x):
    x = np.asarray(x)
    return np.std(x, ddof=1)/np.sqrt(max(len(x),1))

def load_array(path, key=None, col=None):
    """Load 1D array from npy/npz/csv. For csv, supply 'col' header."""
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        d = np.load(path)
        if key and key in d:
            arr = d[key]
        else:
            # try common names then fallback to first
            for k in ("y_true","y_pred","true","pred","arr_0"):
                if k in d:
                    arr = d[k]; break
            else:
                arr = list(d.values())[0]
    elif ext in (".csv",".tsv",".txt"):
        import pandas as pd
        sep = "," if ext==".csv" else ("\t" if ext==".tsv" else None)
        df = pd.read_csv(path, sep=sep)
        # pick a sensible default if col not provided
        if col is None:
            for c in ("y_true","true","label","target","Y","y"):
                if c in df.columns: col = c; break
        if col is None:
            raise ValueError(f"For {path}, please provide column name via --true-col/--pred-col.")
        arr = df[col].values
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return np.ravel(arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", required=False, help="Path to y_true (.npy/.npz/.csv).")
    ap.add_argument("--pred", required=False, help="Path to y_pred (.npy/.npz/.csv).")
    ap.add_argument("--csv", required=False, help="Optional single CSV with columns y_true,y_pred.")
    ap.add_argument("--true-col", default=None, help="Column for y_true if loading from CSV.")
    ap.add_argument("--pred-col", default=None, help="Column for y_pred if loading from CSV.")
    ap.add_argument("--true-key", default=None, help="Key for y_true if loading from .npz.")
    ap.add_argument("--pred-key", default=None, help="Key for y_pred if loading from .npz.")
    ap.add_argument("--square", action="store_true", help="Square both y_true and y_pred before evaluation.")
    ap.add_argument("--outdir", default="eval_extra_plots", help="Output directory.")
    ap.add_argument("--prefix", default="", help="Optional filename prefix.")
    ap.add_argument("--top-frac", type=float, default=0.10, help="True top fraction for capture curve (e.g., 0.10).")
    ap.add_argument("--bins", type=int, default=20, help="Number of quantile bins for heteroscedasticity profile.")
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    # Load data
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        y_true = df[args.true_col or ("y_true" if "y_true" in df.columns else "true")].values
        y_pred = df[args.pred_col or ("y_pred" if "y_pred" in df.columns else "pred")].values
    else:
        if not (args.true and args.pred):
            raise SystemExit("Please provide --csv or both --true and --pred.")
        y_true = load_array(args.true, key=args.true_key, col=args.true_col)
        y_pred = load_array(args.pred, key=args.pred_key, col=args.pred_col)

    # Ensure finite and aligned
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true)[mask].astype(float)
    y_pred = np.asarray(y_pred)[mask].astype(float)

    if args.square:
        y_true = np.square(y_true)
        y_pred = np.square(y_pred)

    resid = y_true - y_pred
    abs_resid = np.abs(resid)

    # Manifest to collect summary numbers
    manifest = {"n": int(y_true.size)}

    # 1) Distribution Alignment: KDE + ECDF + metrics
    grid = np.linspace(np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()]), 512)
    p = _gauss_kde(y_true, grid)
    q = _gauss_kde(y_pred, grid)
    # normalize to form discrete distributions on grid for JS
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    js = _js_divergence(p, q, base=2)            # ∈ [0,1]
    w1 = stats.wasserstein_distance(y_true, y_pred)
    ks_D, ks_p = stats.ks_2samp(y_true, y_pred)

    manifest["distribution_alignment"] = {
        "JS_divergence_base2": float(js),
        "Wasserstein_distance": float(w1),
        "KS_D": float(ks_D),
        "KS_pvalue": float(ks_p),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "std_true": float(np.std(y_true, ddof=1)),
        "std_pred": float(np.std(y_pred, ddof=1))
    }

    fig, ax = plt.subplots()
    ax.plot(grid, p, label="True (KDE)")
    ax.plot(grid, q, label="Pred (KDE)", linestyle="--")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Alignment (KDE)")

    # ECDF inset or twin axes
    ax2 = ax.twinx()
    xt, yt = _ecdf(y_true); xp, yp = _ecdf(y_pred)
    ax2.step(xt, yt, where="post", alpha=0.35, label="True ECDF")
    ax2.step(xp, yp, where="post", alpha=0.35, linestyle="--", label="Pred ECDF")
    ax2.set_ylabel("ECDF")
    # KS annotation
    _anchored_stats(ax, f"JS (base 2) = {js:.3f}\nWasserstein = {w1:.3f}\nKS D = {ks_D:.3f} (p={ks_p:.3g})", loc="upper right")
    # Legend handling
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="lower right", framealpha=0.9)
    _save_all(fig, os.path.join(args.outdir, f"{args.prefix}dist_alignment"))

    # 2) Tolerance–Accuracy Curve (TAC)
    delta_max = np.percentile(abs_resid, 95.0)
    deltas = np.linspace(0, delta_max, 200)
    acc = np.array([(np.mean(abs_resid <= d) if d>0 else np.mean(abs_resid == 0)) for d in deltas])
    # Area under tolerance curve normalized by delta_max
    auta = np.trapz(acc, deltas) / (delta_max + 1e-12)
    # Show a few Accuracy@τ points at quartiles of delta_max
    marks = [0.05, 0.1, 0.2, 0.5, 1.0]  # expressed as fractions of delta_max
    acc_marks = {f"Acc@{m:.0%}Δmax": float(np.mean(abs_resid <= (m*delta_max))) for m in marks}
    manifest["tolerance_accuracy"] = {"AUTA": float(auta), "delta_max": float(delta_max), **acc_marks}

    fig, ax = plt.subplots()
    ax.plot(deltas, acc, lw=2)
    ax.set_xlabel("Tolerance δ")
    ax.set_ylabel("Accuracy  P(|error| ≤ δ)")
    ax.set_title("Tolerance–Accuracy Curve")
    for m in marks:
        ax.axvline(m*delta_max, ls=":", lw=0.8, alpha=0.6)
    _anchored_stats(ax, "AUTA = {:.3f}\nδ₉₅ = {:.3f}\n{}".format(auta, delta_max, 
                   "\n".join([f"{k} = {v:.3f}" for k,v in acc_marks.items()])), loc="lower right")
    _save_all(fig, os.path.join(args.outdir, f"{args.prefix}tolerance_accuracy"))

    # 3) Top‑k Capture Curve (ranking for true top‑T% items)
    T = float(args.top_frac)
    n = y_true.size
    k_true = max(1, int(np.round(T * n)))
    # indices for true top T%
    idx_true_top = np.argsort(y_true)[-k_true:]
    true_top_mask = np.zeros(n, dtype=bool)
    true_top_mask[idx_true_top] = True
    # fraction selected s from 0→1
    ss = np.linspace(0.01, 1.0, 200)
    recall = []
    for s in ss:
        k_sel = max(1, int(np.round(s * n)))
        idx_pred_top = np.argsort(y_pred)[-k_sel:]
        sel_mask = np.zeros(n, dtype=bool); sel_mask[idx_pred_top] = True
        rec = np.sum(true_top_mask & sel_mask) / k_true
        recall.append(rec)
    recall = np.array(recall)
    aucc = np.trapz(recall, ss)  # area (max=1)

    # Precision at fixed small k (1%, 5%, 10%)
    prec_at = {}
    for frac in (0.01, 0.05, 0.10):
        k_sel = max(1, int(np.round(frac * n)))
        idx_pred_top = np.argsort(y_pred)[-k_sel:]
        prec = np.sum(true_top_mask[idx_pred_top]) / k_sel
        prec_at[f"P@{int(frac*100)}%"] = float(prec)

    manifest["topk_capture"] = {"true_top_frac": T, "AUCC": float(aucc), **prec_at}

    fig, ax = plt.subplots()
    ax.plot(ss, recall, lw=2)
    ax.plot([0,1],[0,1], ls="--", lw=1, alpha=0.6, label="Random baseline")
    ax.set_xlabel("Fraction selected by predicted rank")
    ax.set_ylabel(f"Recall of true top {int(T*100)}%")
    ax.set_title("Top‑k Capture Curve")
    ax.legend(loc="lower right", framealpha=0.9)
    _anchored_stats(ax, "AUCC = {:.3f}\n{}".format(aucc, "\n".join([f"{k} = {v:.3f}" for k,v in prec_at.items()])), loc="upper left")
    _save_all(fig, os.path.join(args.outdir, f"{args.prefix}topk_capture"))

    # 4) Prediction Interval Coverage Calibration (PICC) via residual quantiles
    cover_levels = np.linspace(0.50, 0.99, 20)  # nominal coverage
    observed = []
    for c in cover_levels:
        q = np.quantile(abs_resid, c)  # half‑width from residual quantile
        lo = y_pred - q
        hi = y_pred + q
        cov = np.mean((y_true >= lo) & (y_true <= hi))
        observed.append(cov)
    observed = np.array(observed)
    mce = float(np.mean(np.abs(observed - cover_levels)))  # mean calibration error

    manifest["interval_coverage"] = {
        "mean_abs_calibration_error": mce,
        "levels": cover_levels.tolist(),
        "observed": observed.tolist()
    }

    fig, ax = plt.subplots()
    ax.plot(cover_levels, observed, marker="o", lw=1.5)
    ax.plot([0.5, 0.99], [0.5, 0.99], ls="--", lw=1, alpha=0.6, label="Ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Prediction Interval Coverage (Residual‑Quantile)")
    ax.set_xlim(0.5, 0.99); ax.set_ylim(0.5, 0.99)
    ax.legend(loc="upper left", framealpha=0.9)
    _anchored_stats(ax, f"Mean abs calibration error = {mce:.3f}", loc="lower right")
    _save_all(fig, os.path.join(args.outdir, f"{args.prefix}interval_coverage"))

    # 5) Heteroscedasticity Profile by predicted quantiles
    nb = int(max(5, args.bins))
    # quantile edges on y_pred
    qs = np.linspace(0, 1, nb+1)
    edges = np.quantile(y_pred, qs)
    # avoid duplicates if y_pred has ties
    edges = np.unique(edges)
    if edges.size < 4:
        # fallback to uniform bins
        edges = np.linspace(y_pred.min(), y_pred.max(), nb+1)
    centers = 0.5*(edges[:-1]+edges[1:])
    mean_abs = []; se_abs = []; counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_pred >= lo) & (y_pred < hi) if hi<edges[-1] else (y_pred >= lo) & (y_pred <= hi)
        vals = abs_resid[m]
        if vals.size == 0:
            mean_abs.append(np.nan); se_abs.append(np.nan); counts.append(0)
        else:
            mean_abs.append(float(np.mean(vals)))
            se_abs.append(float(_se(vals)))
            counts.append(int(vals.size))
    # Spearman correlation between bin index and mean_abs (ignoring NaNs)
    mean_arr = np.array(mean_abs)
    idx = np.arange(mean_arr.size)
    good = np.isfinite(mean_arr)
    if good.sum() >= 3:
        rho, pval = stats.spearmanr(idx[good], mean_arr[good])
    else:
        rho, pval = np.nan, np.nan

    manifest["heteroscedasticity_profile"] = {
        "bins": [{"center": float(c), "count": int(n), "mean_abs_error": (None if not np.isfinite(m) else float(m)), "se": (None if not np.isfinite(s) else float(s))}
                 for c,n,m,s in zip(centers, counts, mean_abs, se_abs)],
        "spearman_rho_index_vs_mean_abs_error": (None if not np.isfinite(rho) else float(rho)),
        "spearman_pvalue": (None if not np.isfinite(pval) else float(pval))
    }

    fig, ax = plt.subplots()
    ax.errorbar(centers, mean_arr, yerr=se_abs, fmt="o-", capsize=3)
    ax.set_xlabel("Predicted value (quantile centers)")
    ax.set_ylabel("Mean ± SE |residual|")
    ax.set_title("Heteroscedasticity by Predicted Quantiles")
    _anchored_stats(ax, f"Spearman ρ = {rho:.3f} (p={pval:.3g})", loc="upper left")
    _save_all(fig, os.path.join(args.outdir, f"{args.prefix}heteroscedasticity_profile"))

    # Save manifest JSON with summary numbers for easy manuscript reference
    man_path = os.path.join(args.outdir, f"{args.prefix}extra_plots_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Generated extra plots and manifest at: {args.outdir}")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
