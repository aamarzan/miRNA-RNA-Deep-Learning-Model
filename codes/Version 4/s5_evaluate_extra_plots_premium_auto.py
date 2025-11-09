#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s5_evaluate_extra_plots_premium_auto.py
Author: ChatGPT
Purpose: Premium-styled, *additional* evaluation plots with gradient visuals, color legends,
         and on-plot metric tiles. Auto-loads your model/test data from config.json exactly
         like your main evaluation script; no extra inputs required.

New plots (unique vs your current set):
  S3) Distribution Alignment (KDE+ECDF with ECDF-Δ heat ribbon; JS, Wasserstein, KS)
  S4) Tolerance–Accuracy Curve with gradient by local slope; AUTA + Acc@{5,10,20,50,100}%Δ95
  S5) Top‑k Capture Curve with gradient by recall; AUCC + P@{1%,5%,10%}
  S6) Prediction Interval Coverage: points colored by |obs−nom|; mean abs calibration error
  S7) Heteroscedasticity by Pred‑quantiles: markers colored by bin count; Spearman trend

Outputs:
  - High‑res PNG/SVG/PDF for each figure
  - extra_plots_manifest.json with summary metrics and file paths
"""
import os, json, argparse, datetime as dt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy import stats

# ---------------- Premium style ----------------
def set_premium_style(dpi=800):
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": dpi,
        "font.size": 10.5, "font.family": "DejaVu Sans",
        "axes.titlesize": 13, "axes.labelsize": 11.5, "axes.linewidth": 1.0,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "savefig.bbox": "tight"
    })

def golden_figsize(w=5.4):
    phi=(5**0.5-1)/2
    return (w, w*phi*1.06)

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _save_all(fig, outbase):
    for ext in ("png","svg","pdf"):
        fig.savefig(f"{outbase}.{ext}")
    plt.close(fig)

def _anchored(ax, text, loc="upper right", fontsize=9.2, alpha=0.96):
    box = AnchoredText(text, loc=loc, prop=dict(size=fontsize), frameon=True, pad=0.32, borderpad=0.62)
    box.patch.set_alpha(alpha)
    ax.add_artist(box)

def _ecdf(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    xs = np.sort(x); n = xs.size; ys = np.arange(1, n+1) / n
    return xs, ys

def _gauss_kde(x, grid):
    kde = stats.gaussian_kde(x)
    return kde(grid)

def _js_divergence(p, q, base=2):
    m = 0.5*(p+q); eps = 1e-12
    p = np.clip(p, eps, None); q = np.clip(q, eps, None); m = np.clip(m, eps, None)
    kl_pm = stats.entropy(p, m, base=base); kl_qm = stats.entropy(q, m, base=base)
    return 0.5*(kl_pm+kl_qm)

def _se(x):
    x = np.asarray(x); n = len(x)
    return (np.std(x, ddof=1) / np.sqrt(max(n,1))) if n>1 else 0.0

def _colorline(ax, x, y, c, cmap="viridis", lw=2.2, norm=None, zorder=3):
    """Draw a line colored by array c (same length as x), with colorbar handle returned."""
    x, y, c = map(np.asarray, (x,y,c))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    if norm is None:
        norm = Normalize(vmin=np.nanmin(c), vmax=np.nanmax(c))
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder)
    lc.set_array(c)
    ax.add_collection(lc)
    return lc, norm

# ---------------- project‑specific loaders (mirrors your main evaluator) ----------------
def load_config(path):
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def load_npz_test(data_path):
    X_test={}
    test_files=[f for f in os.listdir(data_path) if f.startswith("X_test_") and f.endswith(".npz")]
    if not test_files: raise FileNotFoundError(f"No X_test_*.npz in {data_path}")
    for f in sorted(test_files):
        key=f.replace("X_test_","").replace(".npz","")
        with np.load(os.path.join(data_path,f), mmap_mode="r") as z:
            X_test[key]=z["data"]
    with np.load(os.path.join(data_path,"y_test.npz"), mmap_mode="r") as z:
        y_test=z["data"]
    return X_test, y_test

# ---------------- Figures ----------------
def figS3_distribution_alignment(y_true,y_pred,out_dir,stdnames,cmap="magma"):
    # KDEs
    grid = np.linspace(np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()]), 1024)
    p = _gauss_kde(y_true, grid); q = _gauss_kde(y_pred, grid)
    p = p/(p.sum()+1e-12); q = q/(q.sum()+1e-12)
    js = _js_divergence(p, q, base=2)
    w1 = stats.wasserstein_distance(y_true, y_pred)
    ks_D, ks_p = stats.ks_2samp(y_true, y_pred)
    # ECDFs + heat ribbon for |Δ ECDF| across x
    xt, yt = _ecdf(y_true); xp, yp = _ecdf(y_pred)
    # prepare |Δ| on common grid via linear interpolation
    tt = np.interp(grid, xt, yt, left=0, right=1); pp = np.interp(grid, xp, yp, left=0, right=1)
    ecdf_abs_diff = np.abs(tt-pp)
    fig, ax = plt.subplots(figsize=golden_figsize(6.0))
    # Draw heat ribbon along x as colored background band
    ax.imshow(ecdf_abs_diff[np.newaxis, :], extent=[grid.min(), grid.max(), 0, 1],
              aspect="auto", cmap=cmap, alpha=0.33, origin="lower", zorder=1)
    # Plot KDEs on top
    ax2 = ax.twinx()
    ax.plot(grid, p, color="black", lw=1.8, label="True (KDE)", zorder=4)
    ax.plot(grid, q, color="black", lw=1.8, ls="--", label="Pred (KDE)", zorder=4)
    ax.set_ylabel("Density"); ax2.set_ylabel("ECDF")
    # ECDF curves
    ax2.step(xt, yt, where="post", lw=1.4, alpha=0.7, color="#2a9d8f", label="True ECDF", zorder=3)
    ax2.step(xp, yp, where="post", lw=1.4, alpha=0.7, color="#e76f51", ls="--", label="Pred ECDF", zorder=3)
    ax.set_xlabel("Value"); ax.set_title("Distribution Alignment (KDE · ECDF |Δ| heat)")
    # Colorbar for |Δ ECDF|
    m = mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=np.nanmax(ecdf_abs_diff)), cmap=cmap)
    cbar = plt.colorbar(m, ax=ax, pad=0.012); cbar.set_label("ECDF |Δ|", rotation=90)
    # Legends + metrics
    lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="lower right", framealpha=0.9)
    _anchored(ax, f"JS (base 2) = {js:.3f}\nWasserstein = {w1:.3f}\nKS D = {ks_D:.3f} (p={ks_p:.3g})", loc="upper right")
    base=os.path.join(out_dir,"FigureS3_distribution_alignment" if stdnames else "dist_alignment_premium")
    _save_all(fig,base)
    return {"JS_divergence_base2": float(js), "Wasserstein_distance": float(w1), "KS_D": float(ks_D), "KS_pvalue": float(ks_p)}

def figS4_tolerance_accuracy(y_true,y_pred,out_dir,stdnames,cmap="viridis"):
    resid = y_true - y_pred; abs_resid = np.abs(resid)
    delta_max = np.percentile(abs_resid, 95.0)
    deltas = np.linspace(0, delta_max, 300)
    acc = np.array([(np.mean(abs_resid <= d) if d>0 else np.mean(abs_resid == 0)) for d in deltas])
    # Color by local slope (smooth derivative)
    slope = np.gradient(acc, deltas); slope = np.nan_to_num(slope)
    fig, ax = plt.subplots(figsize=golden_figsize(6.0))
    lc, norm = _colorline(ax, deltas, acc, slope, cmap=cmap, lw=2.8)
    ax.plot([deltas.min(), deltas.max()], [acc[0], acc[0]], lw=0.8, ls=":", color="k", alpha=0.5)
    ax.set_xlabel("Tolerance δ"); ax.set_ylabel("Accuracy  P(|error| ≤ δ)"); ax.set_title("Tolerance–Accuracy (gradient by local slope)")
    cbar = plt.colorbar(lc, ax=ax, pad=0.012); cbar.set_label("Local slope dAcc/dδ")
    auta = np.trapz(acc, deltas) / (delta_max + 1e-12)
    marks = [0.05, 0.1, 0.2, 0.5, 1.0]
    acc_marks = {f"Acc@{int(m*100)}%Δmax": float(np.mean(abs_resid <= (m*delta_max))) for m in marks}
    for m in marks: ax.axvline(m*delta_max, ls="--", lw=0.7, alpha=0.4)
    _anchored(ax, "AUTA = {:.3f}\nδ₉₅ = {:.3f}\n{}".format(auta, delta_max, "\n".join([f"{k} = {v:.3f}" for k,v in acc_marks.items()])), loc="lower right")
    base=os.path.join(out_dir,"FigureS4_tolerance_accuracy" if stdnames else "tolerance_accuracy_premium")
    _save_all(fig,base)
    return {"AUTA": float(auta), "delta_max": float(delta_max), **acc_marks}

def figS5_topk_capture(y_true,y_pred,out_dir,stdnames, top_frac=0.10, cmap="plasma"):
    T = float(top_frac); n = y_true.size; k_true = max(1, int(np.round(T * n)))
    idx_true_top = np.argsort(y_true)[-k_true:]
    true_top_mask = np.zeros(n, dtype=bool); true_top_mask[idx_true_top] = True
    ss = np.linspace(0.01, 1.0, 300); recall = []
    for s in ss:
        k_sel = max(1, int(np.round(s * n)))
        idx_pred_top = np.argsort(y_pred)[-k_sel:]
        recall.append(np.sum(true_top_mask[idx_pred_top]) / k_true)
    recall = np.array(recall)
    fig, ax = plt.subplots(figsize=golden_figsize(6.0))
    lc, norm = _colorline(ax, ss, recall, recall, cmap=cmap, lw=2.8)
    ax.plot([0,1],[0,1], ls="--", lw=0.9, alpha=0.6, color="k", label="Random baseline")
    ax.set_xlabel("Fraction selected by predicted rank"); ax.set_ylabel(f"Recall of true top {int(T*100)}%")
    ax.set_title("Top‑k Capture (gradient by recall)"); ax.legend(loc="lower right", framealpha=0.9)
    cbar = plt.colorbar(lc, ax=ax, pad=0.012); cbar.set_label("Recall")
    aucc = np.trapz(recall, ss)
    prec_at={}
    for frac in (0.01, 0.05, 0.10):
        k_sel = max(1, int(np.round(frac * n)))
        idx_pred_top = np.argsort(y_pred)[-k_sel:]
        prec_at[f"P@{int(frac*100)}%"] = float(np.sum(true_top_mask[idx_pred_top]) / k_sel)
    _anchored(ax, "AUCC = {:.3f}\n{}".format(aucc, "\n".join([f"{k} = {v:.3f}" for k,v in prec_at.items()])), loc="upper left")
    base=os.path.join(out_dir,"FigureS5_topk_capture" if stdnames else "topk_capture_premium")
    _save_all(fig,base)
    return {"true_top_frac": T, "AUCC": float(aucc), **prec_at}

def figS6_interval_coverage(y_true,y_pred,out_dir,stdnames,cmap="cividis"):
    resid = y_true - y_pred; abs_resid = np.abs(resid)
    cover_levels = np.linspace(0.50, 0.99, 20)
    observed = []
    for c in cover_levels:
        q = np.quantile(abs_resid, c)
        lo = y_pred - q; hi = y_pred + q
        observed.append(np.mean((y_true >= lo) & (y_true <= hi)))
    observed = np.array(observed); err = np.abs(observed - cover_levels)
    mce = float(np.mean(err))
    fig, ax = plt.subplots(figsize=golden_figsize(6.0))
    sc = ax.scatter(cover_levels, observed, c=err, cmap=cmap, s=55, zorder=3)
    ax.plot([0.5,0.99],[0.5,0.99], ls="--", lw=0.9, alpha=0.6, color="k", label="Ideal")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage")
    ax.set_title("Prediction Interval Coverage (points colored by |obs−nom|)")
    ax.set_xlim(0.5,0.99); ax.set_ylim(0.5,0.99); ax.legend(loc="upper left", framealpha=0.9)
    cbar = plt.colorbar(sc, ax=ax, pad=0.012); cbar.set_label("|Observed − Nominal|")
    _anchored(ax, f"Mean abs calibration error = {mce:.3f}", loc="lower right")
    base=os.path.join(out_dir,"FigureS6_interval_coverage" if stdnames else "interval_coverage_premium")
    _save_all(fig,base)
    return {"mean_abs_calibration_error": mce,
            "levels": cover_levels.tolist(), "observed": observed.tolist()}

def figS7_heteroscedasticity(y_true,y_pred,out_dir,stdnames,bins=20,cmap="viridis"):
    resid = y_true - y_pred; abs_resid = np.abs(resid)
    nb = int(max(5, bins))
    qs = np.linspace(0, 1, nb+1)
    edges = np.quantile(y_pred, qs); edges = np.unique(edges)
    if edges.size < 4: edges = np.linspace(y_pred.min(), y_pred.max(), nb+1)
    centers = 0.5*(edges[:-1]+edges[1:])
    mean_abs = []; se_abs = []; counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_pred >= lo) & ((y_pred < hi) if hi<edges[-1] else (y_pred <= hi))
        vals = abs_resid[m]
        if vals.size == 0:
            mean_abs.append(np.nan); se_abs.append(np.nan); counts.append(0)
        else:
            mean_abs.append(float(np.mean(vals)))
            se_abs.append(float(_se(vals)))
            counts.append(int(vals.size))
    mean_arr = np.array(mean_abs)
    # color by bin count
    fig, ax = plt.subplots(figsize=golden_figsize(6.0))
    sc = ax.scatter(centers, mean_arr, c=counts, cmap=cmap, s=60, zorder=3)
    ax.errorbar(centers, mean_arr, yerr=se_abs, fmt="none", ecolor="k", elinewidth=0.9, capsize=3, zorder=2, alpha=0.8)
    ax.set_xlabel("Predicted value (quantile centers)"); ax.set_ylabel("Mean ± SE |residual|")
    ax.set_title("Heteroscedasticity by Predicted Quantiles (colored by bin n)")
    cbar = plt.colorbar(sc, ax=ax, pad=0.012); cbar.set_label("Bin count")
    idx = np.arange(mean_arr.size); good = np.isfinite(mean_arr)
    if good.sum() >= 3:
        rho, pval = stats.spearmanr(idx[good], mean_arr[good])
    else:
        rho, pval = np.nan, np.nan
    _anchored(ax, f"Spearman ρ = {rho:.3f} (p={pval:.3g})", loc="upper left")
    base=os.path.join(out_dir,"FigureS7_heteroscedasticity_predq" if stdnames else "heteroscedasticity_profile_premium")
    _save_all(fig,base)
    return {"bins": [{"center": float(c), "count": int(n),
                      "mean_abs_error": (None if not np.isfinite(m) else float(m)),
                      "se": (None if not np.isfinite(s) else float(s))}
                     for c,n,m,s in zip(centers, counts, mean_abs, se_abs)] ,
            "spearman_rho_index_vs_mean_abs_error": (None if not np.isfinite(rho) else float(rho)),
            "spearman_pvalue": (None if not np.isfinite(pval) else float(pval))}

# ---------------- Main ----------------
def main():
    set_premium_style()
    # Locate config.json beside this script (same behavior as your main evaluator)
    cfg_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),"config.json")
    cfg=load_config(cfg_path)

    prj=cfg["project_root"]; exp=cfg.get("experiment_id","default_run")
    exdir=os.path.join(prj,"experiments",exp)
    model_dir=os.path.join(exdir,cfg["output_folders"]["main_models_folder"])
    data_path=os.path.join(prj,cfg["data_folders"]["main_dataset_folder"],cfg["data_folders"]["processed_for_dl_subfolder"])

    eval_params=cfg["evaluation_parameters"]
    model_path=os.path.join(model_dir,eval_params["model_to_evaluate"])
    history_path=os.path.join(model_dir,eval_params["history_to_load"])

    out_root=os.path.join(exdir,"evaluation")
    timestamp=dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir=os.path.join(out_root,f"evalextra_premium_{timestamp}")
    _ensure_dir(out_dir)

    # Load test arrays
    X_test,y_test=load_npz_test(data_path)

    # Load model (with your custom objects if present)
    try:
        import tensorflow as tf
    except Exception as e:
        raise SystemExit(f"[Error] TensorFlow not found. Install requirements and retry. Details: {e}")
    custom_objects={}
    try:
        from s3b_build_model import create_weighted_mse, PositionalEncoding
        if cfg.get("training_parameters",{}).get("advanced_training",{}).get("use_custom_loss",False) and create_weighted_mse is not None:
            loss_instance=create_weighted_mse(cfg["training_parameters"]["advanced_training"]["custom_loss_pos_weight"])
            custom_objects["weighted_mse"]=loss_instance
        custom_objects["PositionalEncoding"]=PositionalEncoding
    except Exception:
        pass

    model=tf.keras.models.load_model(model_path,custom_objects=custom_objects)
    with open(history_path,"r",encoding="utf-8") as f: history=json.load(f)

    # Predict and square to match your evaluation scale
    batch=eval_params.get("prediction_batch_size",1024)
    y_pred_t=model.predict(X_test,batch_size=batch,verbose=1).ravel()
    y_pred=np.square(y_pred_t); y_true=np.square(y_test)

    manifest={"n": int(y_true.size), "files": {"figures": []}, "metrics": {}}

    S3 = figS3_distribution_alignment(y_true,y_pred,out_dir,True,cmap="magma")
    manifest["metrics"]["distribution_alignment"]=S3
    manifest["files"]["figures"] += [os.path.join(out_dir,"FigureS3_distribution_alignment."+ext) for ext in ("png","svg","pdf")]

    S4 = figS4_tolerance_accuracy(y_true,y_pred,out_dir,True,cmap="viridis")
    manifest["metrics"]["tolerance_accuracy"]=S4
    manifest["files"]["figures"] += [os.path.join(out_dir,"FigureS4_tolerance_accuracy."+ext) for ext in ("png","svg","pdf")]

    S5 = figS5_topk_capture(y_true,y_pred,out_dir,True, top_frac=0.10, cmap="plasma")
    manifest["metrics"]["topk_capture"]=S5
    manifest["files"]["figures"] += [os.path.join(out_dir,"FigureS5_topk_capture."+ext) for ext in ("png","svg","pdf")]

    S6 = figS6_interval_coverage(y_true,y_pred,out_dir,True,cmap="cividis")
    manifest["metrics"]["interval_coverage"]=S6
    manifest["files"]["figures"] += [os.path.join(out_dir,"FigureS6_interval_coverage."+ext) for ext in ("png","svg","pdf")]

    S7 = figS7_heteroscedasticity(y_true,y_pred,out_dir,True, bins=20, cmap="viridis")
    manifest["metrics"]["heteroscedasticity_profile"]=S7
    manifest["files"]["figures"] += [os.path.join(out_dir,"FigureS7_heteroscedasticity_predq."+ext) for ext in ("png","svg","pdf")]

    with open(os.path.join(out_dir,"extra_plots_manifest.json"),"w",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2)
    print("[✓] Premium extra plots complete:", out_dir)
    print(json.dumps(manifest, indent=2))

if __name__=="__main__":
    main()
