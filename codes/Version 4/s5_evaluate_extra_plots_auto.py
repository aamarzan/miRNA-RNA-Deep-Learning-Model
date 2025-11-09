#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s5_evaluate_extra_plots_auto.py
Author: ChatGPT (auto-integrated with your project layout)
Purpose: Generate *additional*, model‑driven evaluation figures that complement your existing set,
         without any extra inputs. It auto‑reads config.json, loads model + test data, predicts,
         squares to match your evaluation scale, and saves high‑quality PNG/SVG/PDF + a JSON manifest.

New plot set (unique vs your current scatter/residual/QQ/BA/reliability):
  S3) Distribution Alignment (KDE + ECDF)  — JS (base 2), Wasserstein‑1, KS D/p
  S4) Tolerance–Accuracy Curve (TAC)       — AUTA, Acc@{5,10,20,50,100}% of δ₉₅
  S5) Top‑k Capture Curve                  — AUCC, P@{1%,5%,10%}
  S6) Prediction Interval Coverage (PICC)  — mean abs calibration error
  S7) Heteroscedasticity by Pred‑quantiles — Spearman trend of mean |resid| across bins

Outputs:
  - Figures in PNG, SVG, PDF
  - extra_plots_manifest.json (summary metrics)

Assumptions:
  - Same config.json structure you already use in s5_evaluate_pub_v8.py
  - Test data: X_test_*.npz (key "data"), y_test.npz (key "data") in processed_for_dl_subfolder
  - Model: loaded via tf.keras, with optional custom objects (weighted_mse, PositionalEncoding)
"""
import os, json, argparse, datetime as dt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------- style helpers to match your plotting aesthetic ----------------
def set_base_style(dpi=600):
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": dpi,
        "font.size": 10, "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 0.9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
        "grid.linewidth": 0.5, "grid.alpha": 0.28
    })

def golden_figsize(w=3.35):
    phi=(5**0.5-1)/2
    return (w, w*phi*1.12)

def two_col_figsize(w=7.1):
    phi=(5**0.5-1)/2
    return (w, w*phi*1.12)

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _save_all(fig, outbase):
    for ext in ("png","svg","pdf"):
        fig.savefig(f"{outbase}.{ext}", bbox_inches="tight")
    plt.close(fig)

def _anchored(ax, text, loc="upper right", fontsize=8.2, alpha=0.92):
    box = AnchoredText(text, loc=loc, prop=dict(size=fontsize), frameon=True, pad=0.3, borderpad=0.6)
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

def norm01(v):
    v=np.asarray(v); return (v-np.nanmin(v))/(np.nanmax(v)-np.nanmin(v)+1e-12)

# ---------------- project‑specific loaders (mirrors your s5_evaluate_pub_v8.py) ----------------
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

# ---------------- figures (NEW) ----------------
def figS3_dist_alignment(y_true,y_pred,out_dir,stdnames):
    grid = np.linspace(np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()]), 512)
    p = _gauss_kde(y_true, grid); q = _gauss_kde(y_pred, grid)
    p = p/(p.sum()+1e-12); q = q/(q.sum()+1e-12)
    js = _js_divergence(p, q, base=2)
    w1 = stats.wasserstein_distance(y_true, y_pred)
    ks_D, ks_p = stats.ks_2samp(y_true, y_pred)

    fig, ax = plt.subplots(figsize=golden_figsize(5.0))
    ax.plot(grid, p, label="True (KDE)")
    ax.plot(grid, q, label="Pred (KDE)", linestyle="--")
    ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.set_title("Distribution Alignment (KDE)")

    ax2 = ax.twinx()
    xt, yt = _ecdf(y_true); xp, yp = _ecdf(y_pred)
    ax2.step(xt, yt, where="post", alpha=0.35, label="True ECDF")
    ax2.step(xp, yp, where="post", alpha=0.35, linestyle="--", label="Pred ECDF")
    ax2.set_ylabel("ECDF")
    _anchored(ax, f"JS (base 2) = {js:.3f}\nWasserstein = {w1:.3f}\nKS D = {ks_D:.3f} (p={ks_p:.3g})", loc="upper right")
    lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="lower right", framealpha=0.9)

    base=os.path.join(out_dir,"FigureS3_distribution_alignment" if stdnames else "dist_alignment")
    _save_all(fig,base)
    return {"JS_divergence_base2": float(js), "Wasserstein_distance": float(w1), "KS_D": float(ks_D), "KS_pvalue": float(ks_p)}

def figS4_tolerance_accuracy(y_true,y_pred,out_dir,stdnames):
    resid = y_true - y_pred; abs_resid = np.abs(resid)
    delta_max = np.percentile(abs_resid, 95.0)
    deltas = np.linspace(0, delta_max, 200)
    acc = np.array([(np.mean(abs_resid <= d) if d>0 else np.mean(abs_resid == 0)) for d in deltas])
    auta = np.trapz(acc, deltas) / (delta_max + 1e-12)
    marks = [0.05, 0.1, 0.2, 0.5, 1.0]
    acc_marks = {f"Acc@{int(m*100)}%Δmax": float(np.mean(abs_resid <= (m*delta_max))) for m in marks}

    fig, ax = plt.subplots(figsize=golden_figsize(5.0))
    ax.plot(deltas, acc, lw=2)
    ax.set_xlabel("Tolerance δ"); ax.set_ylabel("Accuracy  P(|error| ≤ δ)"); ax.set_title("Tolerance–Accuracy Curve")
    for m in marks: ax.axvline(m*delta_max, ls=":", lw=0.8, alpha=0.6)
    _anchored(ax, "AUTA = {:.3f}\nδ₉₅ = {:.3f}\n{}".format(auta, delta_max, 
                   "\n".join([f"{k} = {v:.3f}" for k,v in acc_marks.items()])), loc="lower right")
    base=os.path.join(out_dir,"FigureS4_tolerance_accuracy" if stdnames else "tolerance_accuracy")
    _save_all(fig,base)
    return {"AUTA": float(auta), "delta_max": float(delta_max), **acc_marks}

def figS5_topk_capture(y_true,y_pred,out_dir,stdnames, top_frac=0.10):
    T = float(top_frac); n = y_true.size; k_true = max(1, int(np.round(T * n)))
    idx_true_top = np.argsort(y_true)[-k_true:]
    true_top_mask = np.zeros(n, dtype=bool); true_top_mask[idx_true_top] = True
    ss = np.linspace(0.01, 1.0, 200); recall = []
    for s in ss:
        k_sel = max(1, int(np.round(s * n)))
        idx_pred_top = np.argsort(y_pred)[-k_sel:]
        sel_mask = np.zeros(n, dtype=bool); sel_mask[idx_pred_top] = True
        recall.append(np.sum(true_top_mask & sel_mask) / k_true)
    recall = np.array(recall); aucc = np.trapz(recall, ss)
    prec_at = {}
    for frac in (0.01, 0.05, 0.10):
        k_sel = max(1, int(np.round(frac * n)))
        idx_pred_top = np.argsort(y_pred)[-k_sel:]
        prec_at[f"P@{int(frac*100)}%"] = float(np.sum(true_top_mask[idx_pred_top]) / k_sel)

    fig, ax = plt.subplots(figsize=golden_figsize(5.0))
    ax.plot(ss, recall, lw=2); ax.plot([0,1],[0,1], ls="--", lw=1, alpha=0.6, label="Random baseline")
    ax.set_xlabel("Fraction selected by predicted rank"); ax.set_ylabel(f"Recall of true top {int(T*100)}%")
    ax.set_title("Top‑k Capture Curve"); ax.legend(loc="lower right", framealpha=0.9)
    _anchored(ax, "AUCC = {:.3f}\n{}".format(aucc, "\n".join([f"{k} = {v:.3f}" for k,v in prec_at.items()])), loc="upper left")
    base=os.path.join(out_dir,"FigureS5_topk_capture" if stdnames else "topk_capture")
    _save_all(fig,base)
    return {"true_top_frac": T, "AUCC": float(aucc), **prec_at}

def figS6_interval_coverage(y_true,y_pred,out_dir,stdnames):
    resid = y_true - y_pred; abs_resid = np.abs(resid)
    cover_levels = np.linspace(0.50, 0.99, 20)
    observed = []
    for c in cover_levels:
        q = np.quantile(abs_resid, c)
        lo = y_pred - q; hi = y_pred + q
        observed.append(np.mean((y_true >= lo) & (y_true <= hi)))
    observed = np.array(observed); mce = float(np.mean(np.abs(observed - cover_levels)))

    fig, ax = plt.subplots(figsize=golden_figsize(5.0))
    ax.plot(cover_levels, observed, marker="o", lw=1.5)
    ax.plot([0.5,0.99],[0.5,0.99], ls="--", lw=1, alpha=0.6, label="Ideal")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage")
    ax.set_title("Prediction Interval Coverage (Residual‑Quantile)")
    ax.set_xlim(0.5,0.99); ax.set_ylim(0.5,0.99); ax.legend(loc="upper left", framealpha=0.9)
    _anchored(ax, f"Mean abs calibration error = {mce:.3f}", loc="lower right")
    base=os.path.join(out_dir,"FigureS6_interval_coverage" if stdnames else "interval_coverage")
    _save_all(fig,base)
    return {"mean_abs_calibration_error": mce, "levels": cover_levels.tolist(), "observed": observed.tolist()}

def figS7_heteroscedasticity(y_true,y_pred,out_dir,stdnames,bins=20):
    resid = y_true - y_pred; abs_resid = np.abs(resid)
    nb = int(max(5, bins))
    qs = np.linspace(0, 1, nb+1)
    edges = np.quantile(y_pred, qs)
    edges = np.unique(edges)
    if edges.size < 4:
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
    mean_arr = np.array(mean_abs); idx = np.arange(mean_arr.size); good = np.isfinite(mean_arr)
    if good.sum() >= 3:
        rho, pval = stats.spearmanr(idx[good], mean_arr[good])
    else:
        rho, pval = np.nan, np.nan

    fig, ax = plt.subplots(figsize=golden_figsize(5.0))
    ax.errorbar(centers, mean_arr, yerr=se_abs, fmt="o-", capsize=3)
    ax.set_xlabel("Predicted value (quantile centers)"); ax.set_ylabel("Mean ± SE |residual|")
    ax.set_title("Heteroscedasticity by Predicted Quantiles")
    _anchored(ax, f"Spearman ρ = {rho:.3f} (p={pval:.3g})", loc="upper left")
    base=os.path.join(out_dir,"FigureS7_heteroscedasticity_predq" if stdnames else "heteroscedasticity_profile")
    _save_all(fig,base)

    return {
        "bins": [{"center": float(c), "count": int(n),
                  "mean_abs_error": (None if not np.isfinite(m) else float(m)),
                  "se": (None if not np.isfinite(s) else float(s))}
                 for c,n,m,s in zip(centers, counts, mean_abs, se_abs)],
        "spearman_rho_index_vs_mean_abs_error": (None if not np.isfinite(rho) else float(rho)),
        "spearman_pvalue": (None if not np.isfinite(pval) else float(pval))
    }

# ---------------- main (auto integrates with your config + files) ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=str,default=None, help="Defaults to <script_dir>/config.json")
    ap.add_argument("--out",type=str,default=None, help="Optional output folder; defaults under experiments/<exp>/evaluation/")
    ap.add_argument("--formats",type=str,default="png,pdf,svg")
    ap.add_argument("--dpi",type=int,default=600)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--stdnames",action="store_true",default=True)
    ap.add_argument("--top-frac",type=float,default=0.10)
    ap.add_argument("--bins",type=int,default=20)
    args=ap.parse_args()

    np.random.seed(args.seed); set_base_style(args.dpi)
    # Locate config.json like your main script
    if args.config is None:
        args.config=os.path.join(os.path.dirname(os.path.realpath(__file__)),"config.json")
    cfg=load_config(args.config)

    prj=cfg["project_root"]; exp=cfg.get("experiment_id","default_run")
    exdir=os.path.join(prj,"experiments",exp)
    model_dir=os.path.join(exdir,cfg["output_folders"]["main_models_folder"])
    data_path=os.path.join(prj,cfg["data_folders"]["main_dataset_folder"],cfg["data_folders"]["processed_for_dl_subfolder"])

    eval_params=cfg["evaluation_parameters"]
    model_path=os.path.join(model_dir,eval_params["model_to_evaluate"])
    history_path=os.path.join(model_dir,eval_params["history_to_load"])

    out_root=os.path.join(exdir,"evaluation")
    timestamp=dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir=args.out if args.out else os.path.join(out_root,f"evalextra_{timestamp}")
    _ensure_dir(out_dir)

    # Load test arrays
    X_test,y_test=load_npz_test(data_path)

    # Load model (with optional custom objects, mirroring your script)
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

    # Predict and square (to match your evaluation scale)
    batch=eval_params.get("prediction_batch_size",1024)
    y_pred_t=model.predict(X_test,batch_size=batch,verbose=1).ravel()
    y_pred=np.square(y_pred_t); y_true=np.square(y_test)

    # --------------- Generate NEW plots ---------------
    formats=[f.strip() for f in args.formats.split(",") if f.strip() in ("png","pdf","svg")]
    # (respect formats)
    mpl.rcParams["savefig.dpi"]=args.dpi

    manifest={"n": int(y_true.size), "files": {"figures": []}, "metrics": {}}

    # S3 Distribution Alignment
    S3 = figS3_dist_alignment(y_true,y_pred,out_dir,args.stdnames)
    manifest["metrics"]["distribution_alignment"]=S3
    manifest["files"]["figures"] += [
        os.path.join(out_dir, ("FigureS3_distribution_alignment" if args.stdnames else "dist_alignment") + f".{ext}") for ext in formats
    ]

    # S4 Tolerance–Accuracy
    S4 = figS4_tolerance_accuracy(y_true,y_pred,out_dir,args.stdnames)
    manifest["metrics"]["tolerance_accuracy"]=S4
    manifest["files"]["figures"] += [
        os.path.join(out_dir, ("FigureS4_tolerance_accuracy" if args.stdnames else "tolerance_accuracy") + f".{ext}") for ext in formats
    ]

    # S5 Top‑k Capture
    S5 = figS5_topk_capture(y_true,y_pred,out_dir,args.stdnames, top_frac=args.top_frac)
    manifest["metrics"]["topk_capture"]=S5
    manifest["files"]["figures"] += [
        os.path.join(out_dir, ("FigureS5_topk_capture" if args.stdnames else "topk_capture") + f".{ext}") for ext in formats
    ]

    # S6 Interval Coverage
    S6 = figS6_interval_coverage(y_true,y_pred,out_dir,args.stdnames)
    manifest["metrics"]["interval_coverage"]=S6
    manifest["files"]["figures"] += [
        os.path.join(out_dir, ("FigureS6_interval_coverage" if args.stdnames else "interval_coverage") + f".{ext}") for ext in formats
    ]

    # S7 Heteroscedasticity
    S7 = figS7_heteroscedasticity(y_true,y_pred,out_dir,args.stdnames, bins=args.bins)
    manifest["metrics"]["heteroscedasticity_profile"]=S7
    manifest["files"]["figures"] += [
        os.path.join(out_dir, ("FigureS7_heteroscedasticity_predq" if args.stdnames else "heteroscedasticity_profile") + f".{ext}") for ext in formats
    ]

    # Write manifest (separate from your main manifest.json)
    man_path=os.path.join(out_dir,"extra_plots_manifest.json")
    with open(man_path,"w",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2)
    print("[✓] Extra plots complete:", out_dir)
    print(json.dumps(manifest, indent=2))

if __name__=="__main__":
    main()
