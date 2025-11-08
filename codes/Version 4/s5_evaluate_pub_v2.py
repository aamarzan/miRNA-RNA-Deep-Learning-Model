#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (Full script content provided in prior cell; rewriting here in full.)

import os, json, time, argparse, hashlib, datetime as dt
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False

import tensorflow as tf

DEFAULT_PALETTES = [
    "viridis","plasma","cividis","magma","inferno","turbo",
    "coolwarm","Spectral","PRGn","PiYG","RdYlBu","cubehelix"
]

def set_base_style():
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 600,
        "font.size": 10, "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 0.9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
        "grid.linewidth": 0.5, "grid.alpha": 0.3
    })
    if HAS_SNS:
        sns.set_style("whitegrid")

def golden_figsize(width_in=3.35):
    phi = (5 ** 0.5 - 1) / 2
    return (width_in, width_in * phi * 1.1)

def two_col_figsize(width_in=7.2):
    phi = (5 ** 0.5 - 1) / 2
    return (width_in, width_in * phi * 1.1)

def save_multi(fig, base, formats):
    out = []
    for ext in formats:
        path = f"{base}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out

def sha1_of_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def load_npz_test(data_path):
    X_test = {}
    files = sorted(os.listdir(data_path))
    test_files = [f for f in files if f.startswith("X_test_") and f.endswith(".npz")]
    if not test_files:
        raise FileNotFoundError(f"No X_test_*.npz in {data_path}")
    for f in test_files:
        key = f.replace("X_test_", "").replace(".npz", "")
        with np.load(os.path.join(data_path, f), mmap_mode="r") as z:
            X_test[key] = z["data"]
    with np.load(os.path.join(data_path, "y_test.npz"), mmap_mode="r") as z:
        y_test = z["data"]
    return X_test, y_test

def get_palettes(palette_list, variants):
    if palette_list is None or len(palette_list)==0:
        palette_list = DEFAULT_PALETTES
    pal_cycle = []
    while len(pal_cycle) < variants:
        pal_cycle += palette_list
    return pal_cycle[:variants]

def density_alpha(x, y, min_alpha=0.3, max_alpha=0.9):
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
        z = (z - z.min()) / (z.max() - z.min() + 1e-12)
        a = max_alpha - (max_alpha - min_alpha) * z
    except Exception:
        a = np.full_like(x, (min_alpha + max_alpha)/2.0, dtype=float)
    return a

def jitter(arr, scale=0.0):
    if scale <= 0: return arr
    return arr + np.random.uniform(-scale, scale, size=arr.shape)

def norm01(v):
    v = np.asarray(v)
    return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)

def choose_color_by(color_by, y_true, y_pred):
    residuals = y_true - y_pred
    import pandas as pd
    if color_by == "density":
        return None, residuals
    if color_by == "residual":
        return residuals, residuals
    if color_by == "abs_residual":
        return np.abs(residuals), residuals
    if color_by == "true":
        return y_true, residuals
    if color_by == "pred":
        return y_pred, residuals
    if color_by == "bin_true":
        return pd.qcut(y_true, q=10, labels=False, duplicates="drop").astype(float), residuals
    if color_by == "bin_pred":
        return pd.qcut(y_pred, q=10, labels=False, duplicates="drop").astype(float), residuals
    if color_by == "bin_residual":
        return pd.qcut(np.abs(residuals), q=10, labels=False, duplicates="drop").astype(float), residuals
    return y_pred, residuals

def pred_vs_obs_variants(y_true, y_pred, r2, pr, out_dir, formats, stdnames, variants, palettes, color_by, size_by, smin, smax, jitter_scale, outline):
    saved = []
    base_name = 'prediction_correlation_density' if not stdnames else 'Figure3a_pred_vs_obs'
    for i in range(variants):
        cmap = plt.get_cmap(palettes[i])
        fig = plt.figure(figsize=two_col_figsize()); ax = fig.add_subplot(111)

        xx = jitter(y_true, jitter_scale); yy = jitter(y_pred, jitter_scale)
        cvals, residuals = choose_color_by(color_by, xx, yy)
        alphas = density_alpha(xx, yy, min_alpha=0.25, max_alpha=0.95)

        if cvals is None:
            dens = gaussian_kde(np.vstack([xx, yy]))(np.vstack([xx, yy]))
            cvals = norm01(dens)

        if size_by == "abs_residual":
            sizes = smin + (smax - smin) * norm01(np.abs(residuals))
        elif size_by == "residual":
            sizes = smin + (smax - smin) * norm01(residuals)
        else:
            sizes = np.full_like(xx, (smin+smax)/2.0)

        ax.scatter(xx, yy, c=cvals, cmap=cmap, s=sizes, alpha=alphas,
                   edgecolors=('k' if outline else 'none'), linewidths=(0.2 if outline else 0.0),
                   rasterized=True)

        ax.plot([0,1],[0,1], ls='--', lw=1.0, c='gray', label='Identity')
        slope, intercept, _, _, _ = stats.linregress(xx, yy)
        xs = np.linspace(0,1,100); ax.plot(xs, intercept + slope*xs, c='black', lw=1.0, label='Linear fit')

        ax.set_xlabel('Observed affinity'); ax.set_ylabel('Predicted affinity')
        ax.set_title(f'Predicted vs Observed (R²={r2:.3f}; r={pr:.3f}) — {palettes[i]}')
        ax.legend(frameon=True)
        base = os.path.join(out_dir, f"{base_name}__pal-{palettes[i]}")
        saved += save_multi(fig, base, formats)
    return saved

def residuals_variants(y_true, y_pred, out_dir, formats, stdnames, variants, palettes, jitter_scale, outline):
    saved = []
    residuals = y_true - y_pred
    base_b = 'residuals_plot' if not stdnames else 'Figure3b_residuals_vs_fitted'
    for i in range(variants):
        cmap = plt.get_cmap(palettes[i])
        fig = plt.figure(figsize=golden_figsize(4.8)); ax = fig.add_subplot(111)
        xp = jitter(y_pred, jitter_scale); r = jitter(residuals, jitter_scale*0.5)
        alphas = density_alpha(xp, r, 0.25, 0.95)
        ax.scatter(xp, r, c=norm01(np.abs(r)), cmap=cmap, s=12, alpha=alphas,
                   edgecolors=('k' if outline else 'none'), linewidths=(0.2 if outline else 0.0),
                   rasterized=True)
        ax.axhline(0, ls='--', c='gray', lw=1)
        if HAS_SM and len(xp)>50:
            low = sm.nonparametric.lowess(r, xp, frac=0.2, return_sorted=False)
            ax.plot(xp, low, c='black', lw=1.2, label='LOWESS')
            ax.legend()
        ax.set_xlabel('Predicted affinity'); ax.set_ylabel('Residual (obs - pred)')
        ax.set_title(f'Residuals vs Predicted — {palettes[i]}')
        saved += save_multi(fig, os.path.join(out_dir, f"{base_b}__pal-{palettes[i]}"), formats)

    base_c = 'residuals_distribution' if not stdnames else 'Figure3c_residual_histogram'
    for i in range(variants):
        cmap = plt.get_cmap(palettes[i])
        fig = plt.figure(figsize=golden_figsize(4.8)); ax = fig.add_subplot(111)
        ax.hist(residuals, bins=60, alpha=0.9, color=cmap(0.6), edgecolor=cmap(0.8))
        mu, sd = np.mean(residuals), np.std(residuals)
        xs = np.linspace(mu-4*sd, mu+4*sd, 400)
        ax.plot(xs, (1/(sd*np.sqrt(2*np.pi)))*np.exp(-0.5*((xs-mu)/sd)**2)*len(residuals)*(xs[1]-xs[0]),
                c='black', lw=1.2, label='Gaussian')
        ax.set_xlabel('Residual'); ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of residuals — {palettes[i]}'); ax.legend()
        saved += save_multi(fig, os.path.join(out_dir, f"{base_c}__pal-{palettes[i]}"), formats)
    return saved

def qq_variants(y_true, y_pred, out_dir, formats, stdnames, variants, palettes):
    saved = []
    residuals = y_true - y_pred
    base = 'qq_plot' if not stdnames else 'Figure3d_qq_plot'
    osm, osr = stats.probplot(residuals, dist="norm")
    for i in range(variants):
        cmap = plt.get_cmap(palettes[i])
        fig = plt.figure(figsize=golden_figsize(4.2)); ax = fig.add_subplot(111)
        theo = osm[0]; ordered = np.sort(residuals)
        ax.scatter(theo, ordered, s=10, c=np.linspace(0,1,len(ordered)), cmap=cmap, alpha=0.9, rasterized=True)
        slope, intercept, r = osr
        ax.plot(theo, slope*theo + intercept, c='black', lw=1.2, label=f'Fit (R={r:.3f})')
        ax.set_title(f'Q–Q plot of residuals — {palettes[i]}')
        ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Ordered residuals')
        ax.legend()
        saved += save_multi(fig, os.path.join(out_dir, f"{base}__pal-{palettes[i]}"), formats)
    return saved

def bland_altman_variants(y_true, y_pred, out_dir, formats, stdnames, variants, palettes, jitter_scale, outline):
    saved = []
    avg = (y_true + y_pred) / 2.0
    diff = y_true - y_pred
    mean_diff = np.mean(diff); sd_diff = np.std(diff)
    loA = mean_diff - 1.96*sd_diff; hiA = mean_diff + 1.96*sd_diff
    frac_out = np.mean((diff < loA) | (diff > hiA))*100.0
    base = 'bland_altman_plot' if not stdnames else 'Figure3e_bland_altman'

    for i in range(variants):
        cmap = plt.get_cmap(palettes[i])
        fig = plt.figure(figsize=golden_figsize(5.0)); ax = fig.add_subplot(111)
        xa = jitter(avg, jitter_scale); df = jitter(diff, jitter_scale*0.5)
        alphas = density_alpha(xa, df, 0.25, 0.95)
        ax.scatter(xa, df, c=norm01(np.abs(df)), cmap=cmap, s=14, alpha=alphas,
                   edgecolors=('k' if outline else 'none'), linewidths=(0.2 if outline else 0.0),
                   rasterized=True)
        ax.axhline(mean_diff, c='black', ls='--', lw=1.0, label=f'Mean diff={mean_diff:.3f}')
        ax.axhline(loA, c='gray', ls='--', lw=1.0, label=f'LoA={loA:.3f}')
        ax.axhline(hiA, c='gray', ls='--', lw=1.0, label=f'LoA={hiA:.3f}')
        ax.fill_between([xa.min(), xa.max()], loA, hiA, color='gray', alpha=0.12)
        ax.set_xlabel('Average of observed & predicted'); ax.set_ylabel('Difference (obs - pred)')
        ax.set_title(f'Bland–Altman (outside LoA: {frac_out:.1f}%) — {palettes[i]}')
        ax.legend()
        saved += save_multi(fig, os.path.join(out_dir, f"{base}__pal-{palettes[i]}"), formats)
    return saved

def error_by_bin_variants(y_true, y_pred, out_dir, formats, stdnames, variants, palettes):
    saved = []
    abs_err = np.abs(y_true - y_pred)
    bins = np.arange(0,1.00001,0.1); labels = [f"[{bins[b]:.1f},{bins[b+1]:.1f})" for b in range(len(bins)-1)]
    data = [abs_err[(y_true>=bins[b]) & (y_true<bins[b+1])] for b in range(10)]
    base = 'error_by_affinity_bin' if not stdnames else 'FigureS2_error_by_affinity_bin'
    for i in range(variants):
        cmap = plt.get_cmap(palettes[i])
        fig = plt.figure(figsize=two_col_figsize(6.6)); ax = fig.add_subplot(111)
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(cmap(j/10.0))
            patch.set_edgecolor('black'); patch.set_alpha(0.9)
        for whisk in bp['whiskers']: whisk.set_color('black')
        for cap in bp['caps']: cap.set_color('black')
        for med in bp['medians']: med.set_color('black'); med.set_linewidth(1.2)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('True affinity bin'); ax.set_ylabel('Absolute error')
        ax.set_title(f'Absolute prediction error by true-affinity bins — {palettes[i]}')
        saved += save_multi(fig, os.path.join(out_dir, f"{base}__pal-{palettes[i]}"), formats)
    return saved

def reliability_variants(y_true, y_pred, out_dir, formats, stdnames, variants, palettes, n_bins=15):
    saved = []; stats_out = None
    base = 'reliability_diagram' if not stdnames else 'Figure4c_reliability_diagram'
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(y_pred, bins) - 1
    centers = (bins[:-1] + bins[1:]) / 2.0
    mp = np.zeros(n_bins); mt = np.zeros(n_bins); counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx==b
        if np.any(m):
            mp[b] = np.mean(y_pred[m]); mt[b] = np.mean(y_true[m]); counts[b] = m.sum()
        else:
            mp[b] = np.nan; mt[b] = np.nan
    valid = ~np.isnan(mp)
    weights = counts[valid]/counts[valid].sum() if counts[valid].sum()>0 else np.ones(valid.sum())/valid.sum()
    ece = np.sum(weights*np.abs(mp[valid]-mt[valid])); mce = np.nanmax(np.abs(mp[valid]-mt[valid])) if valid.any() else np.nan
    stats_out = {"ECE": float(ece), "MCE": float(mce)}

    for i in range(variants):
        cmap = plt.get_cmap(palettes[i])
        fig = plt.figure(figsize=golden_figsize(5.0)); ax = fig.add_subplot(111)
        ax.plot([0,1],[0,1], ls='--', c='gray', lw=1.0)
        ax.plot(mp, mt, marker='o', ls='-', lw=1.2, c=cmap(0.6), label='bin means')
        for x,y,c in zip(mp, mt, counts):
            if not np.isnan(x) and not np.isnan(y):
                ax.annotate(str(int(c)), (x,y), textcoords="offset points", xytext=(3,3), fontsize=7, alpha=0.7)
        ax.set_xlabel('Predicted affinity (bin mean)')
        ax.set_ylabel('Observed affinity (bin mean)')
        ax.set_title(f'Reliability (ECE={ece:.3f}, MCE={mce:.3f}) — {palettes[i]}'); ax.legend()
        saved += save_multi(fig, os.path.join(out_dir, f"{base}__pal-{palettes[i]}"), formats)
    return saved, stats_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--formats", type=str, default="png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stdnames", action="store_true")
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--palette-list", type=str, default=None)
    ap.add_argument("--color-by", type=str, default="density", choices=["density","residual","abs_residual","true","pred","bin_true","bin_pred","bin_residual"])
    ap.add_argument("--size-by", type=str, default="abs_residual", choices=["none","residual","abs_residual"])
    ap.add_argument("--point-size", type=float, nargs=2, default=[6.0,16.0])
    ap.add_argument("--jitter", type=float, default=0.001)
    ap.add_argument("--outline", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    set_base_style(); mpl.rcParams["savefig.dpi"] = args.dpi

    if args.config is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        args.config = os.path.join(script_dir, "config.json")
    cfg = load_config(args.config)

    prj = cfg["project_root"]; exp = cfg.get("experiment_id","default_run")
    exdir = os.path.join(prj, "experiments", exp)
    model_dir = os.path.join(exdir, cfg["output_folders"]["main_models_folder"])
    data_path = os.path.join(prj, cfg["data_folders"]["main_dataset_folder"], cfg["data_folders"]["processed_for_dl_subfolder"])

    eval_params = cfg["evaluation_parameters"]; train_params = cfg["training_parameters"]
    model_path = os.path.join(model_dir, eval_params["model_to_evaluate"])
    history_path = os.path.join(model_dir, eval_params["history_to_load"])

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_root = os.path.join(exdir, "evaluation")
    out_dir = args.out if args.out else os.path.join(out_root, f"evalpal_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    X_test, y_test = load_npz_test(data_path)
    custom_objects = {}
    try:
        from s3b_build_model import create_weighted_mse, PositionalEncoding
        if train_params.get("advanced_training",{}).get("use_custom_loss",False) and create_weighted_mse is not None:
            loss_instance = create_weighted_mse(train_params["advanced_training"]["custom_loss_pos_weight"])
            custom_objects["weighted_mse"] = loss_instance
        custom_objects["PositionalEncoding"] = PositionalEncoding
    except Exception:
        pass
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    with open(history_path, "r") as f: history = json.load(f)

    batch = eval_params.get("prediction_batch_size", 1024)
    y_pred_t = model.predict(X_test, batch_size=batch, verbose=1).ravel()
    y_pred = np.square(y_pred_t); y_true = np.square(y_test)

    r2 = r2_score(y_true,y_pred); pr,_ = pearsonr(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred); mae = mean_absolute_error(y_true,y_pred)
    pd.DataFrame({"Metric":["R2","Pearson r","MSE","MAE"],"Value":[r2,pr,mse,mae]}).to_csv(
        os.path.join(out_dir,"performance_metrics.csv"), index=False, float_format="%.6f"
    )

    palettes = get_palettes([p.strip() for p in args.palette_list.split(",")] if args.palette_list else None, args.variants)
    formats = [f.strip() for f in args.formats.split(",") if f.strip() in ("png","pdf","svg")]

    saved = []
    saved += pred_vs_obs_variants(y_true, y_pred, r2, pr, out_dir, formats, args.stdnames,
                                  args.variants, palettes, args.color_by, args.size_by,
                                  args.point_size[0], args.point_size[1], args.jitter, args.outline)

    # Training history (variants for line colors)
    for i in range(args.variants):
        cmap = plt.get_cmap(palettes[i])
        fig = plt.figure(figsize=golden_figsize(4.6)); ax=fig.add_subplot(111)
        ax.plot(history.get('loss',[]), label='Training', lw=1.8, color=cmap(0.2))
        ax.plot(history.get('val_loss',[]), label='Validation', lw=1.8, color=cmap(0.8))
        ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
        ax.set_title(f'Model loss over epochs — {palettes[i]}'); ax.legend()
        base = 'training_history' if not args.stdnames else 'FigureS1_training_history'
        saved += save_multi(fig, os.path.join(out_dir, f"{base}__pal-{palettes[i]}"), formats)

    saved += residuals_variants(y_true, y_pred, out_dir, formats, args.stdnames, args.variants, palettes, args.jitter, args.outline)
    saved += qq_variants(y_true, y_pred, out_dir, formats, args.stdnames, args.variants, palettes)
    saved += bland_altman_variants(y_true, y_pred, out_dir, formats, args.stdnames, args.variants, palettes, args.jitter, args.outline)
    saved += error_by_bin_variants(y_true, y_pred, out_dir, formats, args.stdnames, args.variants, palettes)
    rel_paths, rel_stats = reliability_variants(y_true, y_pred, out_dir, formats, args.stdnames, args.variants, palettes, n_bins=15)
    saved += rel_paths

    meta = {
        "timestamp": timestamp,
        "config_path": os.path.abspath(args.config),
        "config_sha1": sha1_of_file(args.config) if os.path.exists(args.config) else None,
        "model_path": os.path.abspath(model_path),
        "history_path": os.path.abspath(history_path),
        "data_path": os.path.abspath(data_path),
        "stdnames": bool(args.stdnames),
        "variants": args.variants,
        "palettes": palettes
    }
    manifest = {"meta": meta, "metrics": {"R2": float(r2), "pearson_r": float(pr), "MSE": float(mse), "MAE": float(mae), **rel_stats}, "files": {"figures": saved}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("[✓] Complete. Outputs:", out_dir)

if __name__ == "__main__":
    main()
