#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s5_evaluate_pub_v4.py — figure-specific palettes, richer stats, dual variants, polished aesthetics.

Key changes vs v3:
- Adds statistical annotations per figure:
  * 3a: N, R², Pearson r (p, 95% CI), Spearman ρ (p), slope±CI, MAE, RMSE, CCC
  * 3b: mean±SD residual, Spearman(pred,resid) (p), JB p-value
  * 3c: skew, kurtosis, JB p-value  (two palettes: Spectral AND coolwarm)
  * 3d: QQ-fit R, normality p (Shapiro if n≤5000 else JB)  (two palettes: magma AND inferno)
  * 3e: mean diff, LoA, % outside, slope and Pearson(avg,diff) with p; no shaded band (per request)
  * 4c: ECE, MCE, Spearman(mp,mt), N bins, bubble key
  * S1: best val_loss & epoch
  * S2: palette switched to coolwarm + vertical colorbar

- 3b LOWESS line uses a premium color ("royalblue") instead of black.
- 3e removes the LoA shading (no fill), cleaner lines.
"""

import os, json, argparse, hashlib, datetime as dt
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, gaussian_kde, jarque_bera, shapiro, linregress
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

# --------------------- Styling ---------------------

def set_base_style(dpi=600):
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": dpi,
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
        p = f"{base}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out

# --------------------- IO utils ---------------------

def sha1_of_file(path):
    import hashlib
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

# --------------------- Helpers ---------------------

def density_alpha(x, y, min_alpha=0.25, max_alpha=0.95):
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

def concordance_cc(y_true, y_pred):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    mu1, mu2 = np.mean(y_true), np.mean(y_pred)
    v1, v2 = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu1)*(y_pred - mu2))
    return (2 * cov) / (v1 + v2 + (mu1 - mu2)**2 + 1e-12)

def fisher_ci_r(r, n, alpha=0.05):
    if n < 4: return (np.nan, np.nan)
    z = np.arctanh(max(-0.9999999, min(0.9999999, r)))
    se = 1/np.sqrt(n-3)
    zcrit = stats.norm.ppf(1-alpha/2)
    lo = np.tanh(z - zcrit*se); hi = np.tanh(z + zcrit*se)
    return lo, hi

# --------------------- Figure builders ---------------------

def fig3a_pred_vs_obs(y_true, y_pred, out_dir, formats, stdnames, jitter_scale=0.001, outline=True):
    n = len(y_true)
    R2 = r2_score(y_true, y_pred)
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    slope, intercept, rvl, p_slope, slope_se = linregress(y_true, y_pred)
    lo_s = slope - 1.96*slope_se; hi_s = slope + 1.96*slope_se
    MSE = mean_squared_error(y_true, y_pred); RMSE = np.sqrt(MSE); MAE = mean_absolute_error(y_true, y_pred)
    CCC = concordance_cc(y_true, y_pred)
    pr_lo, pr_hi = fisher_ci_r(pr, n)

    cmap = plt.get_cmap("Spectral")
    fig = plt.figure(figsize=two_col_figsize()); ax = fig.add_subplot(111)
    xx = jitter(y_true, jitter_scale); yy = jitter(y_pred, jitter_scale)
    dens = gaussian_kde(np.vstack([xx, yy]))(np.vstack([xx, yy]))
    cvals = norm01(dens); alphas = density_alpha(xx, yy)
    sc = ax.scatter(xx, yy, c=cvals, cmap=cmap, s=12, alpha=alphas,
                    edgecolors=('k' if outline else 'none'), linewidths=(0.2 if outline else 0.0),
                    rasterized=True)
    ax.plot([0,1],[0,1], ls='--', lw=1.0, c='gray', label='Identity')
    xs = np.linspace(0,1,100); ax.plot(xs, intercept + slope*xs, c='black', lw=1.0, label='Linear fit')
    ax.set_xlabel('Observed affinity'); ax.set_ylabel('Predicted affinity')
    ax.set_title('Predicted vs Observed')

    # stats box (right)
    txt = (f"N={n}\n"
           f"R²={R2:.3f}\n"
           f"r={pr:.3f} (p={pp:.3g}; 95% CI {pr_lo:.3f}–{pr_hi:.3f})\n"
           f"ρ={sr:.3f} (p={sp:.3g})\n"
           f"slope={slope:.3f} [{lo_s:.3f},{hi_s:.3f}]; intercept={intercept:.3f}\n"
           f"MAE={MAE:.4f}; RMSE={RMSE:.4f}; CCC={CCC:.3f}")
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, va='center', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

    ax.legend(frameon=True, loc='upper left')
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical'); cbar.set_label('Relative point density (KDE)')
    base = os.path.join(out_dir, 'Figure3a_pred_vs_obs' if stdnames else 'prediction_correlation_density')
    return save_multi(fig, base, formats)

def fig3b_residuals_vs_fitted(y_true, y_pred, out_dir, formats, stdnames, jitter_scale=0.001, outline=True):
    cmap = plt.get_cmap("turbo")
    residuals = y_true - y_pred
    n = len(residuals)
    mr, sr = float(np.mean(residuals)), float(np.std(residuals, ddof=1))
    rho, rho_p = spearmanr(y_pred, residuals)
    jb_stat, jb_p = jarque_bera(residuals)

    fig = plt.figure(figsize=golden_figsize(4.9)); ax = fig.add_subplot(111)
    xp = jitter(y_pred, jitter_scale); r = jitter(residuals, jitter_scale*0.5)
    alphas = density_alpha(xp, r)
    sc = ax.scatter(xp, r, c=norm01(np.abs(r)), cmap=cmap, s=14, alpha=alphas,
                    edgecolors=('k' if outline else 'none'), linewidths=(0.2 if outline else 0.0),
                    rasterized=True)
    ax.axhline(0, ls='--', c='gray', lw=1)
    if HAS_SM and len(xp)>50:
        low = sm.nonparametric.lowess(r, xp, frac=0.2, return_sorted=False)
        ax.plot(xp, low, c='royalblue', lw=1.5, label='LOWESS')
        ax.legend()
    ax.set_xlabel('Predicted affinity'); ax.set_ylabel('Residual (obs - pred)')
    ax.set_title('Residuals vs Predicted')

    txt = (f"N={n}\n"
           f"mean={mr:.4f}; SD={sr:.4f}\n"
           f"ρ(pred,resid)={rho:.3f} (p={rho_p:.3g})\n"
           f"JB p={jb_p:.3g}")
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, va='center', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical'); cbar.set_label('Absolute residual')
    base = os.path.join(out_dir, 'Figure3b_residuals_vs_fitted' if stdnames else 'residuals_plot')
    return save_multi(fig, base, formats)

def fig3c_residual_hist_dual(y_true, y_pred, out_dir, formats, stdnames):
    residuals = y_true - y_pred
    skew = stats.skew(residuals, nan_policy='omit')
    kurt = stats.kurtosis(residuals, nan_policy='omit')
    jb_stat, jb_p = jarque_bera(residuals)
    counts, bins = np.histogram(residuals, bins=60)

    outputs = []
    for pal in ["Spectral", "coolwarm"]:
        cmap = plt.get_cmap(pal)
        fig = plt.figure(figsize=golden_figsize(4.9)); ax = fig.add_subplot(111)
        counts, bins, patches = ax.hist(residuals, bins=60, edgecolor='none')
        cmin, cmax = counts.min(), counts.max()
        for c, p in zip(counts, patches):
            if cmax>cmin:
                p.set_facecolor(cmap((c - cmin)/(cmax - cmin)))
            else:
                p.set_facecolor(cmap(0.5))
        ax.set_xlabel('Residual'); ax.set_ylabel('Frequency (count)')
        ax.set_title('Distribution of residuals')
        smap = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax))
        smap.set_array([])
        cbar = plt.colorbar(smap, ax=ax, orientation='vertical'); cbar.set_label('Bin count')

        txt = (f"Skew={skew:.3f}; Kurt={kurt:.3f}\nJB p={jb_p:.3g}")
        ax.text(1.02, 0.7, txt, transform=ax.transAxes, va='center', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

        base = os.path.join(out_dir, f"Figure3c_residual_histogram__pal-{pal}" if stdnames else f"residuals_distribution__pal-{pal}")
        outputs += save_multi(fig, base, formats)
    return outputs

def fig3d_qq_dual(y_true, y_pred, out_dir, formats, stdnames):
    residuals = y_true - y_pred
    n = len(residuals)
    if n <= 5000:
        try:
            shp_stat, shp_p = shapiro(residuals)
            norm_txt = f"Shapiro p={shp_p:.3g}"
        except Exception:
            jb_stat, jb_p = jarque_bera(residuals)
            norm_txt = f"JB p={jb_p:.3g}"
    else:
        jb_stat, jb_p = jarque_bera(residuals)
        norm_txt = f"JB p={jb_p:.3g}"

    outputs = []
    for pal in ["magma", "inferno"]:
        cmap = plt.get_cmap(pal)
        osm, osr = stats.probplot(residuals, dist="norm")
        theo = osm[0]; ordered = np.sort(residuals)
        fig = plt.figure(figsize=golden_figsize(4.5)); ax = fig.add_subplot(111)
        sc = ax.scatter(theo, ordered, c=norm01(ordered), cmap=cmap, s=12, alpha=0.9, rasterized=True)
        slope, intercept, r = osr
        ax.plot(theo, slope*theo + intercept, c='black', lw=1.2, label=f'Fit (R={r:.3f})')
        ax.set_title('Q–Q plot of residuals')
        ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Ordered residuals')
        ax.legend()

        ax.text(1.02, 0.7, norm_txt, transform=ax.transAxes, va='center', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

        cbar = plt.colorbar(sc, ax=ax, orientation='vertical'); cbar.set_label('Residual value')
        base = os.path.join(out_dir, f"Figure3d_qq_plot__pal-{pal}" if stdnames else f"qq_plot__pal-{pal}")
        outputs += save_multi(fig, base, formats)
    return outputs

def fig3e_bland_altman(y_true, y_pred, out_dir, formats, stdnames, jitter_scale=0.001, outline=True):
    cmap = plt.get_cmap("turbo")
    avg = (y_true + y_pred) / 2.0
    diff = y_true - y_pred
    mean_diff = np.mean(diff); sd_diff = np.std(diff)
    loA = mean_diff - 1.96*sd_diff; hiA = mean_diff + 1.96*sd_diff
    frac_out = np.mean((diff < loA) | (diff > hiA))*100.0

    # proportional bias test
    pr, pp = pearsonr(avg, diff)
    slope, intercept, rvl, p_slope, slope_se = linregress(avg, diff)

    fig = plt.figure(figsize=golden_figsize(5.2)); ax = fig.add_subplot(111)
    xa = jitter(avg, jitter_scale); df = jitter(diff, jitter_scale*0.5)
    alphas = density_alpha(xa, df)
    sc = ax.scatter(xa, df, c=norm01(np.abs(df)), cmap=cmap, s=16, alpha=alphas,
                    edgecolors=('k' if outline else 'none'), linewidths=(0.2 if outline else 0.0),
                    rasterized=True)
    ax.axhline(mean_diff, c='black', ls='--', lw=1.0, label=f'Mean diff={mean_diff:.3f}')
    ax.axhline(loA, c='gray', ls='--', lw=1.0, label=f'LoA={loA:.3f}')
    ax.axhline(hiA, c='gray', ls='--', lw=1.0, label=f'LoA={hiA:.3f}')
    # Removed shaded band per request
    ax.set_xlabel('Average of observed & predicted'); ax.set_ylabel('Difference (obs - pred)')
    ax.set_title('Bland–Altman (no shading)')
    ax.legend(loc='upper right')

    txt = (f"Mean diff={mean_diff:.4f}; LoA [{loA:.4f},{hiA:.4f}]\n"
           f"Outside LoA={frac_out:.1f}%\n"
           f"slope={slope:.3f} (p={p_slope:.3g}); r(avg,diff)={pr:.3f} (p={pp:.3g})")
    ax.text(1.02, 0.55, txt, transform=ax.transAxes, va='center', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical'); cbar.set_label('Absolute difference')
    base = os.path.join(out_dir, 'Figure3e_bland_altman' if stdnames else 'bland_altman_plot')
    return save_multi(fig, base, formats)

def fig4c_reliability(y_true, y_pred, out_dir, formats, stdnames, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(y_pred, bins) - 1
    centers = (bins[:-1] + bins[1:]) / 2.0
    mp = np.zeros(n_bins); mt = np.zeros(n_bins); counts = np.zeros(n_bins, dtype=int); se = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx==b
        if np.any(m):
            mp[b] = np.mean(y_pred[m]); mt[b] = np.mean(y_true[m]); counts[b] = m.sum()
            se[b] = np.std(y_true[m]) / np.sqrt(max(1, counts[b]))
        else:
            mp[b] = np.nan; mt[b] = np.nan; se[b] = np.nan
    valid = ~np.isnan(mp)
    weights = counts[valid]/counts[valid].sum() if counts[valid].sum()>0 else np.ones(valid.sum())/valid.sum()
    ece = np.sum(weights*np.abs(mp[valid]-mt[valid])); mce = np.nanmax(np.abs(mp[valid]-mt[valid])) if valid.any() else np.nan
    spr, spp = spearmanr(mp[valid], mt[valid]) if np.sum(valid)>=3 else (np.nan, np.nan)

    cmap = plt.get_cmap("rainbow")
    fig = plt.figure(figsize=golden_figsize(5.4)); ax = fig.add_subplot(111)
    ax.plot([0,1],[0,1], ls='--', c='gray', lw=1.0, label='Ideal')

    colors = cmap(norm01(mt))
    sizes = 30 + 120*norm01(counts.astype(float))  # bubble size
    for x, y, s, ccol, err in zip(mp, mt, sizes, colors, se):
        if not np.isnan(x) and not np.isnan(y):
            ax.errorbar(x, y, yerr=err, fmt='o', ms=0, ecolor=ccol, elinewidth=1, alpha=0.9)
            ax.scatter([x],[y], s=s, c=[ccol], edgecolors='k', linewidths=0.4, zorder=3)

    ax.set_xlabel('Predicted affinity (bin mean)')
    ax.set_ylabel('Observed affinity (bin mean)')
    ax.set_title('Reliability diagram')

    smap = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=np.nanmin(mt), vmax=np.nanmax(mt)))
    smap.set_array([])
    cbar = plt.colorbar(smap, ax=ax, orientation='vertical'); cbar.set_label('Observed affinity (bin mean)')

    for s, lab in zip([30, 90, 150], ['Low n', 'Med n', 'High n']):
        ax.scatter([], [], s=s, c='gray', alpha=0.6, edgecolors='k', linewidths=0.4, label=lab)

    ax.legend(title='Bin size', loc='lower right', frameon=True)

    # Stats box
    txt = (f"N bins={np.sum(valid)}\nECE={ece:.3f}; MCE={mce:.3f}\n"
           f"Spearman(mp,mt)={spr:.3f} (p={spp:.3g})")
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, va='center', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

    base = os.path.join(out_dir, 'Figure4c_reliability_diagram' if stdnames else 'reliability_diagram')
    return save_multi(fig, base, formats)

def figS1_history(history, out_dir, formats, stdnames):
    cmap = plt.get_cmap("turbo")
    tr = np.array(history.get('loss', []), dtype=float)
    va = np.array(history.get('val_loss', []), dtype=float)
    fig = plt.figure(figsize=golden_figsize(5.2)); ax = fig.add_subplot(111)
    if len(tr)>1:
        tnorm = norm01(np.arange(len(tr)))
        for i in range(len(tr)-1):
            ax.plot([i+1,i+2],[tr[i],tr[i+1]], color=cmap(tnorm[i]), lw=1.8)
        sm_t = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=1, vmax=len(tr)))
        sm_t.set_array([])
        cbar_t = plt.colorbar(sm_t, ax=ax, orientation='vertical', pad=0.02)
        cbar_t.set_label('Epoch (training)')
    if len(va)>1:
        vnorm = norm01(np.arange(len(va)))
        for i in range(len(va)-1):
            ax.plot([i+1,i+2],[va[i],va[i+1]], color=cmap(vnorm[i]), lw=1.8, ls='--')
    ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    best = None
    if len(va)>0:
        best_idx = int(np.nanargmin(va))+1
        best = f"best val_loss={np.nanmin(va):.4f} @ epoch {best_idx}"
        ax.set_title(f'Model loss over epochs (turbo gradient; dashed = validation) — {best}')
    else:
        ax.set_title('Model loss over epochs (turbo gradient; dashed = validation)')
    base = os.path.join(out_dir, 'FigureS1_training_history' if stdnames else 'training_history')
    return save_multi(fig, base, formats)

def figS2_error_by_bin(y_true, y_pred, out_dir, formats, stdnames):
    cmap = plt.get_cmap("coolwarm")
    abs_err = np.abs(y_true - y_pred)
    bins = np.arange(0,1.00001,0.1); labels = [f"[{bins[b]:.1f},{bins[b+1]:.1f})" for b in range(len(bins)-1)]
    data = [abs_err[(y_true>=bins[b]) & (y_true<bins[b+1])] for b in range(10)]
    med = np.array([np.median(d) if len(d)>0 else np.nan for d in data], dtype=float)

    fig = plt.figure(figsize=two_col_figsize(6.6)); ax = fig.add_subplot(111)
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    mn, mx = np.nanmin(med), np.nanmax(med)
    for j, patch in enumerate(bp['boxes']):
        if np.isnan(med[j]) or mx==mn:
            patch.set_facecolor(cmap(0.5))
        else:
            patch.set_facecolor(cmap((med[j]-mn)/(mx-mn)))
        patch.set_edgecolor('black'); patch.set_alpha(0.9)
    for whisk in bp['whiskers']: whisk.set_color('black')
    for cap in bp['caps']: cap.set_color('black')
    for medline in bp['medians']: medline.set_color('black'); medline.set_linewidth(1.2)

    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('True affinity bin'); ax.set_ylabel('Absolute error')
    ax.set_title('Absolute prediction error by true-affinity bins')

    smap = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=mn, vmax=mx)); smap.set_array([])
    cbar = plt.colorbar(smap, ax=ax, orientation='vertical'); cbar.set_label('Median |error|')
    base = os.path.join(out_dir, 'FigureS2_error_by_affinity_bin' if stdnames else 'error_by_affinity_bin')
    return save_multi(fig, base, formats)

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--formats", type=str, default="png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stdnames", action="store_true", default=True)
    ap.add_argument("--jitter", type=float, default=0.001)
    ap.add_argument("--outline", action="store_true", default=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    set_base_style(args.dpi)

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
    out_dir = args.out if args.out else os.path.join(out_root, f"evalv4_{timestamp}")
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

    formats = [f.strip() for f in args.formats.split(",") if f.strip() in ("png","pdf","svg")]

    saved = []
    saved += fig3a_pred_vs_obs(y_true, y_pred, out_dir, formats, args.stdnames, jitter_scale=args.jitter, outline=args.outline)
    saved += fig3b_residuals_vs_fitted(y_true, y_pred, out_dir, formats, args.stdnames, jitter_scale=args.jitter, outline=args.outline)
    saved += fig3c_residual_hist_dual(y_true, y_pred, out_dir, formats, args.stdnames)
    saved += fig3d_qq_dual(y_true, y_pred, out_dir, formats, args.stdnames)
    saved += fig3e_bland_altman(y_true, y_pred, out_dir, formats, args.stdnames, jitter_scale=args.jitter, outline=args.outline)
    saved += fig4c_reliability(y_true, y_pred, out_dir, formats, args.stdnames, n_bins=15)
    saved += figS1_history(history, out_dir, formats, args.stdnames)
    saved += figS2_error_by_bin(y_true, y_pred, out_dir, formats, args.stdnames)

    # Metrics file
    R2 = r2_score(y_true,y_pred); pr,_ = pearsonr(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred); mae = mean_absolute_error(y_true,y_pred)
    pd.DataFrame({"Metric":["R2","Pearson r","MSE","MAE"],"Value":[R2,pr,mse,mae]}).to_csv(
        os.path.join(out_dir,"performance_metrics.csv"), index=False, float_format="%.6f"
    )

    # Manifest
    meta = {
        "timestamp": timestamp,
        "config_path": os.path.abspath(args.config),
        "model_path": os.path.abspath(model_path),
        "history_path": os.path.abspath(history_path),
        "data_path": os.path.abspath(data_path),
        "stdnames": bool(args.stdnames)
    }
    manifest = {"meta": meta, "metrics": {"R2": float(R2), "pearson_r": float(pr)}, "files": {"figures": saved}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("[✓] Complete. Outputs:", out_dir)

if __name__ == "__main__":
    main()
