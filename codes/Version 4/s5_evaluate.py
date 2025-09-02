# s5_evaluate_ULTIMATE_plus_final.py
# =============================================================================
# PURPOSE
#   Final Ultra Evaluation script — fully integrated features requested by user:
#     • Configurable bootstrap iterations (default 1000)
#     • Palette registry (10 premium palettes) and automatic colormap assignment
#     • Enforce DejaVu Sans font for consistent rendering
#     • Auto-switch stripplot -> hexbin for large datasets (>10k points)
#     • CLI modes: 'quick' (light), 'full' (heavy with UMAP/SHAP/interactive),
#       and '--fast' to force faster run (lower bootstrap iters)
#     • Robust fixes for categorical bin comparisons, safe MAPE, glyph normalization
#     • Outputs: figures (PNG+SVG), interactive HTML (Plotly), CSVs, and a single report.html
#
# NOTES
#   - Optional heavy features (UMAP, SHAP, Plotly) are used only in 'full' mode
#     and only if the corresponding Python packages are installed.
#   - The script is self-contained; change BOOTSTRAP_ITERS or default_palette at top.
# =============================================================================

import os
import sys
import io
import json
import math
import glob
import warnings
import argparse
import datetime as dt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, pearsonr, spearmanr

import matplotlib
# Enforce DejaVu Sans font for consistent glyph coverage
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Seaborn optional; script works without it
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# Optional heavy libs — used only in full mode
_HAS_SM = False
_HAS_SHAP = False
_HAS_UMAP = False
_HAS_ISO = False
_HAS_PLOTLY = False
try:
    import statsmodels.api as sm
    from statsmodels.nonparametric.smoothers_lowess import lowess
    _HAS_SM = True
except Exception:
    pass
try:
    import shap
    _HAS_SHAP = True
except Exception:
    pass
try:
    import umap
    _HAS_UMAP = True
except Exception:
    pass
try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_ISO = True
except Exception:
    pass
try:
    import plotly.express as px
    import plotly.io as pio
    _HAS_PLOTLY = True
except Exception:
    pass

import tensorflow as tf
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    median_absolute_error
)

# Custom imports from your project (guarded)
try:
    from s3b_build_model import create_weighted_mse, PositionalEncoding
except Exception:
    # If importing fails, we'll still be able to run plots that don't rely on custom objects
    PositionalEncoding = None
    create_weighted_mse = None

# -----------------------------
# User-configurable defaults
# -----------------------------
BOOTSTRAP_ITERS = 1000  # default for 'full' mode; 'quick' mode will override to 200
BOOTSTRAP_SEED = 42
DEFAULT_PALETTE = 'scientific_mix'
HEXBIN_THRESHOLD = 10000  # if n_points > this, use hexbin instead of strip/swarm

# Palette registry: maps human name -> list of Matplotlib/Seaborn palette names or colormaps
PALETTE_REGISTRY = {
    'scientific_mix': ['viridis', 'plasma', 'cividis', 'magma', 'inferno', 'rocket', 'flare', 'crest', 'turbo', 'GnBu'],
    'coolwarm_set': ['coolwarm', 'bwr', 'RdYlBu', 'Spectral', 'viridis', 'plasma', 'cividis', 'magma', 'inferno', 'turbo'],
    'publication': ['cividis', 'viridis', 'magma', 'inferno', 'plasma', 'crest', 'rocket', 'flare', 'turbo', 'GnBu']
}

# -----------------------------
# Helpers
# -----------------------------

def _now_stamp():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def safe_savefig(path: str, dpi: int = 300, bbox_inches: str = "tight"):
    base, _ = os.path.splitext(path)
    for ext in (".png", ".svg"):
        plt.savefig(base + ext, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def safe_text(s: str) -> str:
    """Normalize text to avoid glyph/font issues in matplotlib (replace fancy hyphens)."""
    if s is None:
        return s
    return s.replace('‑', '-').replace('–', '-').replace('—', '-')


def pick_palette(name: str):
    name = name if name in PALETTE_REGISTRY else 'scientific_mix'
    return PALETTE_REGISTRY[name]


def pick_cmap_from_palette(palette_list, i: int):
    cmap_name = palette_list[i % len(palette_list)]
    try:
        return plt.get_cmap(cmap_name)
    except Exception:
        return plt.get_cmap('viridis')

# -----------------------------
# Data & metrics
# -----------------------------

def load_config(config_path: str = None) -> Dict:
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    print(f"--- Loading configuration from: {config_path} ---")
    with open(config_path, 'r') as f:
        return json.load(f)


def load_test_data(data_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    print(f"  - Searching for test data in: {data_path}")
    all_files = os.listdir(data_path)
    test_files = sorted([f for f in all_files if f.startswith('X_test_') and f.endswith('.npz')])
    if not test_files:
        raise FileNotFoundError(f"No test data files (.npz) found in {data_path}.")

    X_test = {}
    for f in test_files:
        key = f.replace('X_test_', '').replace('.npz', '')
        print(f"    - Loading: {f}")
        with np.load(os.path.join(data_path, f), mmap_mode='r') as loaded_file:
            X_test[key] = loaded_file['data']
    with np.load(os.path.join(data_path, 'y_test.npz'), mmap_mode='r') as loaded_file:
        y_test = loaded_file['data']
    return X_test, y_test


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    conc = ties = total = 0
    for i in range(n - 1):
        dy = y_true[i+1:] - y_true[i]
        dp = y_pred[i+1:] - y_pred[i]
        mask = dy != 0
        dy, dp = dy[mask], dp[mask]
        cmp = dy * dp
        conc += np.sum(cmp > 0)
        ties += np.sum(dp == 0)
        total += len(dy)
    return (conc + 0.5 * ties) / total if total > 0 else np.nan


def lins_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mu_x, mu_y = np.mean(y_true), np.mean(y_pred)
    s_x2, s_y2 = np.var(y_true, ddof=1), np.var(y_pred, ddof=1)
    s_xy = np.cov(y_true, y_pred, ddof=1)[0, 1]
    denom = s_x2 + s_y2 + (mu_x - mu_y) ** 2
    return (2 * s_xy) / denom if denom != 0 else np.nan


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    sd = np.std(y_true, ddof=1)
    return (sd / rmse) if rmse > 0 else np.nan


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(denom, 1e-9))) * 100.0


def symmetric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r2_a = r2_score(y_true, y_pred)
    r2_b = r2_score(y_pred, y_true)
    return (r2_a + r2_b) / 2.0


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.maximum(np.abs(y_true), np.abs(y_pred)), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    vals = np.array(vals)
    return float(np.quantile(vals, 0.025)), float(np.mean(vals)), float(np.quantile(vals, 0.975))

# -----------------------------
# Plot helpers
# -----------------------------

def set_theme():
    if _HAS_SNS:
        sns.set_theme(style="whitegrid", palette="muted", context="talk")
    else:
        plt.style.use("seaborn-v0_8")


def add_45(ax, lim=(0, 1), **kw):
    ax.plot(lim, lim, **{"color": "#333", "lw": 1.5, "ls": ":", **kw})


def polyfit_line(x, y):
    b1, b0 = np.polyfit(x, y, 1)
    return b0, b1  # intercept, slope


def taylor_diagram(ax, std_ref, std_model, corr, label_model="Model"):
    theta = np.arccos(np.clip(corr, -1, 1))
    ax.set_title(safe_text("Taylor Diagram"), pad=16, fontweight="bold")
    t = np.linspace(0, np.pi/2, 200)
    ax.plot(std_ref * np.cos(t), std_ref * np.sin(t), lw=1.5, color="#999")
    ax.scatter(std_ref, 0.0, s=60, color="#000", label="Reference σ")
    ax.scatter(std_model * np.cos(theta), std_model * np.sin(theta), s=80,
               color="#1f77b4", label=label_model)
    for r in [std_ref/2, std_ref, std_ref*1.5]:
        ax.plot(r * np.cos(t), r * np.sin(t), color="#ddd", lw=0.8)
    for rr, txt in zip([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],
                       ["0.2","0.4","0.6","0.8","0.9","0.95","1.0"]):
        ang = np.arccos(rr)
        x, y = std_ref * np.cos(ang), std_ref * np.sin(ang)
        ax.text(x, y, txt, fontsize=9, ha="center", va="bottom")
    ax.set_xlabel("Standard Deviation (σ)")
    ax.set_ylabel("")
    ax.set_xlim(0, std_ref*2)
    ax.set_ylim(0, std_ref*2)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', frameon=True)

# -----------------------------
# Main analysis
# -----------------------------

def analyze_model_performance(mode: str = 'quick', palette_name: str = DEFAULT_PALETTE, bootstrap_iters: int = BOOTSTRAP_ITERS, fast: bool = False):
    print(f"--- Starting Ultra Evaluation & Visualization (mode={mode}) ---")
    set_theme()

    # adjust bootstrap iterations for modes
    if mode == 'quick':
        bootstrap_iters = 200
    elif mode == 'full':
        bootstrap_iters = bootstrap_iters
    if fast:
        bootstrap_iters = min(bootstrap_iters, 200)
    print(f"Using bootstrap iterations: {bootstrap_iters}")

    palette_list = pick_palette(palette_name)
    cmap_i = 0

    cfg = load_config()
    eval_params = cfg['evaluation_parameters']
    train_params = cfg['training_parameters']

    project_root = cfg['project_root']
    experiment_id = cfg.get('experiment_id', 'default_run')

    exp_dir = os.path.join(project_root, 'experiments', experiment_id)
    model_dir = os.path.join(exp_dir, cfg['output_folders']['main_models_folder'])
    base_model_name = eval_params['model_to_evaluate'].replace('.keras','').replace('.h5','')
    stamp = _now_stamp()

    eval_root = ensure_dir(os.path.join(exp_dir, 'evaluation'))
    out_dir = ensure_dir(os.path.join(eval_root, f"{eval_params['output_folder_prefix']}_{base_model_name}_{stamp}"))
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    inter_dir = ensure_dir(os.path.join(out_dir, "interactive"))

    data_path = os.path.join(project_root,
                             cfg['data_folders']['main_dataset_folder'],
                             cfg['data_folders']['processed_for_dl_subfolder'])

    print("Step 1: Load test data, model, and history …")
    try:
        X_test, y_test_t = load_test_data(data_path)
        custom_objects = {}
        if PositionalEncoding is not None:
            custom_objects['PositionalEncoding'] = PositionalEncoding
        if train_params['advanced_training'].get('use_custom_loss', False) and create_weighted_mse is not None:
            loss_instance = create_weighted_mse(train_params['advanced_training']['custom_loss_pos_weight'])
            custom_objects['weighted_mse'] = loss_instance
        model_path = os.path.join(model_dir, eval_params['model_to_evaluate'])
        history_path = os.path.join(model_dir, eval_params['history_to_load'])
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        with open(history_path, 'r') as f:
            history = json.load(f)
        print("  - Files loaded OK.")
    except Exception as e:
        print(f"  - FATAL: {e}")
        return

    print(f"Step 2: Predict & inverse‑transform …")
    y_pred_t = model.predict(X_test, batch_size=eval_params.get('prediction_batch_size', 1024), verbose=1).ravel()
    # Inverse transform per your pipeline (square)
    y_pred = np.square(y_pred_t)
    y_true = np.square(y_test_t)

    # Safety: clip to [0, 1] if affinity is defined that way
    if eval_params.get('clip_to_unit_interval', True):
        y_pred = np.clip(y_pred, 0.0, 1.0)
        y_true = np.clip(y_true, 0.0, 1.0)

    print(f"Step 3: Metrics …")
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    mape = mape_safe(y_true, y_pred)
    ci_harrell = concordance_index(y_true, y_pred)
    ccc = lins_ccc(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    rpd_val = rpd(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    sym_r2 = symmetric_r2(y_true, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": [
            "R2", "Symmetric R2", "Pearson r", "Spearman ρ", "Lin's CCC",
            "RMSE", "MAE", "Median AE", "MAPE (%)", "sMAPE (%)",
            "Bias (pred-true)", "RPD", "Harrell's C"
        ],
        "Value": [r2, sym_r2, pr, sr, ccc, rmse, mae, medae, mape, smape_val, bias, rpd_val, ci_harrell]
    })

    # Bootstrap CIs for selected metrics
    print("  - Bootstrapping CIs (this can take a moment)…")
    boot_rows = []
    for name, fn in [
        ("RMSE", lambda a,b: math.sqrt(mean_squared_error(a,b))),
        ("MAE", mean_absolute_error),
        ("R2", r2_score),
        ("Pearson r", lambda a,b: pearsonr(a,b)[0])
    ]:
        lo, mid, hi = bootstrap_ci(fn, y_true, y_pred, n_boot=bootstrap_iters, seed=BOOTSTRAP_SEED)
        boot_rows.append({"Metric": name, "Boot_CI_2.5%": lo, "Boot_Mean": mid, "Boot_CI_97.5%": hi})
    boot_df = pd.DataFrame(boot_rows)

    print("--- Metrics Summary ---")
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    metrics_df.to_csv(os.path.join(out_dir, 'metrics_summary.csv'), index=False, float_format='%.6f')
    boot_df.to_csv(os.path.join(out_dir, 'metrics_bootstrap_ci.csv'), index=False, float_format='%.6f')

    # Save raw predictions/residuals for audit
    residuals = y_true - y_pred
    df_pred = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'residual': residuals,
        'abs_residual': np.abs(residuals)
    })
    df_pred.to_csv(os.path.join(out_dir, 'predictions_and_residuals.csv'), index=False)

    # -----------------------------
    # Step 4: Plots (existing + fixes + premium palettes)
    # -----------------------------
    print("Step 4: Generating next‑gen plots …")

    # convenience
    n_points = len(y_true)
    palette_list = pick_palette(palette_name)

    # 1) Predicted vs Actual — Density & Fit
    plt.figure(figsize=(8, 8))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    hb = plt.hexbin(y_true, y_pred, gridsize=55, mincnt=1, cmap=cmap)
    cb = plt.colorbar(hb)
    cb.set_label(safe_text('Point density'))
    b0, b1 = polyfit_line(y_true, y_pred)
    xs = np.linspace(0, 1, 200)
    plt.plot(xs, b0 + b1*xs, lw=2, ls='--', color=cmap(0.2), label=f'Fit: y={b1:.2f}x+{b0:.2f}')
    add_45(plt.gca())
    plt.title(safe_text('Predicted vs Actual — Density & Fit'), fontweight='bold')
    plt.xlabel(safe_text('Actual Affinity'))
    plt.ylabel(safe_text('Predicted Affinity'))
    plt.legend()
    safe_savefig(os.path.join(fig_dir, '01_correlation_density'))

    # 2) Training history (log) with best epoch
    best_epoch = int(np.argmin(history['val_loss']))
    best_val_loss = float(history['val_loss'][best_epoch])
    plt.figure(figsize=(12, 7))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    col_train = cmap(0.75)
    col_val = cmap(0.35)
    plt.plot(history['loss'], label='Train Loss', lw=2.2, color=col_train)
    plt.plot(history['val_loss'], label='Val Loss', lw=2.2, ls='--', color=col_val)
    plt.axvline(best_epoch, color='crimson', ls=':', lw=2, label=f'Best Epoch {best_epoch}')
    plt.scatter([best_epoch], [best_val_loss], s=180, marker='*', color='crimson', zorder=5)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(safe_text('Training History'))
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    safe_savefig(os.path.join(fig_dir, '02_training_history'))

    # 3) Residuals analysis (scatter + hist/KDE)
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    ax1.scatter(y_pred, residuals, alpha=0.25, s=10, c=cmap(0.6))
    ax1.axhline(0, color='r', ls='--', lw=1)
    ax1.set_xlabel(safe_text('Predicted'))
    ax1.set_ylabel(safe_text('Residual (true - pred)'))
    ax1.set_title(safe_text('Residuals vs Predicted'))
    if _HAS_SM:
        lo = lowess(residuals, y_pred, frac=0.2, return_sorted=True)
        ax1.plot(lo[:,0], lo[:,1], lw=2, color=cmap(0.05), label='LOWESS')
        ax1.legend()
    ax2.hist(residuals, bins=50, density=False, alpha=0.9, color=cmap(0.6))
    mu, sd = norm.fit(residuals)
    xs = np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 200)
    ax2.plot(xs, norm.pdf(xs, mu, sd) * len(residuals) * (ax2.get_xlim()[1]-ax2.get_xlim()[0]) / 50.0,
             ls='--', lw=2, color=cmap(0.9), label='Normal fit')
    ax2.axvline(mu, color='k', ls=':', lw=1.5, label=f'Mean {mu:.3f}')
    ax2.set_title(safe_text('Residual Distribution'))
    ax2.legend()
    fig.suptitle(safe_text('Residuals Analysis'), fontweight='bold')
    safe_savefig(os.path.join(fig_dir, '03_residuals_analysis'))

    # 4) Q–Q plot
    plt.figure(figsize=(7, 7))
    stats.probplot(residuals, dist="norm", plot=plt)
    ax = plt.gca()
    ax.get_lines()[0].set_markerfacecolor('#4C7EF3')
    ax.get_lines()[0].set_markeredgecolor('#4C7EF3')
    ax.get_lines()[1].set_color('#C62828')
    ax.get_lines()[1].set_linewidth(2.5)
    plt.title(safe_text('Q–Q Plot of Residuals'), fontweight='bold')
    safe_savefig(os.path.join(fig_dir, '04_qq_plot'))

    # 5) Bland–Altman
    avg = (y_true + y_pred) / 2
    diff = y_true - y_pred
    md, sd = np.mean(diff), np.std(diff)
    plt.figure(figsize=(10, 7))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    plt.scatter(avg, diff, alpha=0.25, s=10, c=cmap(0.6))
    for val, lab, ls in [(md, 'Mean diff', '--'),
                         (md + 1.96*sd, '+1.96 SD', ':'),
                         (md - 1.96*sd, '-1.96 SD', ':')]:
        plt.axhline(val, color='gray', ls=ls, lw=1.5)
    plt.title(safe_text('Bland–Altman: Agreement'))
    plt.xlabel(safe_text('Average of Actual & Predicted'))
    plt.ylabel(safe_text('Actual − Predicted'))
    safe_savefig(os.path.join(fig_dir, '05_bland_altman'))

    # 6) Absolute Error by true‑value bins (violin+box + strip/hexbin)
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(y_true, bins, right=False)
    plt.figure(figsize=(13, 7))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    if _HAS_SNS:
        df_tmp = pd.DataFrame({'abs_err': np.abs(residuals), 'bin': pd.Categorical(bin_ids, ordered=True)})
        # Convert categorical to codes safely
        if isinstance(df_tmp['bin'].dtype, pd.CategoricalDtype):
            df_tmp['bin'] = df_tmp['bin'].cat.codes
        df_tmp = df_tmp[df_tmp['bin'] > 0]
        pal = sns.color_palette([cmap(v) for v in np.linspace(0.15, 0.85, 10)])
        sns.violinplot(x='bin', y='abs_err', data=df_tmp, inner=None, cut=0, palette=pal)
        sns.boxplot(x='bin', y='abs_err', data=df_tmp, width=0.15, showcaps=True, boxprops={'zorder':3, 'facecolor':'none'})
        # decide between stripplot and hexbin depending on size
        if len(df_tmp) > HEXBIN_THRESHOLD:
            # hexbin per-bin: overlay small alpha points + hexbin density as background
            for i in range(1, 11):
                sel = df_tmp[df_tmp['bin'] == i]
                if len(sel) > 0:
                    x = np.ones(len(sel)) * i
                    # plot background hexbin using raw x jitter for density
                    plt.hexbin(x + np.random.uniform(-0.35, 0.35, size=len(sel)), sel['abs_err'], gridsize=40, extent=(i-0.5, i+0.5, sel['abs_err'].min(), sel['abs_err'].max()), cmap=cmap, alpha=0.5)
        else:
            df_sw = df_tmp.sample(min(len(df_tmp), 2000), random_state=42) if len(df_tmp) > 2000 else df_tmp
            sns.stripplot(x='bin', y='abs_err', data=df_sw, size=2, alpha=0.6, jitter=0.25, color='k')
        plt.xticks(ticks=np.arange(0,10), labels=[f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(10)], rotation=45)
    else:
        groups = [np.abs(residuals)[bin_ids == i] for i in range(1, len(bins))]
        plt.boxplot(groups, showfliers=False)
        plt.xticks(range(1, len(bins)), [f"[{bins[i-1]:.1f},{bins[i]:.1f})" for i in range(1, len(bins))], rotation=45)
    plt.title(safe_text('Absolute Error by True-Value Deciles'))
    plt.xlabel(safe_text('True Value Bins'))
    plt.ylabel(safe_text('Absolute Error'))
    safe_savefig(os.path.join(fig_dir, '06_error_by_bins_violin'))

    # 7) KDE distributions y_true vs y_pred
    plt.figure(figsize=(10, 6))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    if _HAS_SNS:
        sns.kdeplot(y_true, bw_adjust=1.0, label='True', fill=True, alpha=0.35, color=cmap(0.15))
        sns.kdeplot(y_pred, bw_adjust=1.0, label='Pred', fill=True, alpha=0.35, color=cmap(0.8))
    else:
        plt.hist(y_true, bins=40, alpha=0.4, label='True', density=True, color=cmap(0.15))
        plt.hist(y_pred, bins=40, alpha=0.4, label='Pred', density=True, color=cmap(0.8))
    plt.title(safe_text('Distribution: True vs Predicted'))
    plt.xlabel(safe_text('Affinity'))
    plt.ylabel(safe_text('Density'))
    plt.legend()
    safe_savefig(os.path.join(fig_dir, '07_distribution_true_vs_pred'))

    # 8) Calibration curve
    plt.figure(figsize=(8, 8))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    plt.scatter(y_pred, y_true, alpha=0.2, s=10, color=cmap(0.6))
    b0c, b1c = polyfit_line(y_pred, y_true)
    xs = np.linspace(0, 1, 200)
    plt.plot(xs, b0c + b1c*xs, lw=2.5, color=cmap(0.15), label=f'Linear: y={b1c:.2f}x+{b0c:.2f}')
    if _HAS_ISO and mode == 'full':
        iso = IsotonicRegression(out_of_bounds='clip').fit(y_pred, y_true)
        plt.plot(xs, iso.predict(xs), lw=2.5, color=cmap(0.85), label='Isotonic')
    add_45(plt.gca())
    plt.title(safe_text('Calibration Curve (Pred → True)'))
    plt.xlabel(safe_text('Predicted'))
    plt.ylabel(safe_text('Actual'))
    plt.legend()
    safe_savefig(os.path.join(fig_dir, '08_calibration_curve_plus'))

    # 9) Conformal band + coverage curve
    q_lo, q_hi = np.quantile(residuals, [0.05, 0.95])
    lo = y_pred + q_lo
    hi = y_pred + q_hi
    plt.figure(figsize=(9, 7))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    order = np.argsort(y_pred)
    xp = y_pred[order]
    yt = y_true[order]
    lol = lo[order]
    hih = hi[order]
    plt.fill_between(xp, lol, hih, alpha=0.25, color=cmap(0.4), label='Empirical 90% band')
    plt.plot(xp, yt, lw=1, alpha=0.9, color=cmap(0.85), label='Actual (sorted by pred)')
    plt.plot(xp, xp, ls=':', lw=1.2, color='#333', label='Ideal')
    plt.xlabel(safe_text('Predicted (sorted)'))
    plt.ylabel(safe_text('Actual'))
    plt.title(safe_text('Empirical Prediction Interval Coverage'))
    plt.legend()
    safe_savefig(os.path.join(fig_dir, '09_conformal_empirical_band'))

    frac_cal = float(eval_params.get('conformal_calibration_fraction', 0.2))
    n = len(y_true)
    n_cal = max(100, int(frac_cal * n))
    idx_cal = np.arange(n - n_cal, n)
    idx_val = np.arange(0, n - n_cal)
    cal_res = np.abs(y_true[idx_cal] - y_pred[idx_cal])
    alphas = np.linspace(0.01, 0.3, 30)
    cover = []
    for a in alphas:
        qhat = np.quantile(cal_res, 1 - a)
        lo_v = y_pred[idx_val] - qhat
        hi_v = y_pred[idx_val] + qhat
        cov = np.mean((y_true[idx_val] >= lo_v) & (y_true[idx_val] <= hi_v))
        cover.append(cov)
    plt.figure(figsize=(8,6))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    plt.plot(1 - alphas, cover, lw=2, color=cmap(0.65), label='Empirical coverage')
    plt.plot([0.7, 0.99], [0.7, 0.99], ls=':', color='#333', label='Ideal')
    plt.xlim(0.7, 0.99)
    plt.ylim(0.7, 0.99)
    plt.xlabel(safe_text('Nominal confidence'))
    plt.ylabel(safe_text('Empirical coverage'))
    plt.title(safe_text('Split‑Conformal Coverage Curve'))
    plt.legend()
    safe_savefig(os.path.join(fig_dir, '09b_conformal_coverage_curve'))

    # 10) Cumulative absolute error
    ae = np.sort(np.abs(residuals))
    frac = np.linspace(0, 1, len(ae))
    plt.figure(figsize=(9, 6))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    plt.plot(frac, ae, lw=2, color=cmap(0.6))
    for p in [0.5, 0.8, 0.9, 0.95]:
        val = np.quantile(ae, p)
        plt.axhline(val, ls=':', color='#999')
        plt.text(0.02, val, f"{int(p*100)}% ≤ {val:.3f}")
    plt.title(safe_text('Cumulative Absolute Error (Empirical)'))
    plt.xlabel(safe_text('Fraction of Predictions'))
    plt.ylabel(safe_text('|Residual|'))
    safe_savefig(os.path.join(fig_dir, '10_cumulative_abs_error'))

    # 11) ECDF of |residual|
    plt.figure(figsize=(9, 6))
    vals = np.sort(np.abs(residuals))
    y_ec = np.arange(1, len(vals)+1) / len(vals)
    plt.step(vals, y_ec, where='post', color=pick_cmap_from_palette(palette_list, cmap_i)(0.6)); cmap_i += 1
    plt.xlabel(safe_text('|Residual|'))
    plt.ylabel(safe_text('Empirical CDF'))
    plt.title(safe_text('ECDF of Absolute Error'))
    safe_savefig(os.path.join(fig_dir, '11_ecdf_abs_error'))

    # 12) Taylor diagram
    std_ref = np.std(y_true)
    std_mod = np.std(y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]
    fig, ax = plt.subplots(figsize=(7,7))
    taylor_diagram(ax, std_ref, std_mod, corr)
    safe_savefig(os.path.join(fig_dir, '12_taylor_diagram'))

    # 13) Rank‑ordered ribbon plot
    order = np.argsort(y_true)
    yt = y_true[order]
    yp = y_pred[order]
    plt.figure(figsize=(12, 6))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    plt.plot(yt, lw=2, label='Actual (sorted)', color=cmap(0.15))
    plt.plot(yp, lw=1.8, alpha=0.85, label='Predicted', color=cmap(0.85))
    plt.fill_between(range(len(yt)), yt, yp, alpha=0.25, color=cmap(0.5))
    plt.title(safe_text('Rank‑Ordered Actual vs Predicted'))
    plt.xlabel(safe_text('Samples (sorted by Actual)'))
    plt.ylabel(safe_text('Affinity'))
    plt.legend()
    safe_savefig(os.path.join(fig_dir, '13_rank_order_ribbon'))

    # 14) Top‑K worst errors
    K = int(min(50, max(10, 0.01*len(df_pred))))
    worst = df_pred.nlargest(K, 'abs_residual')
    plt.figure(figsize=(max(10, K*0.2), 6))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    plt.bar(range(K), worst['abs_residual'].values, color=cmap(0.7))
    plt.xticks(range(K), worst.index.astype(str), rotation=90)
    plt.ylabel(safe_text('|Residual|'))
    plt.title(safe_text(f'Top-{K} Largest Errors'))
    safe_savefig(os.path.join(fig_dir, '14_topK_worst_errors'))

    # 15) Residual landscape: hexbin residual vs prediction + LOWESS
    plt.figure(figsize=(9,7))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    hb = plt.hexbin(y_pred, residuals, gridsize=55, mincnt=1, cmap=cmap)
    plt.colorbar(hb).set_label(safe_text('Density'))
    plt.axhline(0, color='r', ls='--', lw=1)
    if _HAS_SM:
        lo = lowess(residuals, y_pred, frac=0.2, return_sorted=True)
        plt.plot(lo[:,0], lo[:,1], lw=2.5, color=cmap(0.02), label='LOWESS')
        plt.legend()
    plt.xlabel(safe_text('Predicted'))
    plt.ylabel(safe_text('Residual (true - pred)'))
    plt.title(safe_text('Residual Landscape (Hexbin + LOWESS)'))
    safe_savefig(os.path.join(fig_dir, '15_residual_landscape'))

    # 16) Rolling error curve
    win = max(25, len(residuals)//100)
    order_p = np.argsort(y_pred)
    roll = pd.Series(np.abs(residuals)[order_p]).rolling(win, center=True).mean().values
    plt.figure(figsize=(12,5))
    cmap = pick_cmap_from_palette(palette_list, cmap_i); cmap_i += 1
    plt.plot(roll, lw=2, color=cmap(0.6))
    plt.title(safe_text(f'Rolling Mean Absolute Error (window={win})'))
    plt.xlabel(safe_text('Samples (sorted by Predicted)'))
    plt.ylabel(safe_text('Rolling |Residual|'))
    safe_savefig(os.path.join(fig_dir, '16_rolling_mae_curve'))

    # 17) Optional UMAP embedding colored by abs error (if features available & full mode)
    if mode == 'full' and _HAS_UMAP:
        try:
            first_key = sorted(X_test.keys())[0]
            X0 = X_test[first_key]
            X_flat = X0.reshape((X0.shape[0], -1))
            if X_flat.shape[1] > 2 and X_flat.shape[0] > 50:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
                emb = reducer.fit_transform(X_flat)
                plt.figure(figsize=(8,7))
                sc = plt.scatter(emb[:,0], emb[:,1], c=np.abs(residuals), s=8, alpha=0.9, cmap=pick_cmap_from_palette(palette_list, cmap_i))
                plt.colorbar(sc, label='|Residual|')
                plt.title(safe_text('UMAP of Feature Space — colored by |Residual|'))
                plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
                safe_savefig(os.path.join(fig_dir, '17_umap_residuals'))
                cmap_i += 1
        except Exception as e:
            warnings.warn(f'UMAP plot skipped: {e}')

    # 18) SHAP quick-look (only in full mode and if SHAP installed)
    if mode == 'full' and _HAS_SHAP:
        try:
            ns = min(1000, len(y_true))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(y_true), size=ns, replace=False)
            X_explain = X_flat[idx] if 'X_flat' in locals() else None
            if X_explain is not None and X_explain.shape[1] <= 200:
                f = lambda X: model.predict({first_key: X.reshape((-1,)+X0.shape[1:])}, verbose=0).ravel()
                expl = shap.Explainer(f, X_explain[:200])
                sv = expl(X_explain)
                shap.plots.beeswarm(sv, show=False, max_display=20)
                safe_savefig(os.path.join(fig_dir, '18_shap_beeswarm'))
                plt.figure(figsize=(8,6))
                shap.plots.bar(sv, show=False, max_display=20)
                safe_savefig(os.path.join(fig_dir, '18b_shap_bar'))
        except Exception as e:
            warnings.warn(f'SHAP skipped: {e}')

    # 19) Interactive Plotly: Pred vs Actual & Residual vs Pred (full mode only)
    if mode == 'full' and _HAS_PLOTLY:
        try:
            df_int = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'abs_residual': np.abs(residuals)})
            fig1 = px.scatter(df_int, x='y_true', y='y_pred', title='Predicted vs Actual (interactive)', opacity=0.6, trendline='ols', height=650)
            pio.write_html(fig1, file=os.path.join(inter_dir, 'pred_vs_actual.html'), auto_open=False, include_plotlyjs='cdn')
            fig2 = px.scatter(df_int, x='y_pred', y='abs_residual', title='Residuals vs Predicted (interactive)', opacity=0.6, height=650)
            pio.write_html(fig2, file=os.path.join(inter_dir, 'residuals_vs_pred.html'), auto_open=False, include_plotlyjs='cdn')
        except Exception as e:
            warnings.warn(f'Plotly interactive export skipped: {e}')

    # -----------------------------
    # Step 5: Mini HTML report ++
    # -----------------------------
    print("Step 5: Building HTML report …")
    figs = sorted(glob.glob(os.path.join(fig_dir, "*.png")))
    cards = []
    for fp in figs:
        name = os.path.basename(fp).replace('.png','')
        cards.append(f"<div class='card'><h3>{name}</h3><img src='figures/{os.path.basename(fp)}' loading='lazy'></div>")

    boot_html_rows = ''.join([
        f"<tr><td>{r['Metric']}</td><td>{r['Boot_CI_2.5%']:.4f}</td><td>{r['Boot_Mean']:.4f}</td><td>{r['Boot_CI_97.5%']:.4f}</td></tr>"
        for _, r in boot_df.iterrows()
    ])

    cov_tbl = pd.DataFrame({'Nominal': 1 - alphas, 'Empirical': cover})
    cov_rows = ''.join([f"<tr><td>{nv:.2f}</td><td>{ev:.2f}</td></tr>" for nv, ev in zip(cov_tbl['Nominal'], cov_tbl['Empirical'])])

    html = f"""
<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Evaluation Report — {base_model_name}</title>
<style>
 body {{ font-family: DejaVu Sans, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif; margin: 0; background: #0b1020; color: #eef1f5; }}
 header {{ padding: 24px 32px; background: linear-gradient(135deg,#0b1020 0%,#1b2a4a 60%,#0b1020 100%); box-shadow: 0 2px 0 rgba(255,255,255,.05) inset; }}
 h1 {{ margin: 0; font-size: 28px; letter-spacing: .3px; }}
 .wrap {{ padding: 24px; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
 .card {{ background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: 12px; box-shadow: 0 6px 30px rgba(0,0,0,.35); }}
 .card h3 {{ font-weight: 600; font-size: 15px; margin: 6px 4px 10px; color: #cfe2ff; }}
 .card img {{ width: 100%; height: auto; border-radius: 10px; display:block; }}
 .meta {{ margin-top: 6px; opacity: .8; font-size: 14px; }}
 table {{ width: 100%; border-collapse: collapse; margin: 12px 0 22px; }}
 th, td {{ border-bottom: 1px dashed rgba(255,255,255,.15); padding: 6px 8px; text-align: left; }}
 .metrics {{ background: rgba(255,255,255,.03); border-radius: 12px; padding: 12px; }}
 a.link {{ color: #9ad1ff; }}
 .cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
</style>
</head>
<body>
<header>
  <h1>Evaluation Report — {base_model_name}</h1>
  <div class='meta'>Generated: {_now_stamp()} | Experiment: {experiment_id} | Mode: {mode}</div>
</header>
<div class='wrap'>
  <section class='metrics'>
    <h2>Summary Metrics</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {''.join([f"<tr><td>{m}</td><td>{v:.6f}</td></tr>" for m, v in zip(metrics_df['Metric'], metrics_df['Value'])])}
      </tbody>
    </table>
    <h3>Bootstrap Confidence Intervals</h3>
    <table>
      <thead><tr><th>Metric</th><th>2.5%</th><th>Mean</th><th>97.5%</th></tr></thead>
      <tbody>
        {boot_html_rows}
      </tbody>
    </table>
    <div class='cols'>
      <div>
        <h3>Conformal Coverage (Split)</h3>
        <table>
          <thead><tr><th>Nominal</th><th>Empirical</th></tr></thead>
          <tbody>
            {cov_rows}
          </tbody>
        </table>
      </div>
      <div>
        <h3>Interactive Figures</h3>
        <div><a class='link' href='interactive/pred_vs_actual.html'>Pred vs Actual (interactive)</a></div>
        <div><a class='link' href='interactive/residuals_vs_pred.html'>Residuals vs Pred (interactive)</a></div>
      </div>
    </div>
  </section>
  <section>
    <h2>Figures</h2>
    <div class='grid'>
      {''.join(cards)}
    </div>
  </section>
</div>
</body>
</html>
"""
    with open(os.path.join(out_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ All metrics, figures, and report saved under:{out_dir}")
    print("--- Ultra Evaluation (FINAL) Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ultra evaluation script (quick/full)')
    parser.add_argument('--mode', choices=['quick','full'], default='quick', help='quick (fast) or full (heavy) analysis mode')
    parser.add_argument('--palette', type=str, default=DEFAULT_PALETTE, help='Palette registry name')
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP_ITERS, help='Number of bootstrap iterations (default for full mode)')
    parser.add_argument('--fast', action='store_true', help='Force faster execution (caps bootstrap to 200)')
    args = parser.parse_args()

    analyze_model_performance(mode=args.mode, palette_name=args.palette, bootstrap_iters=args.bootstrap, fast=args.fast)
