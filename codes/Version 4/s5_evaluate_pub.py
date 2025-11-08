#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (content abbreviated in analysis; full script included below exactly)
# -- See full body in this file --

import os, sys, json, time, datetime as dt, argparse, hashlib
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False
import tensorflow as tf
import matplotlib as mpl, matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

try:
    from s3b_build_model import create_weighted_mse, PositionalEncoding
except Exception:
    create_weighted_mse = None
    class PositionalEncoding(tf.keras.layers.Layer):
        def call(self, x): return x

def sha1_of_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def set_plot_style():
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 600, "font.size": 10, "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 0.8,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.linewidth": 0.5, "grid.alpha": 0.3
    })
    if HAS_SNS:
        sns.set_style("whitegrid"); sns.set_palette("colorblind")

def save_multi(fig, out_base, formats):
    saved = []
    for ext in formats:
        fig.savefig(f"{out_base}.{ext}", bbox_inches='tight')
        saved.append(f"{out_base}.{ext}")
    return saved

def golden_figsize(width_in=3.35):
    phi = (5 ** 0.5 - 1) / 2
    return (width_in, width_in * phi * 1.1)

def two_col_figsize(width_in=7.2):
    phi = (5 ** 0.5 - 1) / 2
    return (width_in, width_in * phi * 1.1)

def load_config(config_path):
    with open(config_path,'r') as f: return json.load(f)

def load_test_npz(data_path):
    X_test = {}
    all_files = os.listdir(data_path)
    test_files = sorted([f for f in all_files if f.startswith('X_test_') and f.endswith('.npz')])
    if not test_files:
        raise FileNotFoundError(f"No X_test_*.npz in {data_path}")
    for f in test_files:
        key = f.replace('X_test_','').replace('.npz','')
        with np.load(os.path.join(data_path,f), mmap_mode='r') as z:
            X_test[key] = z['data']
    with np.load(os.path.join(data_path,'y_test.npz'), mmap_mode='r') as z:
        y_test = z['data']
    return X_test, y_test

def reliability_bins(y_true,y_pred,n_bins=15):
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(y_pred,bins)-1
    centers = (bins[:-1]+bins[1:])/2.0
    mean_pred = np.zeros(n_bins); mean_true = np.zeros(n_bins); counts = np.zeros(n_bins,dtype=int)
    for b in range(n_bins):
        m = idx==b
        if np.any(m):
            mean_pred[b]=np.mean(y_pred[m]); mean_true[b]=np.mean(y_true[m]); counts[b]=m.sum()
        else:
            mean_pred[b]=np.nan; mean_true[b]=np.nan
    return centers, mean_pred, mean_true, counts

def expected_calibration_error(y_true,y_pred,n_bins=15):
    centers, mp, mt, counts = reliability_bins(y_true,y_pred,n_bins)
    valid = ~np.isnan(mp)
    weights = counts[valid]/counts[valid].sum() if counts[valid].sum()>0 else np.ones(valid.sum())/valid.sum()
    ece = np.sum(weights*np.abs(mp[valid]-mt[valid]))
    mce = np.nanmax(np.abs(mp[valid]-mt[valid])) if valid.any() else np.nan
    return ece, mce, centers, mp, mt, counts

def plot_pred_vs_obs(y_true,y_pred,r2,pr,out_dir,formats,stdnames=False):
    fig = plt.figure(figsize=two_col_figsize()); ax = fig.add_subplot(111)
    hb = ax.hexbin(y_true,y_pred,gridsize=60,cmap="viridis",mincnt=1)
    ax.plot([0,1],[0,1],'--',lw=1.5,c='white',label='Identity')
    slope,intercept,_,_,_ = stats.linregress(y_true,y_pred)
    xs = np.linspace(0,1,100); ax.plot(xs,intercept+slope*xs,c='orange',lw=1.5,label='Linear fit')
    cbar = fig.colorbar(hb,ax=ax); cbar.set_label('Count')
    ax.set_xlabel('Observed affinity'); ax.set_ylabel('Predicted affinity')
    ax.set_title(f'Predicted vs Observed (R²={r2:.3f}; r={pr:.3f})'); ax.legend()
    base = os.path.join(out_dir, 'prediction_correlation_density' if not stdnames else 'Figure3a_pred_vs_obs')
    p = save_multi(fig,base,formats); plt.close(fig); return p

def plot_training_history(history,out_dir,formats,stdnames=False):
    fig = plt.figure(figsize=golden_figsize(4.5)); ax=fig.add_subplot(111)
    ax.plot(history.get('loss',[]),label='Training',lw=1.8,color='tab:blue')
    ax.plot(history.get('val_loss',[]),label='Validation',lw=1.8,color='tab:orange')
    ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.set_title('Model loss over epochs'); ax.legend()
    base = os.path.join(out_dir, 'training_history' if not stdnames else 'FigureS1_training_history')
    p = save_multi(fig,base,formats); plt.close(fig); return p

def plot_residuals(y_true,y_pred,out_dir,formats,stdnames=False):
    residuals = y_true - y_pred
    fig = plt.figure(figsize=golden_figsize(4.5)); ax=fig.add_subplot(111)
    ax.scatter(y_pred,residuals,s=6,alpha=0.5,edgecolor='none')
    ax.axhline(0,ls='--',c='r',lw=1)
    if HAS_SM and len(y_pred)>50:
        low = sm.nonparametric.lowess(residuals,y_pred,frac=0.2,return_sorted=False)
        ax.plot(y_pred,low,c='orange',lw=1.5,label='LOWESS'); ax.legend()
    ax.set_xlabel('Predicted affinity'); ax.set_ylabel('Residual (obs - pred)')
    ax.set_title('Residuals vs Predicted')
    base = os.path.join(out_dir, 'residuals_plot' if not stdnames else 'Figure3b_residuals_vs_fitted')
    p1 = save_multi(fig,base,formats); plt.close(fig)

    fig = plt.figure(figsize=golden_figsize(4.5)); ax=fig.add_subplot(111)
    ax.hist(residuals,bins=60,alpha=0.85)
    mu,sd = np.mean(residuals), np.std(residuals)
    xs = np.linspace(mu-4*sd,mu+4*sd,200)
    ax.plot(xs,(1/(sd*np.sqrt(2*np.pi)))*np.exp(-0.5*((xs-mu)/sd)**2)*len(residuals)*(xs[1]-xs[0]),
            c='orange',lw=1.5,label='Gaussian')
    ax.set_xlabel('Residual'); ax.set_ylabel('Frequency')
    ax.set_title('Distribution of residuals'); ax.legend()
    base = os.path.join(out_dir, 'residuals_distribution' if not stdnames else 'Figure3c_residual_histogram')
    p2 = save_multi(fig,base,formats); plt.close(fig)
    return p1+p2

def plot_qq(residuals,out_dir,formats,stdnames=False):
    fig = plt.figure(figsize=golden_figsize(4.0)); ax=fig.add_subplot(111)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q–Q plot of residuals')
    base = os.path.join(out_dir, 'qq_plot' if not stdnames else 'Figure3d_qq_plot')
    p = save_multi(fig,base,formats); plt.close(fig); return p

def plot_bland_altman(y_true,y_pred,out_dir,formats,stdnames=False):
    avg = (y_true+y_pred)/2.0; diff = y_true-y_pred
    mean_diff = np.mean(diff); sd_diff = np.std(diff)
    loA = mean_diff - 1.96*sd_diff; hiA = mean_diff + 1.96*sd_diff
    frac_out = np.mean((diff<loA)|(diff>hiA))*100.0
    fig = plt.figure(figsize=golden_figsize(4.8)); ax=fig.add_subplot(111)
    ax.scatter(avg,diff,s=6,alpha=0.5,edgecolor='none')
    ax.axhline(mean_diff,c='r',ls='--',label=f'Mean diff={mean_diff:.3f}')
    ax.axhline(loA,c='gray',ls='--',label=f'LoA={loA:.3f}')
    ax.axhline(hiA,c='gray',ls='--',label=f'HiA={hiA:.3f}')
    ax.fill_between([avg.min(),avg.max()],loA,hiA,color='gray',alpha=0.12)
    ax.set_xlabel('Average of observed & predicted'); ax.set_ylabel('Difference (obs - pred)')
    ax.set_title(f'Bland–Altman (outside LoA: {frac_out:.1f}%)'); ax.legend()
    base = os.path.join(out_dir, 'bland_altman_plot' if not stdnames else 'Figure3e_bland_altman')
    p = save_multi(fig,base,formats); plt.close(fig); return p

def plot_error_by_bin(y_true,y_pred,out_dir,formats,stdnames=False):
    abs_err = np.abs(y_true-y_pred); bins = np.arange(0,1.00001,0.1)
    idx = np.digitize(y_true,bins)-1
    labels = [f"[{bins[b]:.1f},{bins[b+1]:.1f})" for b in range(len(bins)-1)]
    data = [abs_err[idx==b] for b in range(10)]
    fig = plt.figure(figsize=two_col_figsize(6.6)); ax=fig.add_subplot(111)
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('True affinity bin'); ax.set_ylabel('Absolute error')
    ax.set_title('Absolute prediction error by true-affinity bins')
    base = os.path.join(out_dir, 'error_by_affinity_bin' if not stdnames else 'FigureS2_error_by_affinity_bin')
    p = save_multi(fig,base,formats); plt.close(fig); return p

def expected_calibration_error(y_true,y_pred,n_bins=15):
    bins = np.linspace(0.0,1.0,n_bins+1)
    idx = np.digitize(y_pred,bins)-1
    centers = (bins[:-1]+bins[1:])/2.0
    mean_pred = np.zeros(n_bins); mean_true = np.zeros(n_bins); counts = np.zeros(n_bins,dtype=int)
    for b in range(n_bins):
        m = idx==b
        if np.any(m):
            mean_pred[b]=np.mean(y_pred[m]); mean_true[b]=np.mean(y_true[m]); counts[b]=m.sum()
        else:
            mean_pred[b]=np.nan; mean_true[b]=np.nan
    valid = ~np.isnan(mean_pred)
    weights = counts[valid]/counts[valid].sum() if counts[valid].sum()>0 else np.ones(valid.sum())/valid.sum()
    ece = np.sum(weights*np.abs(mean_pred[valid]-mean_true[valid]))
    mce = np.nanmax(np.abs(mean_pred[valid]-mean_true[valid])) if valid.any() else np.nan
    return ece, mce, centers, mean_pred, mean_true, counts

def plot_reliability(y_true,y_pred,out_dir,formats,stdnames=False,n_bins=15):
    ece,mce,centers,mp,mt,counts = expected_calibration_error(y_true,y_pred,n_bins)
    fig = plt.figure(figsize=golden_figsize(4.8)); ax=fig.add_subplot(111)
    ax.plot([0,1],[0,1],'--',c='gray',lw=1)
    ax.plot(mp,mt,marker='o',ls='-',lw=1.5,label='bin means')
    for x,y,c in zip(mp,mt,counts):
        ax.annotate(str(int(c)),(x,y),textcoords='offset points',xytext=(3,3),fontsize=7,alpha=0.7)
    ax.set_xlabel('Predicted affinity (bin mean)'); ax.set_ylabel('Observed affinity (bin mean)')
    ax.set_title(f'Reliability diagram (ECE={ece:.3f}, MCE={mce:.3f})'); ax.legend()
    base = os.path.join(out_dir, 'reliability_diagram' if not stdnames else 'Figure4c_reliability_diagram')
    p = save_multi(fig,base,formats); plt.close(fig); return p, {"ECE":float(ece), "MCE":float(mce)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--formats", type=str, default="png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", type=str, default=None)
    ap.add_argument("--stdnames", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    set_plot_style(); mpl.rcParams["savefig.dpi"] = args.dpi

    if args.config is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        args.config = os.path.join(script_dir,"config.json")
    cfg = load_config(args.config)
    eval_params = cfg["evaluation_parameters"]; train_params = cfg["training_parameters"]
    project_root = cfg["project_root"]; experiment_id = cfg.get("experiment_id","default_run")

    experiment_dir = os.path.join(project_root,"experiments",experiment_id)
    model_dir = os.path.join(experiment_dir, cfg["output_folders"]["main_models_folder"])
    model_name = eval_params["model_to_evaluate"]; history_name = eval_params["history_to_load"]
    data_path = os.path.join(project_root, cfg["data_folders"]["main_dataset_folder"], cfg["data_folders"]["processed_for_dl_subfolder"])

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    out_root = os.path.join(experiment_dir,"evaluation")
    bits = [eval_params.get("output_folder_prefix","eval"), model_name.replace('.keras','').replace('.h5',''), timestamp]
    if args.prefix: bits.insert(0,args.prefix)
    out_dir = args.out if args.out else os.path.join(out_root,"_".join(bits))
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "timestamp": timestamp,
        "config_path": os.path.abspath(args.config),
        "config_sha1": sha1_of_file(args.config) if os.path.exists(args.config) else None,
        "experiment_dir": experiment_dir,
        "model_path": os.path.join(model_dir, model_name),
        "history_path": os.path.join(model_dir, history_name),
        "data_path": data_path,
        "seed": args.seed,
        "stdnames": bool(args.stdnames)
    }

    X_test, y_test = load_test_npz(data_path)

    custom_objects = {}
    if train_params.get("advanced_training",{}).get("use_custom_loss",False) and (create_weighted_mse is not None):
        loss_instance = create_weighted_mse(train_params["advanced_training"]["custom_loss_pos_weight"])
        custom_objects["weighted_mse"] = loss_instance
    custom_objects["PositionalEncoding"] = PositionalEncoding

    model = tf.keras.models.load_model(meta["model_path"], custom_objects=custom_objects)
    with open(meta["history_path"],'r') as f: history = json.load(f)

    batch = eval_params.get("prediction_batch_size",1024)
    t0 = time.time()
    y_pred_transformed = model.predict(X_test, batch_size=batch, verbose=1).ravel()
    pred_seconds = time.time()-t0

    y_pred = np.square(y_pred_transformed)
    y_true = np.square(y_test)

    r2 = r2_score(y_true,y_pred); pr,_ = pearsonr(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred); mae = mean_absolute_error(y_true,y_pred)

    metrics_df = pd.DataFrame({"Metric":["R2","Pearson r","MSE","MAE"],"Value":[r2,pr,mse,mae]})
    metrics_path = os.path.join(out_dir,"performance_metrics.csv"); metrics_df.to_csv(metrics_path,index=False,float_format="%.6f")

    formats = [f.strip() for f in args.formats.split(",") if f.strip() in ("png","pdf","svg")]
    saved = []
    saved += plot_pred_vs_obs(y_true,y_pred,r2,pr,out_dir,formats,stdnames=args.stdnames)
    saved += plot_training_history(history,out_dir,formats,stdnames=args.stdnames)
    saved += plot_residuals(y_true,y_pred,out_dir,formats,stdnames=args.stdnames)
    saved += plot_qq(y_true - y_pred,out_dir,formats,stdnames=args.stdnames)
    saved += plot_bland_altman(y_true,y_pred,out_dir,formats,stdnames=args.stdnames)
    saved += plot_error_by_bin(y_true,y_pred,out_dir,formats,stdnames=args.stdnames)

    rel_paths, rel_stats = plot_reliability(y_true,y_pred,out_dir,formats,stdnames=args.stdnames)

    manifest = {
        "meta": meta,
        "metrics": {"R2": float(r2), "pearson_r": float(pr), "MSE": float(mse), "MAE": float(mae), **rel_stats},
        "files": {"metrics_csv": metrics_path, "figures": saved + rel_paths}
    }
    with open(os.path.join(out_dir,"manifest.json"),"w") as f: json.dump(manifest,f,indent=2)

    print("[✓] Complete. Outputs under:", out_dir)

if __name__ == "__main__":
    main()
