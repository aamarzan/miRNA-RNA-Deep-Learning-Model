#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s5_evaluate_pub_v8.py — tiny placement tweaks
- 3a: stats NW; legend bottom-right
- 3c: stats NE; font a tad smaller
"""

import os, json, argparse, datetime as dt
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, jarque_bera, linregress
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf

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

def save_multi(fig, base, formats):
    outs=[]
    for ext in formats:
        p=f"{base}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs

def load_config(path):
    with open(path,"r") as f: return json.load(f)

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

def norm01(v):
    v=np.asarray(v); return (v-np.nanmin(v))/(np.nanmax(v)-np.nanmin(v)+1e-12)

def concordance_cc(y_true,y_pred):
    y_true=np.asarray(y_true).ravel(); y_pred=np.asarray(y_pred).ravel()
    mu1,mu2=np.mean(y_true),np.mean(y_pred)
    v1,v2=np.var(y_true),np.var(y_pred)
    cov=np.mean((y_true-mu1)*(y_pred-mu2))
    return (2*cov)/(v1+v2+(mu1-mu2)**2+1e-12)

def fisher_ci_r(r,n,alpha=0.05):
    if n<4: return (np.nan,np.nan)
    z=np.arctanh(max(-0.9999999,min(0.9999999,r))); se=1/np.sqrt(n-3)
    zc=stats.norm.ppf(1-alpha/2); return np.tanh(z-zc*se), np.tanh(z+zc*se)

def place_stats_box(ax, text, corner="NE", pad=0.03, fontsize=8.2):
    pts={"NE":(1-pad,1-pad,"right","top"),
         "NW":(pad,1-pad,"left","top"),
         "SE":(1-pad,pad,"right","bottom"),
         "SW":(pad,pad,"left","bottom")}
    x,y,ha,va=pts[corner]
    ax.text(x,y,text,transform=ax.transAxes,ha=ha,va=va,fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.28",fc="white",ec="gray",alpha=0.92))

# ---------- Figures ----------
def fig3a(y_true,y_pred,out_dir,formats,stdnames,jitter=0.001,outline=True):
    n=len(y_true); R2=r2_score(y_true,y_pred)
    pr,pp=pearsonr(y_true,y_pred); sr,sp=spearmanr(y_true,y_pred)
    slope,intercept,rvl,p_slope,se=linregress(y_true,y_pred)
    lo_s,hi_s=slope-1.96*se,slope+1.96*se
    MSE=mean_squared_error(y_true,y_pred); RMSE=np.sqrt(MSE); MAE=mean_absolute_error(y_true,y_pred)
    CCC=concordance_cc(y_true,y_pred); pr_lo,pr_hi=fisher_ci_r(pr,n)

    fig=plt.figure(figsize=two_col_figsize()); ax=fig.add_subplot(111)
    cmap=plt.get_cmap("Spectral")
    xx=y_true+np.random.uniform(-jitter,jitter,size=n); yy=y_pred+np.random.uniform(-jitter,jitter,size=n)
    dens=stats.gaussian_kde(np.vstack([xx,yy]))(np.vstack([xx,yy]))
    ax.scatter(xx,yy,c=norm01(dens),cmap=cmap,s=12,alpha=0.9,
               edgecolors=("k" if outline else "none"), linewidths=(0.2 if outline else 0.0), rasterized=True)
    xs=np.linspace(np.min(xx),np.max(xx),200)
    ax.plot(xs,xs,ls="--",lw=1.0,c="gray",label="Identity")
    ax.plot(xs,intercept+slope*xs,c="black",lw=1.0,label="Linear fit")
    ax.set_xlabel("Observed affinity"); ax.set_ylabel("Predicted affinity"); ax.set_title("Predicted vs Observed")
    ax.legend(loc="lower right")  # bottom-right legend

    txt=(f"N={n}\nR²={R2:.3f}\n"
         f"r={pr:.3f} (p={pp:.3g}; 95% CI {pr_lo:.3f}–{pr_hi:.3f})\n"
         f"ρ={sr:.3f} (p={sp:.3g})\n"
         f"slope={slope:.3f} [{lo_s:.3f},{hi_s:.3f}]; intercept={intercept:.3f}\n"
         f"MAE={MAE:.4f}; RMSE={RMSE:.4f}; CCC={CCC:.3f}")
    place_stats_box(ax,txt,corner="NW",pad=0.03,fontsize=8.2)  # top-left stats
    cbar=plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0,1)), ax=ax)
    cbar.set_label("Relative point density (KDE)")
    base=os.path.join(out_dir,"Figure3a_pred_vs_obs" if stdnames else "prediction_correlation_density")
    return save_multi(fig,base,formats)

def fig3b(y_true,y_pred,out_dir,formats,stdnames,jitter=0.001,outline=True):
    cmap=plt.get_cmap("turbo")
    resid=y_true-y_pred; n=len(resid)
    mr,sd=float(np.mean(resid)),float(np.std(resid,ddof=1))
    rho,rho_p=spearmanr(y_pred,resid); jb_stat,jb_p=jarque_bera(resid)
    fig=plt.figure(figsize=golden_figsize(5.0)); ax=fig.add_subplot(111)
    xp=y_pred+np.random.uniform(-jitter,jitter,size=n)
    rr=resid+np.random.uniform(-jitter,jitter,size=n)
    sc=ax.scatter(xp,rr,c=norm01(np.abs(rr)),cmap=cmap,s=14,alpha=0.9,
                  edgecolors=("k" if outline else "none"), linewidths=(0.2 if outline else 0.0), rasterized=True)
    ax.axhline(0,ls="--",c="gray",lw=1)
    ax.set_xlabel("Predicted affinity"); ax.set_ylabel("Residual (obs - pred)"); ax.set_title("Residuals vs Predicted")
    txt=(f"N={n}\nmean={mr:.4f}; SD={sd:.4f}\nρ(pred,resid)={rho:.3f} (p={rho_p:.3g})\nJB p={jb_p:.3g}")
    place_stats_box(ax,txt,corner="NW",pad=0.03,fontsize=8.2)
    cbar=plt.colorbar(sc,ax=ax,orientation="vertical"); cbar.set_label("Absolute residual")
    base=os.path.join(out_dir,"Figure3b_residuals_vs_fitted" if stdnames else "residuals_plot")
    return save_multi(fig,base,formats)

def fig3c(y_true,y_pred,out_dir,formats,stdnames):
    resid=y_true-y_pred; skew=stats.skew(resid,nan_policy="omit")
    kurt=stats.kurtosis(resid,nan_policy="omit"); jb_stat,jb_p=jarque_bera(resid)
    cmap=plt.get_cmap("Spectral")
    fig=plt.figure(figsize=golden_figsize(5.0)); ax=fig.add_subplot(111)
    counts,bins,patches=ax.hist(resid,bins=60,edgecolor="none")
    cmin,cmax=counts.min(),counts.max()
    for c,p in zip(counts,patches):
        p.set_facecolor(cmap(0.5 if cmax==cmin else (c-cmin)/(cmax-cmin)))
    ax.set_xlabel("Residual"); ax.set_ylabel("Frequency (count)"); ax.set_title("Distribution of residuals")
    smap=mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(vmin=cmin,vmax=cmax)); smap.set_array([])
    cbar=plt.colorbar(smap,ax=ax,orientation="vertical"); cbar.set_label("Bin count")
    txt=(f"Skew={skew:.3f}; Kurt={kurt:.3f}\nJB p={jb_p:.3g}")
    place_stats_box(ax,txt,corner="NE",pad=0.02,fontsize=7.7)  # NE + slightly smaller
    base=os.path.join(out_dir,"Figure3c_residual_histogram__pal-Spectral" if stdnames else "residuals_distribution__pal-Spectral")
    return save_multi(fig,base,formats)

def fig3d(y_true,y_pred,out_dir,formats,stdnames):
    resid=y_true-y_pred; jb_p=jarque_bera(resid)[1]
    cmap=plt.get_cmap("magma")
    osm,osr=stats.probplot(resid,dist="norm"); theo=osm[0]; ordered=np.sort(resid)
    fig=plt.figure(figsize=golden_figsize(4.8)); ax=fig.add_subplot(111)
    sc=ax.scatter(theo,ordered,c=norm01(ordered),cmap=cmap,s=12,alpha=0.9,rasterized=True)
    slope,intercept,R=osr; ax.plot(theo,slope*theo+intercept,c="black",lw=1.2)
    ax.set_title("Q–Q plot of residuals"); ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Ordered residuals")
    txt=f"Fit (R={R:.3f})\nJB p={jb_p:.3g}"
    place_stats_box(ax,txt,corner="NW",pad=0.03,fontsize=8.2)
    cbar=plt.colorbar(sc,ax=ax,orientation="vertical"); cbar.set_label("Residual value")
    base=os.path.join(out_dir,"Figure3d_qq_plot__pal-magma" if stdnames else "qq_plot__pal-magma")
    return save_multi(fig,base,formats)

def fig3e(y_true,y_pred,out_dir,formats,stdnames,jitter=0.001,outline=True):
    cmap=plt.get_cmap("turbo")
    avg=(y_true+y_pred)/2.0; diff=y_true-y_pred
    md=np.mean(diff); sd=np.std(diff); loA=md-1.96*sd; hiA=md+1.96*sd
    frac_out=100.0*np.mean((diff<loA)|(diff>hiA)); pr,pp=pearsonr(avg,diff)
    slope,intercept,rvl,p_slope,se=linregress(avg,diff)
    fig=plt.figure(figsize=two_col_figsize()); ax=fig.add_subplot(111)
    xa=avg+np.random.uniform(-jitter,jitter,size=len(avg)); df=diff+np.random.uniform(-jitter,jitter,size=len(diff))
    sc=ax.scatter(xa,df,c=norm01(np.abs(df)),cmap=cmap,s=16,alpha=0.9,
                  edgecolors=("k" if outline else "none"), linewidths=(0.2 if outline else 0.0), rasterized=True)
    ax.axhline(md,c="black",ls="--",lw=1.0,label=f"Mean diff={md:.3f}")
    ax.axhline(loA,c="gray",ls="--",lw=1.0,label=f"LoA={loA:.3f}")
    ax.axhline(hiA,c="gray",ls="--",lw=1.0,label=f"LoA={hiA:.3f}")
    ax.set_xlabel("Average of observed & predicted"); ax.set_ylabel("Difference (obs - pred)")
    ax.set_title("Bland–Altman (no shading)")
    leg=ax.legend(loc="upper right", frameon=True)
    for t in leg.get_texts(): t.set_fontsize(8.0)
    txt=(f"Mean diff={md:.4f}; LoA [{loA:.4f},{hiA:.4f}]\nOutside LoA={frac_out:.1f}%  "
         f"slope={slope:.3f} (p={p_slope:.3g})\n"
         f"r(avg,diff)={pr:.3f} (p={pp:.3g})")
    place_stats_box(ax,txt,corner="NW",pad=0.03,fontsize=8.2)
    cbar=plt.colorbar(sc,ax=ax,orientation="vertical"); cbar.set_label("Absolute difference")
    base=os.path.join(out_dir,"Figure3e_bland_altman" if stdnames else "bland_altman_plot")
    return save_multi(fig,base,formats)

def fig4c(y_true,y_pred,out_dir,formats,stdnames,n_bins=10):
    bins=np.linspace(0.0,1.0,n_bins+1); idx=np.digitize(y_pred,bins)-1
    mp,mt,counts,se=np.zeros(n_bins),np.zeros(n_bins),np.zeros(n_bins,int),np.zeros(n_bins)
    for b in range(n_bins):
        m=idx==b
        if np.any(m):
            mp[b]=np.mean(y_pred[m]); mt[b]=np.mean(y_true[m]); counts[b]=m.sum()
            se[b]=np.std(y_true[m])/np.sqrt(max(1,counts[b]))
        else:
            mp[b]=np.nan; mt[b]=np.nan; se[b]=np.nan
    valid=~np.isnan(mp); weights=counts[valid]/counts[valid].sum() if counts[valid].sum()>0 else np.ones(valid.sum())/max(1,valid.sum())
    ece=np.sum(weights*np.abs(mp[valid]-mt[valid])); mce=np.nanmax(np.abs(mp[valid]-mt[valid])) if valid.any() else np.nan
    spr,spp=spearmanr(mp[valid],mt[valid]) if np.sum(valid)>=3 else (np.nan,np.nan)

    cmap=plt.get_cmap("rainbow")
    fig=plt.figure(figsize=golden_figsize(5.6)); ax=fig.add_subplot(111)
    ax.plot([0,1],[0,1],ls="--",c="gray",lw=1.0,label="Ideal")
    colors=cmap(norm01(mt)); sizes=32+120*norm01(counts.astype(float))
    for x,y,s,cc,err in zip(mp,mt,sizes,colors,se):
        if not (np.isnan(x) or np.isnan(y)):
            ax.errorbar(x,y,yerr=err,fmt='o',ms=0,ecolor=cc,elinewidth=1,alpha=0.9)
            ax.scatter([x],[y],s=s,c=[cc],edgecolors='k',linewidths=0.4,zorder=3)
    ax.set_xlabel("Predicted affinity (bin mean)"); ax.set_ylabel("Observed affinity (bin mean)"); ax.set_title("Reliability diagram")
    smap=mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(vmin=np.nanmin(mt),vmax=np.nanmax(mt))); smap.set_array([])
    cbar=plt.colorbar(smap,ax=ax,orientation="vertical"); cbar.set_label("Observed affinity (bin mean)")
    for s,lab in zip([32,90,150],["Low n","Med n","High n"]):
        ax.scatter([],[],s=s,c="gray",alpha=0.6,edgecolors="k",linewidths=0.4,label=lab)
    ax.legend(title="Bin size",loc="lower right",frameon=True)
    txt=(f"N bins={np.sum(valid)}  ECE={ece:.3f}; MCE={mce:.3f}\nSpearman(mp,mt)={spr:.3f} (p={spp:.3g})")
    place_stats_box(ax,txt,corner="NW",pad=0.03,fontsize=8.2)
    base=os.path.join(out_dir,"Figure4c_reliability_diagram" if stdnames else "reliability_diagram")
    return save_multi(fig,base,formats)

def figS1(history,out_dir,formats,stdnames):
    cmap=plt.get_cmap("turbo")
    tr=np.array(history.get("loss",[]),float); va=np.array(history.get("val_loss",[]),float)
    fig=plt.figure(figsize=two_col_figsize(7.2)); ax=fig.add_subplot(111)
    epochs=np.arange(1,max(len(tr),len(va))+1); norm=mpl.colors.Normalize(vmin=epochs.min() if len(epochs) else 1, vmax=epochs.max() if len(epochs) else 1)
    if len(tr)>1:
        for i in range(1,len(tr)):
            ax.plot([i,i+1],[tr[i-1],tr[i]],lw=1.8,c=cmap(norm(i)))
    if len(va)>1:
        for i in range(1,len(va)):
            ax.plot([i,i+1],[va[i-1],va[i]],lw=1.8,ls="--",c=cmap(norm(i)))
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    best_txt=""; 
    if len(va)>0:
        best_idx=int(np.nanargmin(va))+1; best_txt=f"Val best={np.nanmin(va):.4f} (ep {best_idx})"
    train_end=f"Train end={tr[-1]:.4f}" if len(tr)>0 else ""
    legend_line="Solid: train | Dashed: val"
    msg=" | ".join([t for t in [train_end, best_txt, legend_line] if t])
    if msg:
        place_stats_box(ax,msg,corner="NW",pad=0.03,fontsize=8.2)
    smap=mpl.cm.ScalarMappable(cmap=cmap,norm=norm); smap.set_array([])
    cbar=plt.colorbar(smap,ax=ax,orientation="vertical"); cbar.set_label("Epoch (training)")
    base=os.path.join(out_dir,"FigureS1_training_history" if stdnames else "training_history")
    return save_multi(fig,base,formats)

def figS2(y_true,y_pred,out_dir,formats,stdnames):
    cmap=plt.get_cmap("Spectral")
    abs_err=np.abs(y_true-y_pred)
    bins=np.arange(0,1.00001,0.1); labels=[f"[{bins[b]:.1f},{bins[b+1]:.1f})" for b in range(len(bins)-1)]
    data=[abs_err[(y_true>=bins[b])&(y_true<bins[b+1])] for b in range(10)]
    med=np.array([np.median(d) if len(d)>0 else np.nan for d in data],float)
    fig=plt.figure(figsize=two_col_figsize(6.6)); ax=fig.add_subplot(111)
    bp=ax.boxplot(data,showfliers=False,patch_artist=True)
    mn,mx=np.nanmin(med),np.nanmax(med)
    for j,patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(0.5 if mx==mn or np.isnan(med[j]) else (med[j]-mn)/(mx-mn)))
        patch.set_alpha(0.85)
        patch.set_edgecolor("black"); patch.set_linewidth(0.8)
    for w in bp["whiskers"]: w.set_color("black"); w.set_linewidth(0.8)
    for c in bp["caps"]: c.set_color("black"); c.set_linewidth(0.8)
    for m in bp["medians"]: m.set_color("black"); m.set_linewidth(1.2)
    ax.set_xticklabels(labels,rotation=45,ha="right"); ax.set_xlabel("True affinity bin"); ax.set_ylabel("Absolute error"); ax.set_title("Absolute prediction error by true-affinity bins")
    smap=mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(vmin=mn,vmax=mx)); smap.set_array([])
    cbar=plt.colorbar(smap,ax=ax,orientation="vertical"); cbar.set_label("Median |error|")
    base=os.path.join(out_dir,"FigureS2_error_by_affinity_bin" if stdnames else "error_by_affinity_bin")
    return save_multi(fig,base,formats)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=str,default=None)
    ap.add_argument("--out",type=str,default=None)
    ap.add_argument("--formats",type=str,default="png,pdf,svg")
    ap.add_argument("--dpi",type=int,default=600)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--stdnames",action="store_true",default=True)
    ap.add_argument("--jitter",type=float,default=0.001)
    ap.add_argument("--outline",action="store_true",default=True)
    args=ap.parse_args()

    np.random.seed(args.seed); set_base_style(args.dpi)
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
    out_dir=args.out if args.out else os.path.join(out_root,f"evalv8_{timestamp}")
    os.makedirs(out_dir,exist_ok=True)

    X_test,y_test=load_npz_test(data_path)

    custom_objects={}
    try:
        from s3b_build_model import create_weighted_mse, PositionalEncoding
        if cfg["training_parameters"].get("advanced_training",{}).get("use_custom_loss",False) and create_weighted_mse is not None:
            loss_instance=create_weighted_mse(cfg["training_parameters"]["advanced_training"]["custom_loss_pos_weight"])
            custom_objects["weighted_mse"]=loss_instance
        custom_objects["PositionalEncoding"]=PositionalEncoding
    except Exception:
        pass

    model=tf.keras.models.load_model(model_path,custom_objects=custom_objects)
    with open(history_path,"r") as f: history=json.load(f)

    batch=eval_params.get("prediction_batch_size",1024)
    y_pred_t=model.predict(X_test,batch_size=batch,verbose=1).ravel()
    y_pred=np.square(y_pred_t); y_true=np.square(y_test)

    formats=[f.strip() for f in args.formats.split(",") if f.strip() in ("png","pdf","svg")]

    saved=[]
    saved+=fig3a(y_true,y_pred,out_dir,formats,args.stdnames,args.jitter,args.outline)
    saved+=fig3b(y_true,y_pred,out_dir,formats,args.stdnames,args.jitter,args.outline)
    saved+=fig3c(y_true,y_pred,out_dir,formats,args.stdnames)
    saved+=fig3d(y_true,y_pred,out_dir,formats,args.stdnames)
    saved+=fig3e(y_true,y_pred,out_dir,formats,args.stdnames,args.jitter,args.outline)
    saved+=fig4c(y_true,y_pred,out_dir,formats,args.stdnames)
    saved+=figS1(history,out_dir,formats,args.stdnames)
    saved+=figS2(y_true,y_pred,out_dir,formats,args.stdnames)

    R2=r2_score(y_true,y_pred); pr,_=pearsonr(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred); mae=mean_absolute_error(y_true,y_pred)
    pd.DataFrame({"Metric":["R2","Pearson r","MSE","MAE"],"Value":[R2,pr,mse,mae]}).to_csv(
        os.path.join(out_dir,"performance_metrics.csv"), index=False, float_format="%.6f"
    )
    with open(os.path.join(out_dir,"manifest.json"),"w") as f:
        json.dump({"files":{"figures":saved}},f,indent=2)
    print("[✓] Complete:", out_dir)

if __name__=="__main__":
    main()
