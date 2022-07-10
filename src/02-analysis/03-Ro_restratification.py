import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from tqdm import tqdm
import seawater as sw
from xhistogram.xarray import histogram

from cmocean import cm

# --- Loading data
#####


yslice = slice(-130, 130)

data = dict(
    submesoscale = dict(label="SUB"),
    coarse_mle = dict(label="MLE"),
    coarse_averaging = dict(label="NVF")
)

sinking = True
mu = 0.5

sinking_text = "" if sinking else "_nosinking"
for k in tqdm(data):

    ds = xr.open_dataset(f"../../data/raw/output_{k}{sinking_text}_mu{mu}.nc").sel(time=slice(None,None,3))
    ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
    ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
    data[k]["D"] = ds.sel(yC=yslice)




# --- Ro joint histograms
#####

color = "#D81B60" 
bins = np.arange(-1,1+0.02,0.02)
zmin = -100
ti = 20
kw = dict(
    pdf=dict(robust=True, add_colorbar=False, cmap="pink_r"),
    cdf=dict(levels=[0.01, 0.2, 0.5, 0.8, 0.99], linewidths=0.9, alpha=0.5, colors="0.2"),
    clabel=dict(fmt=lambda level: f"{level*100:.0f}%", fontsize=8),
    text=dict(ha="left", va="top", fontsize=10, fontweight="bold", color="0.3"),
)


fig,ax = plt.subplots(2,len(data), figsize=(10,4))


for col,k in enumerate(data):
    if "coarse" in k:
        ax[0,col].axhspan(0,60, facecolor="w", edgecolor="0.7", hatch="//")
    D = data[k]["D"]
    a = ax[0,col]
    H = histogram(D.Ro.sel(zC=slice(zmin,0)), bins=bins, dim=["xC", "yC", "zC"])
    H = H/H.sum("Ro_bin")
    H.plot(ax=a, **kw["pdf"])
    ct = (H/H.sum("Ro_bin")).cumsum("Ro_bin").plot.contour(ax=a, **kw["cdf"])
    a.clabel(ct, **kw["clabel"])
    a.axhline(ti, color=color, linestyle="--", linewidth=1)
    a.set(ylim=[0,80], xlabel="", ylabel="time [days]")
    a.text(0.01, 0.97, data[k]["label"]+f"\n(z>{zmin:.0f} m)", 
           transform=a.transAxes, **kw["text"])

    
    a = ax[1,col]
    H = histogram(D.Ro.sel(time=ti,method="nearest"), bins=bins, dim=["xC", "yC"])
    H = H/H.sum("Ro_bin")
    H.plot(ax=a, **kw["pdf"])
    ct = (H/H.sum("Ro_bin")).cumsum("Ro_bin").plot.contour(ax=a, **kw["cdf"])
    a.clabel(ct, **kw["clabel"])
    a.set(xlabel="Ro [$\zeta/f$]", ylabel="z [m]", yticks=-np.arange(0,1000,250))
    a.text(0.01, 0.97, data[k]["label"]+f"\n({ti} days)", transform=a.transAxes, **kw["text"])

[a.set(xticklabels=[]) for a in ax[0,:]]
[a.set(ylabel="") for a in np.ravel(ax[:,1:])]
[a.grid(True, linestyle="--", alpha=0.8) for a in np.ravel(ax)]

letters = "a b c d e f".split()
_ = [a.set(title=f"{letter})"+60*" ") for a,letter in zip(np.ravel(ax),letters)]

fig.savefig(f"../../reports/figures/Ro_pdf{sinking_text}.png", facecolor="w", dpi=300, bbox_inches="tight")

# --- MLD and N2
#####

color = "#D81B60" 
ps = dict(
    h=dict(label="mld [m]", factor=-1, kw=dict(
        add_colorbar=False, vmin=-200, vmax=0, levels=np.hstack([np.arange(-210,0,25),0]), cmap=cm.ice
    )),
    N2=dict(label="N$^2$", factor=1, kw=dict(
        add_colorbar=False, vmin=0, vmax=3e-5, levels=np.arange(0,3e-5,0.2e-5), cmap=cm.haline
    ))
)

kw_text = dict(ha="right", va="bottom", fontsize=10, fontweight="bold", color="0.5")

zmin = -150
fig,ax = plt.subplots(len(data),2, figsize=(9,8))
fig.subplots_adjust(wspace=0.25)
for row,k in enumerate(data):
        p = "h"
        a = ax[row,0]
        C1 = (
            (ps[p]["factor"]*data[k]["D"][p].mean(["xC"])).T
        ).plot.contourf(ax=a, **ps[p]["kw"])
        a.set(xlim=[1,80], ylabel="y [km]", xlabel="time [days]")
        text = a.text(0.97, 0.03, data[k]["label"], 
               transform=a.transAxes, **kw_text)
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'),
                              path_effects.Normal()])
        
        a.axvspan(0,60, facecolor="w", edgecolor="0.7", hatch="//",zorder=0)
            
        p = "N2"
        a = ax[row,1]
        C2 = (
            (ps[p]["factor"]*data[k]["D"][p].mean(["xC", "yC"])).sel(zC=slice(zmin,0)).T
        ).plot.contourf(ax=a, **ps[p]["kw"])
        (ps["h"]["factor"]*data[k]["D"]["h"].median(["xC", "yC"])).plot(ax=a, color=color)
        a.set(xlim=[1,40], ylim=[zmin,-5], ylabel="z [m]", xlabel="time [days]")
        text = a.text(0.97, 0.03, data[k]["label"], 
               transform=a.transAxes, **kw_text)
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'),
                              path_effects.Normal()])
        ax[row,1].set(ylim=[data[k]["D"].sel(zC=slice(zmin,0)).zC.min(),0])

        a.axvspan(0,60, facecolor="w", edgecolor="0.7", hatch="//",zorder=0)
        
fig.colorbar(C1, ax=ax[:,0], orientation="horizontal", label="mld [m]", ticks=[-210,-160,-110,-60,-10,0])
fig.colorbar(C2, ax=ax[:,1], orientation="horizontal", label="N$^2$ [s$^{-1}$]")

[a.set(xticklabels=[], xlabel="") for a in ax[0,:]]
[a.grid(True, linestyle="--", alpha=0.8) for a in np.ravel(ax)]

letters = "a b c d e f".split()
_ = [a.set(title=f"{letter})"+70*" ") for a,letter in zip(np.ravel(ax),letters)]

fig.savefig(f"../../reports/figures/mld_n2{sinking_text}.png", facecolor="w", dpi=300, bbox_inches="tight")