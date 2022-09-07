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

sinking_text = "" if sinking else "_nosinking"

mus = [0.5,0.75,1.0,1.25]


for k in tqdm(data):
    data[k]["D"] = dict()
    for mu in mus:
        ds = xr.open_dataset(f"../../data/raw/output_{k}{sinking_text}_mu{mu}.nc").isel(time=slice(None,None,3))
        ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
        ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
        data[k]["D"][mu] = ds.sel(yC=yslice)



vmax = 1.7
kw = dict(
    text = dict(ha="right", va="bottom", fontsize=10, fontweight="bold", color="0.5"),
    h = dict(levels=[-200, -100, -50, -10, 0], colors="0.3", linestyles="-", linewidths=1, alpha=0.7),
    new_production = dict(vmin=0, vmax=vmax, levels=np.arange(0, vmax, 0.1), cmap=cm.tempo, add_colorbar=False, extend="max"),
)

for mu in tqdm(mus):
    fig, ax = plt.subplots(len(data),2, figsize=(11.5,5))

    zmin = -90
    for row,k in enumerate(data):
        D = data[k]["D"][mu].sel(zC=slice(zmin,0))

        newprod = 86400*D.new_production

        newprod.mean(["xC", "zC"]).T.plot.contourf(ax=ax[row,0], **kw["new_production"])
        ct = (-D.h).mean(["xC"]).T.plot.contour(ax=ax[row,0], **kw["h"])
        ax[row,0].clabel(ct, fmt="%.0fm", fontsize=8)

        C = newprod.mean(["xC", "yC"]).T.plot.contourf(ax=ax[row,1], **kw["new_production"])

        H = histogram(-D.h, bins=np.arange(-400,0,1), dim=["xC", "yC"]).T
        H = (H/H.sum("h_bin")).cumsum("h_bin")

        H.plot.contour(ax=ax[row,1], levels=[0.5], colors=["#b14e99"])
        # H.plot.contourf(ax=ax[row,1], levels=[0.4,0.6], extend="neither", linestyle="--", colors=["#b14e99"], alpha=0.3, add_colorbar=False)

        ax[row,1].set(ylim=[D.zC.min(),0])

        for a in ax[row,:]:
            text = a.text(0.97, 0.03, data[k]["label"], 
                   transform=a.transAxes, **kw["text"])
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'),
                                  path_effects.Normal()])

    [a.set(xlim=[1,40]) for a in np.ravel(ax)]
    for a in ax[1,:]:
        a.axvspan(1,40, facecolor="w", edgecolor="0.7", hatch="//",zorder=0)

    [a.set(ylabel="y [km]") for a in ax[:,0]]
    [a.set(ylabel="z [m]") for a in ax[:,1]]

    [a.set(xticklabels=[], xlabel="") for a in np.ravel(ax[:-1,:])]
    [a.set(xlabel="time [days]") for a in ax[-1,:]]

    [a.grid(True, linestyle="--", alpha=0.8) for a in np.ravel(ax)]

    letters = "a b c d".split()
    _ = [a.set(title=f"{letter})"+60*" ") for a,letter in zip(np.ravel(ax),letters)]

    fig.colorbar(C, ax=ax, shrink=0.7, label="New Production [mmol N / m$^3$ / day]")

    fig.savefig(f"../../reports/figures/new_production{sinking_text}_mu{mu}.png", facecolor="w", dpi=300, bbox_inches="tight")   

    
    
    
    
    
    
integ = []
for mu in mus:
    integi = []   
    for k in tqdm(data):
        integi.append((86400*data[k]["D"][mu].new_production).mean(["xC","yC"]).integrate("zC"))
    integi = xr.concat(integi,dim="exp").assign_coords(exp=list(data.keys()))
    integi = integi.interpolate_na("time")
    integ.append(integi.expand_dims("mu").assign_coords(mu=[mu]))
integ = xr.concat(integ,"mu")

print(integ.sel(time=slice(12,32)).integrate("time")-integ.sel(time=slice(12,32),exp="submesoscale").integrate("time"))
print(1-integ.sel(time=slice(12,32)).integrate("time")/integ.sel(time=slice(12,32)).integrate("time").sel(exp="submesoscale"))

colors = ["0.1","0.4","0.7","0.8"]
titles = [
    "a)"+15*" "+"SUB"+15*" ",
    "b)"+15*" "+"MLE"+15*" ",
    "c)"+15*" "+"NVF"+15*" ",
]
fig,ax = plt.subplots(1,3, figsize=(10,4))
for title,a,k in zip(titles,ax,integ.exp.values):
    for color,mu in zip(colors,mus[::-1]):
        integ.sel(exp=k, mu=mu).plot.line(ax=a,x="time",color=color)
    a.set(xlim=[0,40],ylim=[0,70],title=title,xlabel="time [days]",ylabel="New Production [mmol N / m$^2$ / day]")
    a.axvline(12,alpha=0.3,linestyle="--",zorder=10)
    a.axvline(32,alpha=0.3,linestyle="--",zorder=10)
leg = a.legend([f"$\mu$ = {mu:.2f}" for mu in mus[::-1]],framealpha=1)
leg.set_zorder(11)
_ = [a.grid(True,linestyle="--",alpha=0.5) for a in ax]
_ = [a.set(ylabel="") for a in ax[1:]]
fig.savefig(f"../../reports/figures/integrated_new_production{sinking_text}.png", facecolor="w", dpi=300, bbox_inches="tight")  

