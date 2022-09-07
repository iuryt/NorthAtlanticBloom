import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from cmocean import cm
from tqdm import tqdm

yslice = slice(-150, 150)

data = dict(
    submesoscale = dict(label="SUB"),
    coarse_mle = dict(label="MLE"),
    coarse_averaging = dict(label="NVF")
)

sinking = True

sinking_text = "" if sinking else "_nosinking"

mus = [0.5,0.75,1.0,1.25]

var = ["b","w","N","h"]
for k in tqdm(data):
    data[k]["D"] = dict()
    for mu in mus:
        ds = xr.open_dataset(f"../../data/raw/output_{k}{sinking_text}_mu{mu}.nc").isel(time=slice(None,None,5))
        ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
        ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3).sel(yC=yslice)
        if k=="coarse_mle":
            ds = ds.interp(xF=ds.xC).drop("xF").interp(yF=ds.yC).drop("yF").interp(zF=ds.zC).drop("zF")
        else:
            ds = ds[var]
        
        data[k]["D"][mu] = ds
    

    
mu = 0.75
mlds = np.arange(0,1.6,0.1)
datamld = []
for k in ["submesoscale","coarse_mle"]:
    for i,mldi in enumerate(tqdm(mlds)):
        if i==0:
            datai = (
                data[k]["D"][mu].sel(zC=-mldi*data[k]["D"][mu]["h"],method="nearest")
                     .drop("zC").expand_dims(["exp","mld"]).assign_coords(mld=[mldi],exp=[k])
                    )
        else:
            datai = xr.concat([
                datai,
                (
                data[k]["D"][mu].sel(zC=-mldi*data[k]["D"][mu]["h"],method="nearest")
                     .drop("zC").expand_dims(["exp","mld"]).assign_coords(mld=[mldi],exp=[k])
                    )
            ],"mld")
    datamld.append(datai)
datamld = xr.concat(datamld,"exp")
datamld = datamld.assign(w_mle=datai.w_mle)
datamld = datamld.interpolate_na("time")

#### Omand

D = data["coarse_mle"]["D"][mu]
dz = 30
where = (D.zC>-D.h-dz)&(D.zC<-D.h)
N2 = np.abs(D.b.differentiate("zC"))
M2 = np.sqrt((D.b.differentiate("xC")/1e3)**2+(D.b.differentiate("yC")/1e3)**2)
omand = 86400*np.sqrt(D.Ψx**2+D.Ψy**2).max("zC")*(M2/N2).where(where).mean("zC")*D["N"].where(where).mean("zC")/D.h
omand = omand.fillna(0)
omand = omand.where(np.isfinite(omand))


mosaic = """
    AAC
    AAC
    BBC
    BBC
    DD.
    DD.
    """
fig = plt.figure(constrained_layout=True,figsize=(9,8))
ax = fig.subplot_mosaic(mosaic)
fig.subplots_adjust(wspace=0.3, hspace=0.4)

(
    (86400*datamld.sel(exp="submesoscale").w*datamld.sel(exp="submesoscale").N)
 .mean(["xC","yC"]).plot(ax=ax["A"],ylim=[1.4,0],vmin=-70,vmax=70,cmap="bwr",add_colorbar=False)
)
C = (
    (86400*(datamld.sel(exp="coarse_mle").w+datamld.sel(exp="coarse_mle").w_mle)*datamld.sel(exp="coarse_mle").N)
    .mean(["xC","yC"]).plot(ax=ax["B"],ylim=[1.4,0],vmin=-70,vmax=70,cmap="bwr",add_colorbar=False)
)
fig.colorbar(C,ax=ax["C"],label="$\mathcal{N}_{flux}$ [mmol N / m$^2$ / day]")

(
    (86400*datamld.sel(exp="submesoscale").w*datamld.sel(exp="submesoscale").N)
 .mean(["xC","yC"]).sel(mld=1).plot(ax=ax["D"],color="#D3612C")
)
(
    (86400*(datamld.sel(exp="coarse_mle").w+datamld.sel(exp="coarse_mle").w_mle)*datamld.sel(exp="coarse_mle").N)
    .mean(["xC","yC"]).sel(mld=1).plot(ax=ax["D"],color="0.3")
)
ax["D"].set(ylim=[-5,32])

(
    (86400*datamld.sel(exp="submesoscale").w*datamld.sel(exp="submesoscale").N)
    .mean(["xC","yC"]).sel(time=slice(12,25)).mean("time").plot(ax=ax["C"],y="mld",ylim=[1.5,0],color="#D3612C")
)
(
    (86400*(datamld.sel(exp="coarse_mle").w+datamld.sel(exp="coarse_mle").w_mle)*datamld.sel(exp="coarse_mle").N)
    .mean(["xC","yC"]).sel(time=slice(12,25)).mean("time").plot(ax=ax["C"],y="mld",ylim=[1.5,0],color="0.3")
)
omand.mean(["xC","yC"]).plot(ax=ax["D"],color="0.3",linestyle="--")

_ = [ax[k].set(xlabel="") for k in ["A","B","C","D"]]

ax["C"].set(xlabel="$\mathcal{N}_{flux}$ [mmol N / m$^2$ / day]")
ax["D"].set(ylabel="$\mathcal{N}_{flux}$ [mmol N / m$^2$ / day]",xlabel="time [days]",title="")
_ = [ax[k].set(xlim=[13,40]) for k in ["A","B","D"]]
# _ = [ax[k].axhline(1,linestyle="--",color="0.4") for k in ["A","B"]]    

_ = [ax[k].set(xticklabels=[]) for k in ["A","B"]]
_ = [ax[k].set(ylabel="z / mld") for k in ["A","B","C"]]
_ = [ax[k].grid(True,linestyle="--",alpha=0.5) for k in ["A","B","C","D"]]

ax["D"].legend(["SUB","MLE","Omand (2015)"])

ax["A"].set(title=f"a)"+80*" ")
ax["B"].set(title=f"b)"+80*" ")
ax["C"].set(title=f"c)"+30*" ")
ax["D"].set(title=f"c)"+80*" ")


text = ax["A"].text(0.97, 0.03, "SUB", 
       transform=ax["A"].transAxes, ha="right", va="bottom", fontsize=13, fontweight="bold", color="0.5")
text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'),
                      path_effects.Normal()])

text = ax["B"].text(0.97, 0.03, "MLE", 
       transform=ax["B"].transAxes, ha="right", va="bottom", fontsize=13, fontweight="bold", color="0.5")
text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'),
                      path_effects.Normal()])
 
fig.savefig(f"../../reports/figures/nflux{sinking_text}_mu{mu}.png", facecolor="w", dpi=300, bbox_inches="tight")


# 207.91 mmol N / m2
(
    (
        (86400*datamld.sel(exp="submesoscale").w*datamld.sel(exp="submesoscale").N)
        .mean(["xC","yC"]).sel(time=slice(12,32),mld=1).integrate("time")
    ),
    (
        (86400*(datamld.sel(exp="coarse_mle").w+datamld.sel(exp="coarse_mle").w_mle)*datamld.sel(exp="coarse_mle").N)
        .mean(["xC","yC"]).sel(time=slice(12,32),mld=1).integrate("time")
    )
)



kw = lambda vm: dict(vmin=-vm,vmax=vm,levels=np.arange(-vm,vm+20,20),cmap="bwr",add_colorbar=False)
ti = 15
fig, ax = plt.subplots(2,2,figsize=(10,7))
for a,k in zip(ax,datamld.exp.values):
    D = datamld.sel(exp=k).sel(mld=1).interpolate_na("xC").interpolate_na("yC")
    
    if k=="submesoscale":
        w = D.w
    else:
        w = D.w+D.w_mle
        
    C1 = (86400*w*D.N).mean("xC").T.plot.contourf(**kw(300),ax=a[0])
    D.b.mean("xC").T.plot.contour(levels=np.arange(9,10,0.001),ax=a[0],colors="0.3")
    
    C2 = (86400*w*D.N).sel(time=ti,method="nearest").T.plot.contourf(**kw(600),ax=a[1])
    D.b.sel(time=ti,method="nearest").T.plot.contour(levels=np.arange(9,10,0.001),ax=a[1],colors="0.3")
    
    a[0].axvline(ti,linestyle="--")
    a[0].set(xlim=[12,40],xlabel="time [days]",ylabel="y [km]")
    a[1].set(ylim=[-25,25],xlabel="x [km]", ylabel="y [km]")
    
    for ai in a:
        ai.set(title="")
        ai.grid(True, linestyle="--", alpha=0.5)
        text = ai.text(0.97, 0.03, data[k]["label"], 
               transform=ai.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold", color="0.5")
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'),
                              path_effects.Normal()])

fig.colorbar(C1,ax=ax[:,0],label="$\mathcal{N}_{flux}$ [mmol N / m$^2$ / day]",orientation="horizontal")
fig.colorbar(C2,ax=ax[:,1],label="$\mathcal{N}_{flux}$ [mmol N / m$^2$ / day]",orientation="horizontal")

[a.set(xlabel="") for a in ax[0]]

letters = "a b c d".split()
_ = [a.set(title=f"{letter})"+60*" ") for a,letter in zip(np.ravel(ax),letters)]

fig.savefig(f"../../reports/figures/nflux_map{sinking_text}_mu{mu}.png", facecolor="w", dpi=300, bbox_inches="tight")
