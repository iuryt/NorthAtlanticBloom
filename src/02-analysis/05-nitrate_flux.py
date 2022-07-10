import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
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
    datai = []
    for mldi in tqdm(mlds):
        datai.append(data[k]["D"][mu].sel(zC=-mldi*data[k]["D"][mu]["h"],method="nearest"))
    datamld.append(xr.concat(datai,dim="mld").assign_coords(mld=mlds))
        

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

(86400*datamld[0].w*datamld[0].N).mean(["xC","yC"]).plot(ax=ax["A"],ylim=[1.4,0],vmin=-70,vmax=70,cmap=cm.curl,add_colorbar=False)
C = (86400*(datamld[1].w+datamld[1].w_mle)*datamld[1].N).mean(["xC","yC"]).plot(ax=ax["B"],ylim=[1.4,0],vmin=-70,vmax=70,cmap=cm.curl,add_colorbar=False)
fig.colorbar(C,ax=ax["C"],label="$\mathcal{N}_{flux}$ [mmol N / m$^2$ / day]")

(86400*datamld[0].w*datamld[0].N).mean(["xC","yC"]).sel(mld=1).plot(ax=ax["D"],color="#D3612C")
(86400*(datamld[1].w+datamld[1].w_mle)*datamld[1].N).mean(["xC","yC"]).sel(mld=1).plot(ax=ax["D"],color="0.3")
ax["D"].set(ylim=[-5,32])

(86400*datamld[0].w*datamld[0].N).mean(["xC","yC"]).sel(time=slice(12,25)).mean("time").plot(ax=ax["C"],y="mld",ylim=[1.5,0],color="#D3612C")
(86400*(datamld[1].w+datamld[1].w_mle)*datamld[1].N).mean(["xC","yC"]).sel(time=slice(12,25)).mean("time").plot(ax=ax["C"],y="mld",ylim=[1.5,0],color="0.3")
omand.mean(["xC","yC"]).plot(ax=ax["D"],color="0.3",linestyle="--")

_ = [ax[k].set(xlabel="") for k in ["A","B","C","D"]]

ax["C"].set(xlabel="$\mathcal{N}_{flux}$ [mmol N / m$^2$ / day]")
ax["D"].set(ylabel="$\mathcal{N}_{flux}$ [mmol N / m$^2$ / day]",xlabel="time [days]",title="")
_ = [ax[k].set(xlim=[12,40]) for k in ["A","B","D"]]
# _ = [ax[k].axhline(1,linestyle="--",color="0.4") for k in ["A","B"]]    

_ = [ax[k].set(xticklabels=[]) for k in ["A","B"]]
_ = [ax[k].set(ylabel="z / mld") for k in ["A","B","C"]]
_ = [ax[k].grid(True,linestyle="--",alpha=0.5) for k in ["A","B","C","D"]]

ax["D"].legend(["SUB","MLE","Omand (2015)"])

ax["A"].set(title=f"a)        SUB"+70*" ")
ax["B"].set(title=f"b)        MLE"+70*" ")
ax["C"].set(title=f"c)"+30*" ")
ax["D"].set(title=f"c)"+80*" ")
    
fig.savefig(f"../../reports/figures/nflux{sinking_text}_mu{mu}.png", facecolor="w", dpi=300, bbox_inches="tight")


# 207.91 mmol N / m2
(
    (86400*datamld[0].w*datamld[0].N).mean(["xC","yC"]).sel(time=slice(12,40),mld=1).integrate("time"),
    (86400*(datamld[1].w+datamld[1].w_mle)*datamld[1].N).mean(["xC","yC"]).sel(time=slice(12,40),mld=1).integrate("time")
)


vm = 600
kw = dict(vmin=-vm,vmax=vm,levels=np.arange(-vm,vm+25,25),cmap=cm.curl,add_colorbar=False)
fig, ax = plt.subplots(2,3,figsize=(8,10))
for row in range(2):
    for col,ti in enumerate([20,25,35]):
        D = datamld[row].sel(mld=1,time=ti,method="nearest")
        (86400*D.w*D.N).plot(**kw,ax=ax[row,col])
        D.b.plot.contour(levels=np.arange(9,10,0.001),ax=ax[row,col],colors="0.3")
    

