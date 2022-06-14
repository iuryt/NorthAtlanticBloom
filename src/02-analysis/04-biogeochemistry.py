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

ds = xr.open_dataset("../../data/raw/output_submesoscale.nc")
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
submeso = ds.sel(yC=yslice)


ds = xr.open_dataset("../../data/raw/output_coarse_mle.nc")
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
coarse = ds.sel(yC=yslice)

ds = xr.open_dataset("../../data/raw/output_coarse_no_mle.nc")
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
coarse_no_mle = ds.sel(yC=yslice)


data = dict(
    submeso=dict(D=submeso, label="1 km"),
    coarse_no_mle=dict(D=coarse_no_mle, label="10 km\n(no MLE)"),
    coarse=dict(D=coarse, label="10 km"),
)






vmax = 1.7
kw = dict(
    text = dict(ha="right", va="bottom", fontsize=10, fontweight="bold", color="0.5"),
    h = dict(levels=[-200, -100, -50, -10, 0], colors="0.3", linestyles="-", linewidths=1, alpha=0.7),
    new_production = dict(vmin=0, vmax=vmax, levels=np.arange(0, vmax, 0.1), cmap=cm.tempo, add_colorbar=False, extend="max"),
)

fig, ax = plt.subplots(len(data),2, figsize=(12,5))

zmin = -90
for row,k in enumerate(data):
    D = data[k]["D"].sel(zC=slice(zmin,0))

    newprod = 86400*D.new_production

    newprod.mean(["xC", "zC"]).T.plot.contourf(ax=ax[row,0], **kw["new_production"])
    ct = (-D.h).mean(["xC"]).T.plot.contour(ax=ax[row,0], **kw["h"])
    ax[row,0].clabel(ct, fmt="%.0fm", fontsize=8)

    C = newprod.mean(["xC", "yC"]).T.plot.contourf(ax=ax[row,1], **kw["new_production"])
    
    H = histogram(-D.h, bins=np.arange(-400,0,1), dim=["xC", "yC"]).T
    H = (H/H.sum("h_bin")).cumsum("h_bin")

    H.plot.contour(ax=ax[row,1], levels=[0.5], colors=["#b14e99"])
    H.plot.contourf(ax=ax[row,1], levels=[0.4,0.6], extend="neither", linestyle="--", colors=["#b14e99"], alpha=0.3, add_colorbar=False)

    ax[row,1].set(ylim=[D.zC.min(),0])
    
    for a in ax[row,:]:
        text = a.text(0.97, 0.03, data[k]["label"], 
               transform=a.transAxes, **kw["text"])
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w'),
                              path_effects.Normal()])

[a.set(xlim=[0,60]) for a in np.ravel(ax)]
for a in ax[1,:]:
    a.axvspan(0,coarse.time.min(), facecolor="w", edgecolor="0.7", hatch="//")

[a.set(ylabel="y [km]") for a in ax[:,0]]
[a.set(ylabel="z [m]") for a in ax[:,1]]

[a.set(xticklabels=[], xlabel="") for a in ax[0,:]]
[a.set(xlabel="time [days]") for a in ax[1,:]]

[a.grid(True, linestyle="--", alpha=0.8) for a in np.ravel(ax)]

letters = "a b c d".split()
_ = [a.set(title=f"{letter})"+60*" ") for a,letter in zip(np.ravel(ax),letters)]

fig.colorbar(C, ax=ax, shrink=0.7, label="New Production [mmol N / m$^3$ / day]")

fig.savefig("../../reports/figures/new_production.png", facecolor="w", dpi=300, bbox_inches="tight")   



zmin=-60
fig, ax = plt.subplots()
for k in data:
    (
        data[k]["D"].new_production
        .mean(["xC","yC"]).sel(zC=slice(zmin,0)).integrate("zC")
    ).plot(ax=ax, label=data[k]["label"])
ax.legend()






step = 5
bins = np.arange(-(100+step/2),100+step/2,step)

nflux = (submeso.w*submeso.N*86400).sel(zC=slice(-100,0))
nflux.name = "N_flux"

H = histogram(nflux, bins=bins, dim=["xC", "yC"]).T

fig,ax = plt.subplots()
H.sum("zC").plot(robust=True)
nflux.median(["xC", "yC", "zC"]).plot(x="time")
ax.set(ylim=[-50,50])

submeso_mld = submeso.interp(zC=-submeso.h).compute()
coarse_mld = coarse.interp(zC=-coarse.h).compute()


tis = [15,25,50]
vm = 0.04
kw = dict(
    nflux=dict(vmin=-vm, vmax=vm, add_colorbar=False, cmap="seismic")
)
fig, ax = plt.subplots(2,len(tis))
for col,ti in enumerate(tis):
    C = (submeso_mld.w*submeso_mld.N).sel(time=ti).plot(ax=ax[0,col], **kw["nflux"])
    (coarse_mld.w*coarse_mld.N).sel(time=ti).plot(ax=ax[1,col], **kw["nflux"])
fig.colorbar(C, ax=ax)
    

fig, ax = plt.subplots(2,len(tis))
for col,ti in enumerate(tis):
    C = (submeso.w*submeso.N).sel(zC=-100, time=ti, method="nearest").plot(ax=ax[0,col], **kw["nflux"])
    (coarse.w*coarse.N).sel(zC=-100, time=ti, method="nearest").plot(ax=ax[1,col], **kw["nflux"])
fig.colorbar(C, ax=ax)
    

submeso.new_production.mean(["xC", "yC"]).integrate("zC").plot()
coarse.new_production.mean(["xC", "yC"]).integrate("zC").plot()



fig, ax = plt.subplots(1,2)
submeso.h.mean("xC").plot(ax=ax[0])
coarse.h.mean("xC").plot(ax=ax[1])

submeso_coarsen = submeso.h.mean("xC").coarsen(yC=10).mean().interp(yC=coarse.yC)

fig,ax = plt.subplots()
(submeso_coarsen-coarse.h.mean("xC")).plot(vmin=-100,vmax=100,cmap="RdBu")
vm = 10
(submeso_coarsen-coarse.h.mean("xC")).plot.contour(levels=[-vm,vm], colors="0.2")
ax.axvline(-50)
ax.axvline(170)

data = dict(
    submeso=dict(D=submeso, label="1 km"),
    coarse=dict(D=coarse, label="10 km"),
)














fig,ax = plt.subplots(3,1)

for p,a in zip(["P", "N", "new_production"], np.ravel(ax)):

    dims = ["xC", "yC", "zC"]
    average = lambda A: A.mean(dims)
    
    for D in [submeso, coarse]:
        var = average((86400*D[p]))
        # print(var.sel(time=slice(12,80)).integrate("time"))
        var.plot(ax=a)
    a.set(xlim=[0,80])



# z_mld = (submeso.zC/submeso.h).chunk({"time":1})
# z_mld.name = "z_mld"

# nflux = (submeso.w*submeso.N).chunk({"time":1})*86400
# nflux.name = "nflux"

# bins = [
#     np.linspace(-2,0,20),
#     np.linspace(-1e3,1e3,700),
# ]
# H = histogram(z_mld, nflux, bins=bins).load()
# H = H/H.sum("nflux_bin")

# # H.sel(nflux_bin=slice(-60,60)).plot(vmax=0.01)
# ((H*H.nflux_bin).sum("nflux_bin")/H.sum("nflux_bin")).plot(y="z_mld_bin", color="r")
# # H.cumsum("nflux_bin").plot.contour(levels=[0.01,0.2,0.5,0.8,0.99], colors="0.5")


# z_mld = (coarse.zC/coarse.h).chunk({"time":1})
# z_mld.name = "z_mld"

# nflux = (coarse.w*coarse.N).chunk({"time":1})*86400
# nflux.name = "nflux"

# bins = [
#     np.linspace(-2,0,20),
#     np.linspace(-1e3,1e3,700),
# ]
# H = histogram(z_mld, nflux, bins=bins).load()
# H = H/H.sum("nflux_bin")

# # H.sel(nflux_bin=slice(-60,60)).plot(vmax=0.01)
# ((H*H.nflux_bin).sum("nflux_bin")/H.sum("nflux_bin")).plot(y="z_mld_bin", color="r")
# # H.cumsum("nflux_bin").plot.contour(levels=[0.01,0.2,0.5,0.8,0.99], colors="0.5")




# bins = [
#     np.linspace(0,1e-7,50),
#     np.linspace(0,0.1,50),
# ]

# zmin = -100
# H = histogram(submeso.sel(zC=slice(zmin,0))["∇b"].chunk({"time":1}), 
#               86400*submeso.sel(zC=slice(zmin,0)).new_production.chunk({"time":1}), bins=bins).compute()
# H.plot(robust=True)

# H = histogram(coarse.sel(zC=slice(zmin,0))["∇b"].chunk({"time":1}), 
#               86400*coarse.sel(zC=slice(zmin,0)).new_production.chunk({"time":1}), bins=bins).compute()
# H.plot(robust=True)

# submeso.new_production.mean(["xC", "yC"]).integrate("zC").plot()
# coarse.new_production.mean(["xC", "yC"]).integrate("zC").plot()

# tslice = slice(12,80)

# da = 86400*submeso.new_production.mean(["xC", "yC"]).integrate("zC").sel(time=tslice)
# da = (da*np.gradient(da.time)).cumsum("time")

# db = 86400*coarse.new_production.mean(["xC", "yC"]).integrate("zC").sel(time=tslice)
# db = (db*np.gradient(db.time)).cumsum("time")

# ((da-db)/db).plot()


# submeso.new_production.mean(["xC", "yC"]).T.sel(zC=slice(-60,0), time=slice(12,80)).plot.contourf(levels=10,vmax=3e-5)
# coarse.new_production.mean(["xC", "yC"]).T.sel(zC=slice(-60,0), time=slice(12,80)).plot.contourf(levels=10,vmax=3e-5)


# submeso.new_production.mean(["xC"]).integrate("zC").T.sel(time=slice(12,80)).plot.contourf(levels=10,vmin=0, vmax=1e-3)
# coarse.new_production.mean(["xC"]).integrate("zC").T.sel(time=slice(12,80)).plot.contourf(levels=10,vmin=0, vmax=1e-3)


# submeso.new_production.mean(["xC", "yC"]).integrate("zC").T.sel(time=slice(12,80)).plot()
# coarse.new_production.mean(["xC", "yC"]).integrate("zC").T.sel(time=slice(12,80)).plot()