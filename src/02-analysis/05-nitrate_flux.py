import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def volume(A):
    dx = np.gradient(A.xC)*xr.ones_like(A.xC)
    dy = np.gradient(A.yC)*xr.ones_like(A.yC)
    dz = np.gradient(A.zC)*xr.ones_like(A.zC)
    
    return dx*dy*dz



yslice = slice(-150e3, 150e3)

ds = xr.open_dataset("../../data/raw/output_submesoscale.nc")
submeso = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400).sel(yC=yslice)


ds = xr.open_dataset("../../data/raw/output_coarse_mle.nc")
coarse_mle = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400).sel(yC=yslice)


a = (submeso.new_production.integrate(["xC", "yC", "zC"])/volume(submeso).sum())
b = (coarse_mle.new_production.integrate(["xC", "yC", "zC"])/volume(coarse_mle).sum())

kw = dict(vmin=0, vmax=3e-5, levels=10, cmap="Greens")
fig,ax = plt.subplots(3,1, figsize=(9,9))

submeso.new_production.mean(["xC", "yC"]).integrate("zC").plot(ax=ax[0])
coarse_mle.new_production.mean(["xC", "yC"]).integrate("zC").plot(ax=ax[0])

submeso.new_production.mean(["xC", "yC"]).T.plot.contourf(ax=ax[1], add_colorbar=False, **kw)
coarse_mle.new_production.mean(["xC", "yC"]).T.plot.contourf(ax=ax[2], add_colorbar=False, **kw)

_ = [a.set(ylim=[-80,0]) for a in ax[1:]]
_ = [a.set(xlim=[0,80]) for a in ax]


submeso_mld = submeso.interp(zC=-submeso.h).compute()
coarse_mle_mld = coarse_mle.interp(zC=-coarse_mle.h)

tslice = slice(13, 80)
fig,ax = plt.subplots(figsize=(7,5))
((submeso_mld.w*submeso_mld.N).integrate(["xC", "yC"])/(submeso.dy*submeso.dx)).rolling(time=5,center=True).mean().sel(time=tslice).plot()
((coarse_mle_mld.w*coarse_mle_mld.N).integrate(["xC", "yC"])/(coarse_mle.dy*coarse_mle.dx)).rolling(time=5,center=True).mean().sel(time=tslice).plot()

tslice = slice(12, 80)

fig,ax = plt.subplots(1,4, figsize=(13,5))

(-submeso.h).mean(["xC", "yC"]).plot(ax=ax[0])
(-coarse_mle.h).mean(["xC", "yC"]).plot(ax=ax[0])
ax[0].set(
    xlim=[0,80]
)

kw = dict(add_colorbar=False, levels=-np.arange(0,120,10), vmin=-120, vmax=0)
(-submeso.h).sel(time=tslice).mean("xC").T.plot(ax=ax[1], **kw)
C = (-coarse_mle.h).sel(time=tslice).mean("xC").T.plot(ax=ax[3], **kw)

fig.colorbar(C, ax=ax)


submeso_new_production = (submeso.new_production.integrate(["zC", "xC", "yC"])/submeso.volume)
coarse_mle_new_production = (coarse_mle.new_production.integrate(["zC", "xC", "yC"])/coarse_mle.volume)

fig,ax = plt.subplots()
submeso_new_production.sel(time=slice(12,80)).plot()
coarse_mle_new_production.sel(time=slice(12,80)).plot()

fig,ax = plt.subplots()
(submeso.P.integrate(["zC", "xC", "yC"])/submeso.volume).plot()
(coarse_mle.P.integrate(["zC", "xC", "yC"])/coarse_mle.volume).plot()

fig,ax = plt.subplots()
(submeso.N.integrate(["zC", "xC", "yC"])/submeso.volume).plot()
(coarse_mle.N.integrate(["zC", "xC", "yC"])/coarse_mle.volume).plot()

submeso.new_production.sel(yC=yslice, time=slice(12,80)).integrate(["zC", "xC", "yC", "time"])
coarse_mle.new_production.sel(yC=yslice, time=slice(12,80)).integrate(["zC", "xC", "yC", "time"])

submeso.P.sel(yC=yslice).mean(["yC", "xC"]).T.plot(ylim=[-200,0])

submeso.N.sel(yC=yslice).mean(["yC", "xC"]).T.plot(ylim=[-200,0])





(submeso.u**2+submeso.v**2+submeso.w**2).integrate("zC").mean("xC").plot()

submeso.v.sel(zC=-10, method="nearest").mean("xC").plot(robust=True)

submeso.new_production.sel(yC=yslice).integrate(["zC"]).mean("xC").plot()


submeso.new_production.sel(yC=yslice).integrate(["zC", "xC", "yC"]).plot()


# sponge_size = 50e3
# slope = 10e3
# ymin = y.min()
# ymax = y.max()

# mask_func = lambda y: ((
#      np.tanh((y-(ymax-sponge_size))/slope)
#     *np.tanh((y-(ymin+sponge_size))/slope)
# )+1)/2

# horizontal_closure_func = lambda y: 1 + mask_func(y) * (100 - 1)