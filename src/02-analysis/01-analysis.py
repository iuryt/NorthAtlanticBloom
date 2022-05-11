import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("../../data/raw/output_submesoscale.nc")
submeso = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
# submeso = ds.interp(xF=ds.xC, yF=ds.yC, zF=ds.zC).drop(["xF", "yF", "zF"])

ds = xr.open_dataset("../../data/raw/output_coarse_no_mle.nc")
coarse_no_mle = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
# coarse_no_mle = ds.interp(xF=ds.xC, yF=ds.yC, zF=ds.zC).drop(["xF", "yF", "zF"])

ds = xr.open_dataset("../../data/raw/output_coarse_mle.nc")
coarse_mle = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
# coarse_mle = ds.interp(xF=ds.xC, yF=ds.yC, zF=ds.zC).drop(["xF", "yF", "zF"])



fig,ax = plt.subplots(1,4)

(-submeso.h).sel(yC=yslice).mean(["xC", "yC"]).plot(ax=ax[0])
(-coarse_no_mle.h).sel(yC=yslice).mean(["xC", "yC"]).plot(ax=ax[0])
(-coarse_mle.h).sel(yC=yslice).mean(["xC", "yC"]).plot(ax=ax[0])

kw = dict(add_colorbar=False, levels=15)
(-submeso.h).sel(yC=yslice).mean("xC").T.plot(ax=ax[1], **kw)
(-coarse_no_mle.h).sel(yC=yslice).mean("xC").T.plot(ax=ax[2], **kw)
(-coarse_mle.h).sel(yC=yslice).mean("xC").T.plot(ax=ax[3], **kw)



yslice = slice(-150e3, 150e3)

fig,ax = plt.subplots()
submeso.new_production.sel(yC=yslice).integrate(["zC", "xC", "yC"]).plot()
coarse_no_mle.new_production.sel(yC=yslice).integrate(["zC", "xC", "yC"]).plot()
coarse_mle.new_production.sel(yC=yslice).integrate(["zC", "xC", "yC"]).plot()

fig,ax = plt.subplots()
submeso.P.sel(yC=yslice).integrate(["zC", "xC", "yC"]).plot()
# coarse_no_mle.P.sel(yC=yslice).integrate(["zC", "xC", "yC"]).plot()
coarse_mle.P.sel(yC=yslice).integrate(["zC", "xC", "yC"]).plot()

submeso.new_production.sel(yC=yslice, time=slice(12,80)).integrate(["zC", "xC", "yC", "time"])
coarse_no_mle.new_production.sel(yC=yslice, time=slice(12,80)).integrate(["zC", "xC", "yC", "time"])
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