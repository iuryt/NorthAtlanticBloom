import xarray as xr
import xgcm
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset("../data/output_1km.nc")
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
    
days = 25
new_resolution_km = 10 

dsi = ds.sel(time=days).interp(xF=ds.xC, yF=ds.yC)
dsm = dsi.coarsen({dim:new_resolution_km for dim in ["xC", "yC"]}, boundary="trim").mean()

# fig, ax = plt.subplots(2, 2, figsize=(7, 7))
# ax = np.ravel(ax)

# dsi.sel(zC=0, method="nearest").b.plot(ax=ax[0], add_colorbar=False)
# dsi.sel(zC=0, method="nearest").b.plot.contour(ax=ax[0], levels=10, colors="k", linestyles="-")

# C = dsm.sel(zC=0, method="nearest").b.plot(ax=ax[1], add_colorbar=False)
# dsm.sel(zC=0, method="nearest").b.plot.contour(ax=ax[1], levels=10, colors="k", linestyles="-")

# fig.colorbar(C, ax=ax[:2])

# dsi.sel(xC=0, method="nearest").b.plot(ax=ax[2], add_colorbar=False)
# dsi.sel(xC=0, method="nearest").b.plot.contour(ax=ax[2], levels=10, colors="k", linestyles="-")

# C = dsm.sel(xC=0, method="nearest").b.plot(ax=ax[3], add_colorbar=False)
# dsm.sel(xC=0, method="nearest").b.plot.contour(ax=ax[3], levels=10, colors="k", linestyles="-")

# fig.colorbar(C, ax=ax[2:])

# _ = [a.set(title="") for a in ax]

dsm.to_netcdf("../data/input_coarsen.nc")