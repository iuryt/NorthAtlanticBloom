import xarray as xr
import numpy as np

ds = xr.open_dataset("../../data/raw/output_submesoscale.nc")
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
    
days = 12
new_resolution_km = 10 

dsi = ds.sel(time=days, method="nearest").interp(xF=ds.xC, yF=ds.yC)


dsi_periodic = xr.concat([
    dsi.assign_coords(xC=dsi.xC-(dsi.xC.max()-dsi.xC.min())-1e3),
    dsi,
    dsi.assign_coords(xC=dsi.xC+(dsi.xC.max()-dsi.xC.min())+1e3),
],"xC")


dsm = dsi_periodic.coarsen({dim:new_resolution_km for dim in ["xC", "yC"]}, boundary="trim").mean()

dsm.sel(xC=slice(dsi.xC.min(), dsi.xC.max())).to_netcdf("../../data/interim/input_coarse.nc")