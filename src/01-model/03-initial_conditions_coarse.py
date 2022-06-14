import xarray as xr
import numpy as np
from tqdm import tqdm

ds = xr.open_dataset("../../data/raw/output_submesoscale.nc")
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
    

new_resolution_km = 10 

dsm = []

for ti in tqdm(np.arange(ds.time.size)):
    dsi = ds.isel(time=ti)

    dsi_periodic = xr.concat([
        dsi.assign_coords(xC=dsi.xC-(dsi.xC.max()-dsi.xC.min())-1e3),
        dsi,
        dsi.assign_coords(xC=dsi.xC+(dsi.xC.max()-dsi.xC.min())+1e3),
    ],"xC")


    dsmi = dsi_periodic.coarsen({dim:new_resolution_km for dim in ["xC", "yC", "xF", "yF"]}, boundary="trim").mean()

    dsm.append(dsmi.sel(
        xC=slice(dsi.xC.min(), dsi.xC.max()),
        xF=slice(dsi.xF.min(), dsi.xF.max()),
    ))

dsm = xr.concat(dsm, dim="time")

dsm.to_netcdf("../../data/interim/input_coarse.nc")

