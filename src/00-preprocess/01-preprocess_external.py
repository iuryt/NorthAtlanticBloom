from argopy import DataFetcher as ArgoDataFetcher
import numpy as np
import matplotlib.pyplot as plt
import seawater as sw
import xarray as xr
from xhistogram.xarray import histogram
import pandas as pd
from scipy.optimize import curve_fit
from os.path import exists
from glob import glob
from tqdm import tqdm


BOX = [-45, -15, 55, 65]
month = 2

# ----
# ---- Chlorophyll

fnames = glob("../../data/external/bioargo/*")

ds = []
pmin = 0
for fname in fnames:
    dsi = xr.open_dataset(fname)
    dsi = dsi.assign_coords(LONGITUDE=("TIME",dsi.LONGITUDE.values), LATITUDE=("TIME",dsi.LATITUDE.values))
    dsi = dsi.drop("POSITION_QC")

    if (dsi.TIME.dt.month==month).sum().values>0:
        dsi = dsi.where(dsi.TIME.dt.month==month, drop=True)[["CPHL_ADJUSTED", "PRES"]]
        dsi = dsi.where(~np.isnan(dsi.CPHL_ADJUSTED.mean("DEPTH")), drop=True)
        dsi = dsi.stack(POINT=["TIME", "DEPTH"])
        dsi = dsi.assign_coords(POINT=np.arange(pmin,pmin+dsi.POINT.size))
        ds.append(dsi)
        pmin += dsi.POINT.size
chla = xr.concat(ds,"POINT")

chla.to_netcdf("../../data/interim/bioargo_north_atlantic.nc")

# ----
# ---- Nitrate

base = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa"
no3 = xr.open_dataset(f"{base}/nitrate/all/1.00/woa18_all_n{month:02d}_01.nc", decode_times=False)
no3_p = no3.n_an.sel(lon=slice(BOX[0],BOX[1]), lat=slice(BOX[2],BOX[3])).squeeze().stack(profile=["lon", "lat"])
no3_p = no3_p.assign_coords(profile=np.arange(no3_p.profile.size)).drop("time")

no3_p.to_netcdf("../../data/interim/woa18_north_atlantic.nc")


# ----
# ---- Buoyancy

src = "argovis" #"erddap" "argovis" "gdac"
argo = ArgoDataFetcher(src=src, progress=True).region([*BOX, 0, 1000, '2010-01', '2020-12']).to_xarray()
argo = argo.argo.point2profile().argo.interp_std_levels(np.arange(0,1000))

argo = argo.where(argo.TIME.dt.month==month,drop=True)

PRES = (argo.PRES_INTERPOLATED*xr.ones_like(argo.TEMP)).T
argo = argo.assign(PDEN=(("N_PROF", "PRES_INTERPOLATED"),sw.pden(argo.PSAL.values,argo.TEMP.values,PRES.values)))
argo = argo.assign(B=-(9.82/1025)*argom.PDEN)

argo.to_netcdf("../../data/interim/argo_north_atlantic.nc")



# https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.derived.surfaceflux.html
shortwave = xr.open_dataset("../../data/external/dswrf_sfc_mon_mean.nc")
shortwave = shortwave.assign(lon=(((shortwave.lon + 180) % 360) - 180)).sortby("lon")

shortwave_month = shortwave.sel(lon=np.mean(BOX[:2]), lat=np.mean(BOX[2:]), method="nearest").dswrf.groupby("time.month").mean()
shortwave_yearday = shortwave_month.assign_coords(month=np.linspace(0,365,12)).rename(month="yearday")

shortwave_yearday.to_netcdf("../../data/interim/shortwave_north_atlantic.nc")





