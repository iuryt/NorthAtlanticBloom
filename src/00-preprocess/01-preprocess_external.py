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
ds = xr.concat(ds,"POINT")

dc = 0.01
h_chla = histogram(-ds.PRES, ds.CPHL_ADJUSTED, bins=[np.arange(-1000,0+10,10), np.arange(-0.1-dc/2,0.3,dc)])
cdf_chla = h_chla.cumsum("CPHL_ADJUSTED_bin")/h_chla.sum("CPHL_ADJUSTED_bin")
cdf_chla = cdf_chla.assign_coords(CPHL_ADJUSTED_bin=cdf_chla.CPHL_ADJUSTED_bin+dc/2)

h_chla.plot(robust=True)
cdf_chla.plot.contour(levels=[0.5], colors=["red"])

# ----
# ---- Buoyancy

src = "gdac" #"erddap" "argovis"
argo = ArgoDataFetcher(src=src, parallel=True).region([*BOX, 0, 1000, '2010-01', '2020-12']).to_xarray()
argo = argo.argo.point2profile().argo.interp_std_levels(np.arange(0,1000))

argom = argo.groupby("TIME.month").mean("N_PROF").sel(month=month) #FEB
argom = argom.assign(PDEN=("PRES_INTERPOLATED",sw.pden(argom.PSAL,argom.TEMP,argom.PRES_INTERPOLATED)))
argom = argom.assign(B=-(9.82/1025)*argom.PDEN)
argom = argom.rolling(PRES_INTERPOLATED=50,min_periods=1).mean()

fig, ax = plt.subplots()
(argom.PDEN-argom.PDEN.sel(PRES_INTERPOLATED=0, method="nearest")).plot(ax=ax, y="PRES_INTERPOLATED", ylim=[1000,0])
ax.axvline(0.03, color="red")

argom.to_netcdf("../../data/interim/argo_north_atlantic.nc")



# ----
# ---- Nitrate
base = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa"
no3 = xr.open_dataset(f"{base}/nitrate/all/1.00/woa18_all_n{month:02d}_01.nc", decode_times=False)
no3_m = no3.n_an.sel(lon=slice(BOX[0],BOX[1]), lat=slice(BOX[2],BOX[3])).squeeze().stack(profile=["lon", "lat"])
no3_m = no3_m.assign_coords(profile=np.arange(no3_m.profile.size))



# ----
# ---- Chlorophyll

lons, lats = [], []
months = []
fnames = []
dates = []
for fname in tqdm(glob("../../data/external/Dossier_cal_netcdf/*")):
    try:
        ds = xr.open_dataset(fname)
        datei = ds.DATE.values
        
        if datei<19000000:
            datei += 19000000
        
        monthi = int(str(datei)[5:7])
        loni, lati = ds.LON.values, ds.LAT.values
        
        dates.append(datei)
        months.append(monthi)
        lons.append(loni)
        lats.append(lati)

        if (monthi==month)&(loni>BOX[0])&(loni<BOX[1])&(lati>BOX[2])&(lati<BOX[3]):
            fnames.append(fname)
    except:
        pass
    
lons = np.hstack(lons)
lats = np.hstack(lats)
months = np.hstack(months)
dates = np.hstack(dates)

for mo in range(1, 13):
    fig, ax = plt.subplots()
    ax.scatter(lons[months==mo], lats[months==mo])
    ax.plot(
        np.array(BOX)[[0,1,1,0,0]],
        np.array(BOX)[[2,2,3,3,2]],
        color="red"
    )
    di = 20
    ax.set(
        xlim=[BOX[0]-di, BOX[1]+di],
        ylim=[BOX[2]-di, BOX[3]+di],
        title=mo
    )

profiles = xr.open_mfdataset(fnames, concat_dim="FILE_NUMBER", combine="nested")
profiles = profiles.assign_coords(FILE_NUMBER=np.arange(profiles.FILE_NUMBER.size))#.sel(N_DEPTH=slice(0,1000))

fig, ax = plt.subplots()
plt.scatter(np.abs(profiles.CHLA), -profiles.DEPTH,s=0.1)
ax.set(
    ylim=[-1000, 0],
    xlim=[0,1],
)


# h_no3 = histogram(-profiles.DEPTH, profiles.CHLA, bins=[np.arange(-1000,0+1), np.arange(0,0.5,0.01)])



