from argopy import DataFetcher as ArgoDataFetcher
import numpy as np
import matplotlib.pyplot as plt
import seawater as sw
import xarray as xr
from scipy.optimize import curve_fit
from os.path import exists

BOX = [-25, -15, 58, 62]

if not(exists("../../data/interim/argo_north_atlantic.nc")):
    argo = ArgoDataFetcher().region([*BOX, 0, 1000, '2010-01', '2020-12']).to_xarray()
    argo = argo.argo.point2profile().argo.interp_std_levels(np.arange(0,1000))

    argom = argo.groupby("TIME.month").mean("N_PROF").sel(month=2) #FEB
    argom = argom.assign(PDEN=("PRES_INTERPOLATED",sw.pden(argom.PSAL,argom.TEMP,argom.PRES_INTERPOLATED)))
    argom = argom.assign(B=-(9.82/1025)*argom.PDEN)
    argom = argom.rolling(PRES_INTERPOLATED=50,min_periods=1).mean()

    argom.to_netcdf("../../data/interim/argo_north_atlantic.nc")
else:
    argom = xr.open_dataset("../../data/interim/argo_north_atlantic.nc")
    
    
def func(z, a, b, c,d,e):
    return a * np.tanh(b*(z-c)) + d*z**2 + e

popt_argo, pcov_argo = curve_fit(func, argom.PRES_INTERPOLATED.values, argom.PDEN.values,p0=[0.2,6e-3,500,0,1027.6])
argom = argom.assign(PDENf=("PRES_INTERPOLATED",func(argom.PRES_INTERPOLATED.values,*popt_argo)))
argom = argom.assign(Bf=-(9.82/1025)*argom.PDENf)
title_argo =  f"{popt_argo[0]:.4f} tanh( {popt_argo[1]:.06f}( z-{popt_argo[2]:.03f} ) ) {popt_argo[3]*1e5:.3f} 10$^5$ z + {popt_argo[4]:.3f}"

print(title_argo)



# https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.derived.surfaceflux.html
shortwave = xr.open_dataset("../../data/external/dswrf_sfc_mon_mean.nc")
shortwave = shortwave.assign(lon=(((shortwave.lon + 180) % 360) - 180)).sortby("lon")

shortwave_month = shortwave.sel(lon=np.mean(BOX[:2]), lat=np.mean(BOX[2:]), method="nearest").dswrf.groupby("time.month").mean()
shortwave_yearday = shortwave_month.assign_coords(month=np.linspace(0,365,12)).rename(month="yearday")

def func_shortwave(t, a, b, c, d):
    return a * np.sin(2*np.pi*t/b - c) + d

popt_shortwave, pcov_shortwave = curve_fit(func_shortwave, shortwave_yearday.yearday.values, shortwave_yearday.values, p0=[200, 365, 0, 100])
a, b, c, d = np.round(popt_shortwave,decimals=1)


fig, ax = plt.subplots(1, 2, figsize=(6.5,3))

_ = (argom.PDEN-1000).plot.line(ax=ax[0], y="PRES_INTERPOLATED", ylim=[1000,0], lw=4, color="0.1")
_ = (argom.PDENf-1000).plot.line(ax=ax[0], y="PRES_INTERPOLATED", ylim=[1000,0], linestyle="--", color="red")
ax[0].set(
    xlabel="$\sigma_\\theta$ [kg/m$^{3}$]",
    ylabel="z [m]",
    yticks=np.arange(0, 1000, 200),
    yticklabels=-np.arange(0, 1000, 200),
    title=title_argo
)
ax[0].plot([], [], color="0.1", label="Argo")
ax[0].legend(loc=1, fontsize=8)

t0 = 50
shortwave_yearday.plot(ax=ax[1], lw=4, color="0.1")
ax[1].plot(np.arange(365), func_shortwave(np.arange(365), a, b, c, d), linestyle="--", color="red")

ax[1].axvspan(t0, t0+90, color="0.7")
ax[1].set(
    ylabel="Shortwave Radiation [W/m$^{2}$]",
    xlabel="Day of the year",
    xlim=[0,365],
    xticks=np.arange(0,365,50),
    title=f"{a} sin( 2$\pi$ ( t + t$_0$ ) / {b} + {c} ) + {d}"
)
ax[1].plot([], [], color="0.1", label="NCEP")
ax[1].legend(loc=1, fontsize=8)
fig.tight_layout()

fig.savefig("../../reports/figures/data_driven_initial_conditions.png")