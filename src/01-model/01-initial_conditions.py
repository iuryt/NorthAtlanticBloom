import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from xhistogram.xarray import histogram
from scipy.optimize import curve_fit


chla = xr.open_dataset("../../data/interim/bioargo_north_atlantic.nc")
chla = chla.assign(biomass=chla.CPHL_ADJUSTED*(16/(0.06*106*14)))
chla.biomass["units"] = "mmol N m-3"

no3 = xr.open_dataset("../../data/interim/woa18_north_atlantic.nc")
shortwave = xr.open_dataarray("../../data/interim/shortwave_north_atlantic.nc")
argo = xr.open_dataset("../../data/interim/argo_north_atlantic.nc").mean("N_PROF")

argo = argo.assign(PRES_INTERPOLATED=-argo.PRES_INTERPOLATED.astype("float")).rename(PRES_INTERPOLATED="z")


titles = {}
    
def func_argo(z, a, b, c, d):
    return a * np.tanh(b*(z-c)) + d

popt_argo, pcov_argo = curve_fit(func_argo, argo.z.values, argo.PDEN.values,p0=[-0.4, 1e-2, -500, 1027.57])
popt_argo[0] = np.round(popt_argo[0],3)
popt_argo[1] = np.round(popt_argo[1],4)
popt_argo[2] = np.round(popt_argo[2],0)
popt_argo[3] = np.round(popt_argo[3],1)

argo = argo.assign(PDENf=("z",func_argo(argo.z.values,*popt_argo)))
argo = argo.assign(Bf=-(9.82/1025)*argo.PDENf)
titles["pden"] =  f"{popt_argo[0]}tanh({popt_argo[1]*1e3}(z+{-popt_argo[2]})/1000)+{popt_argo[3]-1000:.1f}"


def func_shortwave(t, a, b, c, d):
    return a * np.sin(2*np.pi*t/b - c) + d

popt_shortwave, pcov_shortwave = curve_fit(func_shortwave, shortwave.yearday.values, shortwave.values, p0=[200, 365, 0, 100])
a, b, c, d = np.round(popt_shortwave,decimals=1)

titles["shortwave"] =  f"{a:.0f} sin(2$\pi$*t/{b}-{c})+{d}"



h_no3 = histogram(no3.n_an, bins=np.arange(7,22,0.5), dim=["profile"]).rolling(depth=5, center=True, min_periods=1).mean()
h_no3 = h_no3.assign_coords(depth=-h_no3.depth).rename(depth="z")
cdf_no3 = h_no3.cumsum("n_an_bin")/h_no3.sum("n_an_bin")
popt_no3 = np.array([12,16,0,-800,12])
titles["no3"] = f"z ({popt_no3[0]}-{popt_no3[1]})/({popt_no3[2]}+{-popt_no3[3]}) + {popt_no3[4]}"

dc = 0.001
h_chla = histogram(-chla.PRES, chla.biomass, bins=[np.arange(-1000,0+10,10), np.arange(-0.001-dc/2,0.05,dc)])
cdf_chla = h_chla.cumsum("biomass_bin")/h_chla.sum("biomass_bin")
cdf_chla = cdf_chla.assign_coords(CPHL_ADJUSTED_bin=cdf_chla.biomass_bin+dc/2)
popt_chla = np.array([0.02,1.2e-2,-260])
titles["chla"] = f"{popt_chla[0]}(tanh({popt_chla[1]}(z+{-popt_chla[2]}))+1)/2 "


def make_figure():
    colors = {
        "obs": "#1E88E5",
        "model": "#D81B60",
    }

    fig = plt.figure(figsize=(6,6), constrained_layout=True)
    ax = fig.subplot_mosaic(
        [
            ["pden", "chla"],
            ["no3", "shortwave"],
        ],
    )

    _ = (argo.PDEN-1000)[::100].plot.line(ax=ax["pden"], y="z", ylim=[1000,0], lw=0, marker="o", color=colors["obs"])
    _ = (argo.PDENf-1000).plot.line(ax=ax["pden"], y="z", ylim=[1000,0], color=colors["model"])
    ax["pden"].text(0.5, 0.02, "Argo", ha="center", color="0.3", fontsize=12, transform=ax["pden"].transAxes)

    cdf_chla.plot.contourf(ax=ax["chla"], levels=[0.2,0.8], colors=colors["obs"], alpha=0.5, extend="neither", add_colorbar=False)
    cdf_chla.plot.contour(ax=ax["chla"], levels=[0.5], colors=colors["obs"])
    ax["chla"].text(0.5, 0.02, "BioArgo", ha="center", color="0.3", fontsize=12, transform=ax["chla"].transAxes)

    cdf_no3.plot.contourf(ax=ax["no3"], levels=[0.2,0.8], colors=colors["obs"], alpha=0.5, extend="neither", add_colorbar=False)
    cdf_no3.plot.contour(ax=ax["no3"], levels=[0.5], colors=colors["obs"])
    ax["no3"].text(0.3, 0.02, "WOA18", ha="center", color="0.3", fontsize=12, transform=ax["no3"].transAxes)

    zi = -np.arange(1000)
    ax["chla"].plot(popt_chla[0]*(np.tanh(popt_chla[1]*(zi-popt_chla[2]))+1)/2, zi, color=colors["model"])
    ax["no3"].plot(zi*(popt_no3[0]-popt_no3[1])/(popt_no3[2]-popt_no3[3])+popt_no3[4], zi, color=colors["model"])

    t0 = 50
    shortwave.plot(ax=ax["shortwave"], lw=0, marker="o", color=colors["obs"])
    ax["shortwave"].plot(np.arange(365), func_shortwave(np.arange(365), a, b, c, d), color=colors["model"])
    ax["shortwave"].axvline(t0, color="0.2", linestyle="--")
    ax["shortwave"].text(0.5, 0.02, "NCEP", ha="center", color="0.3", fontsize=12, transform=ax["shortwave"].transAxes)

    ax["pden"].set(
        xlabel="$\sigma_\\theta$ [kg m$^{-3}$]",
        ylabel="z [m]",
        yticks=-np.arange(0,1000,200),
        ylim=[-1000,-10],
        xlim=[27.4,27.75],
    )

    ax["chla"].set(
        ylabel="z [m]",
        yticks=-np.arange(0,1000,200),
        ylim=[-1000,-10],
        xlabel="Chl-a [mmol N m$^{-3}$]",
    )

    ax["no3"].set(
        ylabel="z [m]",
        yticks=-np.arange(0,1000,200),
        ylim=[-800,-10],
        xlabel="Nitrate [mmol N m$^{-3}$]"
    )


    ax["shortwave"].set(
        ylabel="Shortwave [W m$^{-2}$]",
        xlabel="Day of the year",
        title="",
        ylim=[20,280],
        xlim=[0,370],
        xticks=np.arange(0,365,50),
    )
    # title=f"{a} sin( 2$\pi$ ( t + t$_0$ ) / {b} + {c} ) + {d}"

    _ = [ax[k].grid(True, linestyle="--", alpha=0.5) for k in ax]
    letters = "a b c d".split()
    _ = [ax[k].set(title=f"{letter})"+50*" ") for k,letter in zip(ax, letters)]
    
    # kw = dict(fontsize=9, color=colors["model"], rotation=90, va="center")
    # _ = [ax[k].text(0.93,0.5,titles[k], **kw, transform=ax[k].transAxes) for k in titles]

    return fig,ax

fig,ax = make_figure()
fig.savefig("../../reports/figures/data_driven_initial_conditions.png", facecolor="w", dpi=300, bbox_inches="tight")