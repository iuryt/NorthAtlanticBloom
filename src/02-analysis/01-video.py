import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seawater as sw

from cmocean import cm

import matplotlib
matplotlib.rcdefaults()

yslice = slice(-150, 150)

data = dict(
    submesoscale = dict(label="1 km"),
    coarse_mle = dict(label="10 km\n(MLE)"),
    coarse_averaging = dict(label="10 km\n(avg.)")
)

sinking = True

sinking_text = "" if sinking else "_nosinking"

mus = [0.5,0.75,1.0,1.25]

for k in data:
    data[k]["D"] = dict()
    for mu in mus:
        ds = xr.open_dataset(f"../../data/raw/output_{k}{sinking_text}_mu{mu}.nc").isel(time=slice(None,None,3))
        ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
        ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
        data[k]["D"][mu] = ds.sel(yC=yslice)




kw = dict(
    Ro=dict(
        vmin=-1.5, vmax=1.5, levels=np.linspace(-2,2,20), cmap="RdBu_r",
    ),
    new_production=dict(
        vmin=0, vmax=2.5, levels=np.linspace(0,2.5,20), cmap=cm.tempo,
    ),
    b=dict(
        vmin=9.8, vmax=9.9, levels=np.linspace(9.8,9.9,150), colors="0.2",
        linewidths=1.5, alpha=0.5
    ),
)

ps = dict(
    Ro=dict(label="Ro [$\dfrac{\zeta}{f}$]", factor=1),
    new_production=dict(label="New Production [mmol N / m$^3$ / day]", factor=86400)
)



zmin = -500

# for ti in [12,20,38,60]:
ti = 20
fig = plt.figure(figsize=(12,8), facecolor="w")
ax = np.array([
    [fig.add_subplot(231,projection='3d'), fig.add_subplot(232,projection='3d'), fig.add_subplot(233,projection='3d')],
    [fig.add_subplot(234,projection='3d'), fig.add_subplot(235,projection='3d'), fig.add_subplot(236,projection='3d')],
])
# fig.subplots_adjust(hspace=0.0, wspace=-0.1)

mu = 0.5
for row,p in enumerate(ps):
    for col,k in enumerate(data):
        A = data[k]["D"][mu]
        a = ax[row, col]

        da = A[p].sel(zC=slice(zmin,0))*ps[p]["factor"]
        b = A["b"].sel(zC=slice(zmin,0))
        da = da.where(da<kw[p]["vmax"],kw[p]["vmax"])
        h = A["h"].sel(time=ti,method='nearest')
        h = h.where(h<-zmin)

        dsi = da.sel(time=ti,method='nearest').sel(zC=slice(-50,0))
        dsi = dsi.integrate("zC")/np.gradient(dsi.zC).sum()

        bi = b.sel(time=ti,zC=0,method='nearest')
        dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
        C = a.contourf(dims[0],dims[1],dsi.values,zdir='z',offset=0,**kw[p])
        a.contour(dims[0],dims[1],bi.values,zdir='z',offset=0,**kw["b"])

        dsi = da.sel(time=ti,method='nearest').isel(yC=0)
        bi = b.sel(time=ti,method='nearest').isel(yC=0)
        dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
        C = a.contourf(dims[0],dsi.values,dims[1],zdir='y',offset=dsi.yC.values,**kw[p])
        a.contour(dims[0],bi.values,dims[1],zdir='y',offset=dsi.yC.values,**kw["b"])

        dsi = da.sel(time=ti,xC=da.xC.max(),method='nearest')
        bi = b.sel(time=ti,xC=da.xC.max(),method='nearest')
        dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
        C = a.contourf(dsi.values,dims[0],dims[1],zdir='x',offset=dsi.xC.values,**kw[p])
        a.contour(bi.values,dims[0],dims[1],zdir='x',offset=dsi.xC.values,**kw["b"])

        a.plot(da.yC*0+da.xC[-1],da.yC,-h.isel(xC=-1),color='magenta',zorder=1e4)
        a.plot(da.xC,da.xC*0+da.yC[0],-h.isel(yC=0),color='magenta',zorder=1e4)

        a.view_init(40, -30)
        a.dist = 11

        a.set(
            xlim=[da.xC.min(),da.xC.max()],
            ylim=[da.yC.min(),da.yC.max()],
             )

        xlim,ylim,zlim = list(map(np.array,[a.get_xlim(),a.get_ylim(),a.get_zlim()]))

        color = '0.3'
        a.plot(xlim*0+xlim[1],ylim,zlim*0,color,linewidth=1,zorder=1e4)
        a.plot(xlim,ylim*0+ylim[0],zlim*0,color,linewidth=1,zorder=1e4)
        a.plot(xlim*0+xlim[1],ylim*0+ylim[0],zlim,color,linewidth=1,zorder=1e4)

        a.set(
            xlabel='\n x [km]',
            ylabel='\n y [km]',
            zlabel='\n z [m]',
            zticks=[0,-150,-300,-450],
        )
        a.set_box_aspect((150,150,80))
    # cax = plt.axes([.42, .5, .2, .01])
    # cbar = fig.colorbar(C,cax=cax,orientation='horizontal')
    cbar = fig.colorbar(C,ax=ax[row,:],orientation='horizontal', shrink=0.2, pad=0.6)
fig.tight_layout()