import xarray as xr
import xgcm
import numpy as np
import matplotlib.pyplot as plt
import proplot as pplt
from tqdm import tqdm
import seawater as sw

import matplotlib
matplotlib.rcdefaults()

yslice = slice(-150, 150)

ds = []
D = xr.open_dataset("../../data/raw/output_submesoscale.nc")
D = D.assign_coords(time=D.time.astype("float")*1e-9/86400)
for ti in tqdm(range(D.time.size)):
    ds.append(D.isel(time=ti).interp(xF=D.xC, yF=D.yC, zF=D.zC).drop(["xF", "yF", "zF"]))
ds = xr.concat(ds, "time")
ds.u[:,:,:,-1] = ds.u[:,:,:,0]

ds = ds.assign(Ro=(ds.v.differentiate("xC")-ds.u.differentiate("yC"))/sw.f(60))
ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
submeso = ds.sel(yC=yslice).isel(xC=slice(None,-1))


ds = xr.open_dataset("../../data/raw/output_coarse_mle.nc")
ds = ds.interp(xF=ds.xC, yF=ds.yC, zF=ds.zC).drop(["xF", "yF", "zF"])
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
ds.u[:,:,:,-1] = ds.u[:,:,:,0]

ds = ds.assign(Ro=(ds.v.differentiate("xC")-ds.u.differentiate("yC"))/sw.f(60))
ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
coarse = ds.sel(yC=yslice)








kw = dict(
    Ro=dict(
        vmin=-1.5, vmax=1.5, levels=np.linspace(-2,2,20), cmap="curl",
    ),
    new_production=dict(
        vmin=0, vmax=2.5, levels=np.linspace(0,2.5,20), cmap="marine",
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
    [fig.add_subplot(221,projection='3d'), fig.add_subplot(222,projection='3d')],
    [fig.add_subplot(223,projection='3d'), fig.add_subplot(224,projection='3d')]
])
# fig.subplots_adjust(hspace=0.0, wspace=-0.1)

for row,p in enumerate(ps):
    for col,A in enumerate([submeso, coarse]):
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