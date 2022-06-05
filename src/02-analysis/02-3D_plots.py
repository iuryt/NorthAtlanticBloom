import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from cmocean import cm
from cmaps import BuGnWYlRd



yslice = slice(-130, 130)

ds = xr.open_dataset("../../data/raw/output_submesoscale.nc")
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
submeso = ds.sel(yC=yslice)



tis = [0.1,13,25,60]


vm = 2.0

mld = submeso.sel(time=tis,method='nearest')['h'].compute()
da = submeso.sel(time=tis,method='nearest').sel(zC=slice(-600,0))["Ro"].compute()
b = submeso.sel(time=tis,method='nearest').sel(zC=slice(-600,0))["b"].compute()

da = da.where(da<vm,vm).where(da>-vm,-vm)


cmap = BuGnWYlRd.reversed()
kw = {
    "vmin":-vm,
    "vmax":vm,
    "levels":np.arange(-vm,vm+0.1,0.1),
    "cmap":cmap,
    "alpha":0.9,
}

kwb = {
    "levels":np.arange(b.min(),b.max()+0.1,0.5/1000),
    "colors":"0.2",
    "linewidths":0.5,
    "alpha":0.9,
    "zorder":1e10,
}



fig = plt.figure(figsize=np.array([14,8])*0.8)

ax = [fig.add_subplot(220+i,projection='3d',computed_zorder=False) for i in np.arange(1,4+1)]

letters = ['a)','b)','c)','d)']
for i,a in enumerate(ax):
    letter = letters[i]
    ti = tis[i]
    
    # a.set_title(f'{letter}'+5*' '+f'{ti} days'+' '*5,rotation=10)
    a.text(-40, -100, 200, f'{letter}'+5*' '+f'{ti} days'+' '*5, ha="left", va="bottom", zdir="y")
    
    
    dsi = da.sel(time=ti,zC=0,method='nearest')
    bi = b.sel(time=ti,zC=0,method='nearest')
    dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
    C = a.contourf(dims[0],dims[1],dsi.values,zdir='z',offset=0,**kw)
    a.contour(dims[0],dims[1],bi.values,zdir='z',offset=0,**kwb)

    dsi = da.sel(time=ti,method='nearest').isel(yC=0)
    bi = b.sel(time=ti,method='nearest').isel(yC=0)
    dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
    C = a.contourf(dims[0],dsi.values,dims[1],zdir='y',offset=dsi.yC.values,**kw)
    a.contour(dims[0],bi.values,dims[1],zdir='y',offset=dsi.yC.values,**kwb)

    dsi = da.sel(time=ti,xC=da.xC.max(),method='nearest')
    bi = b.sel(time=ti,xC=da.xC.max(),method='nearest')
    dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
    C = a.contourf(dsi.values,dims[0],dims[1],zdir='x',offset=dsi.xC.values,**kw)
    a.contour(bi.values,dims[0],dims[1],zdir='x',offset=dsi.xC.values,**kwb)

    di = mld.sel(time=ti,method='nearest')
    a.plot(di.yC*0+di.xC[-1],di.yC,-di.isel(xC=-1),color='magenta',zorder=1e4)
    a.plot(di.xC,di.xC*0+di.yC[0],-di.isel(yC=0),color='magenta',zorder=1e4)
    
    a.set(
        xlim=[da.xC.min(),da.xC.max()],
        ylim=[da.yC.min(),da.yC.max()],
        zlim=[da.zC.min(),da.zC.max()],
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
        box_aspect=[1,1,0.4]
    )
    
    a.view_init(30, -30)
    a.dist = 10

_ = [a.set(zlabel='') for a in [ax[0],ax[2]]]

cax = plt.axes([.42, .5, .2, .01])
cbar = fig.colorbar(C,cax=cax,orientation='horizontal')
cbar.set_ticks(np.arange(-2,2+1,1))
cbar.set_label(r'$\dfrac{\zeta}{f}$')

fig.subplots_adjust(wspace=0)


pax = [
    plt.axes([.05, .87, .1, .1]),
    plt.axes([.83, .87, .1, .1]),
    plt.axes([.05, .40, .1, .1]),
    plt.axes([.83, .40, .1, .1])
]

for i,a in enumerate(pax):
    da.isel(zC=np.abs(da.zC+mld).argmin('zC')).isel(time=i).plot.hist(
        ax=a,bins=np.arange(-(vm-0.1),(vm-0.1)+0.1,0.1),density=True,color='0.4'
    )
    a.set(yticks=[],title='',xlabel='')
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_visible(False)
#     a.spines['bottom'].set_visible(False)

pax[0].text(0.2,0.8,r"pdf$\,\|_{mld}$",fontsize=12,ha='center',transform=pax[0].transAxes)
pax[0].set_xlabel(r'$\dfrac{\zeta}{f}$')

fig.tight_layout()
fig.savefig("../../reports/figures/3D_Ro.png", facecolor="w", dpi=300, bbox_inches="tight")  






#######################################################################################################


vm = 150.0
zmax = -30
mld = submeso.sel(time=tis,method='nearest')['h'].compute()
da = submeso.sel(time=tis,method='nearest').sel(zC=slice(-600,zmax))['w'].compute()*86400
da = da.where(da<vm,vm).where(da>-vm,-vm)

b = submeso.sel(time=tis,method='nearest').sel(zC=slice(-600,zmax))["b"].compute()


# cmap = BuGnWYlRd.reversed()
cmap = cm.balance#pplt.Colormap('dusk_r', 'fire', name='Diverging', save=True)
kw = {
    "vmin":-vm,
    "vmax":vm,
    "levels":np.arange(-vm,vm+10,10),
    "cmap":cmap,
    "alpha":1.0,
}

kwb = {
    "levels":np.arange(b.min(),b.max()+0.1,0.5/1000),
    "colors":"0.2",
    "linewidths":0.5,
    "alpha":0.9,
    "zorder":1e10,
}

fig = plt.figure(figsize=np.array([14,8])*0.8)

ax = [fig.add_subplot(220+i,projection='3d') for i in np.arange(1,4+1)]

letters = ['a)','b)','c)','d)']
for i,a in enumerate(ax):
    letter = letters[i]
    ti = tis[i]
    
    # a.set_title(f'{letter}'+5*' '+f'{ti} days'+' '*5,rotation=10)
    a.text(-40, -100, 200, f'{letter}'+5*' '+f'{ti} days'+' '*5, ha="left", va="bottom", zdir="y")
    
    
    dsi = da.sel(time=ti,zC=0,method='nearest')
    bi = b.sel(time=ti,zC=0,method='nearest')
    dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
    C = a.contourf(dims[0],dims[1],dsi.values,zdir='z',offset=dsi.zC.max(),**kw)
    a.contour(dims[0],dims[1],bi.values,zdir='z',offset=0,**kwb)

    dsi = da.sel(time=ti,method='nearest').isel(yC=0)
    bi = b.sel(time=ti,method='nearest').isel(yC=0)
    dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
    C = a.contourf(dims[0],dsi.values,dims[1],zdir='y',offset=dsi.yC.values,**kw)
    a.contour(dims[0],bi.values,dims[1],zdir='y',offset=dsi.yC.values,**kwb)

    dsi = da.sel(time=ti,xC=da.xC.max(),method='nearest')
    bi = b.sel(time=ti,xC=da.xC.max(),method='nearest')
    dims = np.meshgrid(dsi[dsi.dims[1]].values,dsi[dsi.dims[0]].values)
    C = a.contourf(dsi.values,dims[0],dims[1],zdir='x',offset=dsi.xC.values,**kw)
    a.contour(bi.values,dims[0],dims[1],zdir='x',offset=dsi.xC.values,**kwb)
    di = mld.sel(time=ti,method='nearest')
    a.plot(di.yC*0+di.xC[-1],di.yC,-di.isel(xC=-1),color='magenta',zorder=1e4)
    a.plot(di.xC,di.xC*0+di.yC[0],-di.isel(yC=0),color='magenta',zorder=1e4)
    
    a.set(
        xlim=[da.xC.min(),da.xC.max()],
        ylim=[da.yC.min(),da.yC.max()],
        zlim=[da.zC.min(),da.zC.max()],
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
        zticks=[np.round(dsi.zC.max()),-150,-300,-450],
        box_aspect=[1,1,0.4]
    )
    
    a.view_init(30, -30)
    a.dist = 10

_ = [a.set(zlabel='') for a in [ax[0],ax[2]]]

cax = plt.axes([.42, .5, .2, .01])
cbar = fig.colorbar(C,cax=cax,orientation='horizontal')
cbar.set_ticks(np.arange(-vm,vm+100,100))
cbar.set_label(r'w [m/day]')

fig.subplots_adjust(wspace=0)


pax = [
    plt.axes([.05, .87, .1, .1]),
    plt.axes([.83, .87, .1, .1]),
    plt.axes([.05, .40, .1, .1]),
    plt.axes([.83, .40, .1, .1])
]

step = 10
for i,a in enumerate(pax):
    
    da.isel(zC=np.abs(da.zC+mld).argmin('zC')).isel(time=i).plot.hist(
        ax=a,bins=np.arange(-(vm-step),(vm-step)+step,step),density=True,color='0.4'
    )
    a.set(yticks=[],title='',xlabel='')
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_visible(False)
#     a.spines['bottom'].set_visible(False)

pax[0].text(0.2,0.8,r"pdf$\,\|_{mld}$",fontsize=12,ha='center',transform=pax[0].transAxes)
pax[0].set_xlabel(r'w [m/day]')

fig.tight_layout()
fig.savefig("../../reports/figures/3D_w.png", facecolor="w", dpi=300, bbox_inches="tight")  
