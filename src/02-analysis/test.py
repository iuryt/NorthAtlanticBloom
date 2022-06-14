import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from tqdm import tqdm
import seawater as sw
from xhistogram.xarray import histogram

from cmocean import cm

# --- Loading data
#####

yslice = slice(-130, 130)

ds = xr.open_dataset("../../data/raw/output_submesoscale.nc")
ds = ds.assign_coords(time=ds.time.astype("float")*1e-9/86400)
ds = ds.assign_coords(xC=ds.xC*1e-3, yC=ds.yC*1e-3)
submeso = ds.sel(yC=yslice)




submeso.b.isel(time=0).mean(["xC", "yC"]).plot(y="zC")

# 0.08*tanh(0.005*(-618-z))+0.014*(z^2)/1e5+1027.45

# bg = lambda z: 0.147 * np.tanh( 2.6 * ( -z - 623 ) ) + 1027.6
bg = lambda z: -0.147 * np.tanh( 2.6 * ( z + 623 ) / 1000) + 1027.6
(bg(submeso.zC)-bg(submeso.zC).sel(zC=0,method="nearest")).plot(y="zC")
