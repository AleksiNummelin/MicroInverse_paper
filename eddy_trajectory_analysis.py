import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import interpolate
import micro_inverse_utils as mutils
import NorESM_utils as utils
from mpl_toolkits.basemap import Basemap, addcyclic, interp, maskoceans
from matplotlib.colors import BoundaryNorm, LogNorm, SymLogNorm, from_levels_and_colors
from joblib import Parallel, delayed
import tempfile
import shutil
import os
import dist
#
#plot the length of trajectories as scatter plots
data=Dataset('/export/scratch/anummel1/EddyAtlas/eddy_trajectory_19930101_20160925.nc')
inds=np.where(np.diff(data.variables['track'][:]))[0]
#
inds0=range(len(inds))
inds0[1:]=inds[:-1]+1
times=data['time'][inds]-data['time'][inds0]
#
times_all=data['time'][:]
lat_all=data['latitude'][:]
lon_all=data['longitude'][:]
lon_all[np.where(lon_all>180)]=lon_all[np.where(lon_all>180)]-360
#
lat=data.variables['latitude'][inds0]
lon=data.variables['longitude'][inds0]
#
lat_end=data.variables['latitude'][inds]
lon_end=data.variables['longitude'][inds]
#
####################################################
# --- CALCULATE THE AVERAGE SPEED OF THE CORE  --- #
####################################################
#
def calc_speed_core_average(j,lat_all,lon_all,la_mean,lo_mean,inds0,inds,speed_core_average,u_core_average,v_core_average):
    '''Calculate the average speed of the core of the eddy '''
    #
    la        = lat_all[int(inds0[j]):int(inds[j]+1)]
    lo        = lon_all[int(inds0[j]):int(inds[j]+1)]
    speed_all = dist.distance((la[:-1],lo[:-1]),(la[1:],lo[1:]))*1E3/(24*3600.)
    u_all     = np.sign(lo[1:]-lo[:-1])*dist.distance((la[:-1],lo[:-1]),(la[:-1],lo[1:]))*1E3/(24*3600.)
    v_all     = np.sign(la[1:]-la[:-1])*dist.distance((la[:-1],lo[:-1]),(la[1:],lo[:-1]))*1E3/(24*3600.)
    speed_core_average[j]=np.mean(speed_all)
    u_core_average[j]=np.mean(u_all)
    v_core_average[j]=np.mean(v_all)
    #
    lam,lom    = utils.geographic_midpoint(la,lo,w=None);
    la_mean[j] = lam
    lo_mean[j] = lom
    #
    if j%10000==0:
       print j

def calc_speed_core_bin(j,lat_all,lon_all,inds0,inds,gy,gx,u_core_map,v_core_map):
    '''Calculate the speed of the core of the eddy and bin it'''
    #
    la        = lat_all[int(inds0[j]):int(inds[j]+1)]
    lo        = lon_all[int(inds0[j]):int(inds[j]+1)]
    #
    u_all     = np.sign(lo[1:]-lo[:-1])*dist.distance((la[:-1],lo[:-1]),(la[:-1],lo[1:]))*1E3/(24*3600.)
    v_all     = np.sign(la[1:]-la[:-1])*dist.distance((la[:-1],lo[:-1]),(la[1:],lo[:-1]))*1E3/(24*3600.)
    #
    fy=interpolate.interp1d(gy,np.arange(720))
    fx=interpolate.interp1d(gx,np.arange(1440))
    y=fy(0.5*(la[1:]+la[:-1])).astype(int)
    x=fx(0.5*(lo[1:]+lo[:-1])).astype(int)
    #
    for k in xrange(len(y)):
      c=len(np.where(~np.isnan(u_core_map[:,y[k],x[k]]))[0])
      if c>=u_core_map.shape[0]:
         continue
      u_core_map[c,y[k],x[k]]=u_all[k]
      v_core_map[c,y[k],x[k]]=v_all[k]
    #
    if j%1000==0:
      print j

def calc_core_bin(j,lat_all,lon_all,inds0,inds,gy,gx,core_map):
    '''Calculate core of the eddy and bin it'''
    #
    la        = lat_all[int(inds0[j]):int(inds[j]+1)]
    lo        = lon_all[int(inds0[j]):int(inds[j]+1)]
    #
    fy=interpolate.interp1d(gy,np.arange(720))
    fx=interpolate.interp1d(gx,np.arange(1440))
    y=fy(0.5*(la[1:]+la[:-1])).astype(int)
    x=fx(0.5*(lo[1:]+lo[:-1])).astype(int)
    #
    for k in xrange(len(y)):
      c=len(np.where(~np.isnan(core_map[:,y[k],x[k]]))[0])
      if c>=core_map.shape[0]:
         continue
      core_map[c,y[k],x[k]]=r_all[k]
    #
    if j%1000==0:
      print j


##########################################################
# ATTRIBUTING EDDY MEAN VELOCITY TO THE ITS MEAN POSITION
folder1 = tempfile.mkdtemp()
path1   = os.path.join(folder1, 'inds.mmap')
path2   = os.path.join(folder1, 'inds0.mmap')
path3   = os.path.join(folder1, 'lat_all.mmap')
path4   = os.path.join(folder1, 'lon_all.mmap')
path5   = os.path.join(folder1, 'speed_core_average.mmap')
path6   = os.path.join(folder1, 'la_mean.mmap')
path7   = os.path.join(folder1, 'lo_mean.mmap')
path8   = os.path.join(folder1, 'u_core_average.mmap')
path9   = os.path.join(folder1, 'v_core_average.mmap')
#
speed_core_average = np.memmap(path1, dtype=float, shape=(len(inds)), mode='w+')
inds0_m            = np.memmap(path2, dtype=float, shape=(len(inds)), mode='w+')
inds_m             = np.memmap(path3, dtype=float, shape=(len(inds)), mode='w+')
lon_all_m          = np.memmap(path4, dtype=float, shape=(len(lat_all)), mode='w+')
lat_all_m          = np.memmap(path5, dtype=float, shape=(len(lat_all)), mode='w+')
la_mean            = np.memmap(path6, dtype=float, shape=(len(inds)), mode='w+')
lo_mean            = np.memmap(path7, dtype=float, shape=(len(inds)), mode='w+')
u_core_average     = np.memmap(path8, dtype=float, shape=(len(inds)), mode='w+')
v_core_average     = np.memmap(path9, dtype=float, shape=(len(inds)), mode='w+')
#
speed_core_average[:] = np.zeros(len(inds))
u_core_average[:]     = np.zeros(len(inds))
v_core_average[:]     = np.zeros(len(inds))
inds0_m[:]            = inds0
inds_m[:]             = inds
lon_all_m[:]          = lon_all
lat_all_m[:]          = lat_all
la_mean[:]            = np.zeros(len(inds))
lo_mean[:]            = np.zeros(len(inds))
#
num_cores=20
Parallel(n_jobs=num_cores)(delayed(calc_speed_core_average)(j,lat_all_m,lon_all_m,la_mean,lo_mean,inds0_m,inds_m,speed_core_average,u_core_average,v_core_average) for j in range(len(inds)))
#
speed_core_average = np.array(speed_core_average)
u_core_average     = np.array(u_core_average)
v_core_average     = np.array(v_core_average)
la_mean            = np.array(la_mean)
lo_mean            = np.array(lo_mean)
#
try:
  shutil.rmtree(folder1)
except OSError:
  pass

############################################################################################
# CALCULATING THE SPEED AT WHICH THE EDDY CORE MOVES.
# SORT OF NEAREST NEIGHBOUR APPROACH: VELOCITIES ARE BINNED IN A PRE-DETERMINED GRID
# 
scale=2
ny=720/scale
nx=1440/scale
ne=250*(scale**2)
grid_x, grid_y = np.mgrid[-180:180:complex(0,nx), -90:90:complex(0,ny)]
#grid_y, grid_x = np.meshgrid(np.arange(-90,90+180./ny,180./ny),np.arange(-180,180+360./nx,360./nx))
#
folder2 = tempfile.mkdtemp()
path1   = os.path.join(folder2, 'inds0.mmap')
path2   = os.path.join(folder2, 'inds.mmap')
path3   = os.path.join(folder2, 'lat_all.mmap')
path4   = os.path.join(folder2, 'lon_all.mmap')
path5   = os.path.join(folder2, 'u_core_map.mmap')
path6   = os.path.join(folder2, 'v_core_map.mmap')
path7   = os.path.join(folder2, 'gx.mmap')
path8   = os.path.join(folder2, 'gy.mmap')
#
inds0_m    = np.memmap(path1, dtype=float, shape=(len(inds)), mode='w+')
inds_m     = np.memmap(path2, dtype=float, shape=(len(inds)), mode='w+')
lon_all_m  = np.memmap(path3, dtype=float, shape=(len(lat_all)), mode='w+')
lat_all_m  = np.memmap(path4, dtype=float, shape=(len(lat_all)), mode='w+')
u_core_map = np.memmap(path5, dtype=float, shape=(ne,ny,nx), mode='w+')
v_core_map = np.memmap(path5, dtype=float, shape=(ne,ny,nx), mode='w+')
gx         = np.memmap(path7, dtype=float, shape=(nx), mode='w+')
gy         = np.memmap(path8, dtype=float, shape=(ny), mode='w+')
#
inds0_m[:]            = inds0
inds_m[:]             = inds
lon_all_m[:]          = lon_all
lat_all_m[:]          = lat_all
u_core_map[:]         = np.ones((ne,ny,nx))*np.nan
v_core_map[:]         = np.ones((ne,ny,nx))*np.nan
gx[:]                 = grid_x[:,0]
gy[:]                 = grid_y[0,:]
#
#SERIAL VERSION WORKS MUCH BETTER???
for j in xrange(len(inds)):
   calc_speed_core_bin(j,lat_all,lon_all,inds0,inds,gy,gx,u_core_map,v_core_map)
#num_cores=20
#Parallel(n_jobs=num_cores)(delayed(calc_speed_core_bin)(j,lat_all,lon_all,inds0,inds,gy,gx,u_core_map,v_core_map) for j in xrange(len(inds)))
#
u_core_map  = np.array(u_core_map)
v_core_map  = np.array(v_core_map)
core_count  = u_core_map.copy(); core_count[np.where(~np.isnan(core_count))]=1; core_count=np.nansum(core_count,0)
mask=np.zeros(core_count.shape)
mask[np.where(core_count==0)]=1
#
np.savez('/home/anummel1/Projects/MicroInv/eddy_core_velocity_binned_scale'+str(scale)+'.npz', grid_x=grid_x.T, grid_y=grid_y.T, u_grid=np.nanmean(u_core_map,0), v_grid=np.nanmean(v_core_map,0),core_count=core_count, mask=mask)
#
try:
  shutil.rmtree(folder2)
except OSError:
  pass
#
##############################################################################################
method    = 'linear'
u_grid     = griddata((lo_mean,la_mean), u_core_average, (grid_x.T, grid_y.T), method=method)
v_grid     = griddata((lo_mean,la_mean), v_core_average, (grid_x.T, grid_y.T), method=method)
speed_grid = griddata((lo_mean,la_mean), speed_core_average, (grid_x.T, grid_y.T), method=method)
#
grid_x2=grid_x.T; grid_x2[ma.where(grid_x2>180)]=grid_x2[ma.where(grid_x2>180)]-360
mask=maskoceans(grid_x2, grid_y.T,np.zeros(u_grid.shape),inlands=False,resolution='c')
mask=1-mask.mask
mask[np.where(abs(grid_y.T)<5)]=1
icedata=np.load('/home/anummel1/Projects/MicroInv/icedata.npz')
icemask=np.round(icedata['icetot25'][:].T)
icemask2=np.zeros(icemask.shape)
icemask2[:,:720]=icemask[:,720:]
icemask2[:,720:]=icemask[:,:720]
#
mask[np.where(icemask2)]=1
#
u_grid=ma.masked_array(u_grid,mask=mask)
v_grid=ma.masked_array(v_grid,mask=mask)
speed_grid=ma.masked_array(speed_grid,mask=mask)
#SAVE DATA
#
np.savez('/home/anummel1/Projects/MicroInv/eddy_core_velocity.npz', grid_x=grid_x.T, grid_y=grid_y.T, icemask2=icemask2, u_grid=u_grid.data, v_grid=v_grid.data, mask=mask)
#
####################################################
# --- CALCULATE THE AVERAGE RADIUS IN PARALLEL --- #
####################################################
def calc_radius_average(j,radius,radius_average,inds0,inds):
    ''' '''
    #
    radius_average[j]=np.mean(radius[int(inds0[j]):int(inds[j]+1)])
    if j%10000==0:
       print j

folder1 = tempfile.mkdtemp()
path1 =  os.path.join(folder1, 'radius_ave.mmap')
path2 =  os.path.join(folder1, 'radius.mmap')
path3 =  os.path.join(folder1, 'inds.mmap')
path4 =  os.path.join(folder1, 'inds0.mmap')
path5 =  os.path.join(folder1, 'speed.mmap')
path6 =  os.path.join(folder1, 'speed_average.mmap')
#
radius         = np.memmap(path1, dtype=float, shape=(data.variables['speed_radius'].shape[0]), mode='w+')
radius_average = np.memmap(path2, dtype=float, shape=(len(inds)), mode='w+')
inds0_m        = np.memmap(path3, dtype=float, shape=(len(inds)), mode='w+')
inds_m         = np.memmap(path4, dtype=float, shape=(len(inds)), mode='w+')
speed          = np.memmap(path5, dtype=float, shape=(data.variables['speed_average'].shape[0]), mode='w+')
speed_average  = np.memmap(path6, dtype=float, shape=(len(inds)), mode='w+') 
#
radius[:]         = data.variables['speed_radius'][:]
radius_average[:] = np.zeros(len(inds))
speed[:]          = data.variables['speed_average'][:]
speed_average[:]  = np.zeros(len(inds))
inds0_m[:]        = inds0
inds_m[:]         = inds
num_cores=20
# AVERAGE RADIUS
Parallel(n_jobs=num_cores)(delayed(calc_radius_average)(j,radius,radius_average,inds0_m,inds_m) for j in range(len(inds)))
radius_average=np.array(radius_average)
# AVERAGE SPEED (THIS IS THE CURRENT SPEED)
Parallel(n_jobs=num_cores)(delayed(calc_radius_average)(j,speed,speed_average,inds0_m,inds_m) for j in range(len(inds)))
speed_average=np.array(speed_average)
#
try:
  shutil.rmtree(folder1)
except OSError:
  pass
