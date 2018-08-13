#from save_CORE_data import load_CORE
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import NorESM_utils as utils
import micro_inverse_utils as mutils
from scipy.stats import pearsonr
from scipy.signal import detrend
import os
import matplotlib as mpl
mpl.use('Agg') #plot in screen session
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm, SymLogNorm, from_levels_and_colors
from mpl_toolkits.basemap import Basemap, addcyclic, interp, maskoceans
from joblib import Parallel, delayed
#from joblib import load, dump
import tempfile
import shutil
import dist
import cv2
from scipy.spatial import ConvexHull
#
#MAKE THIS SCRIPT TO WORK UNDER THE SAME LOOP AS INVERSION SCRIPT
#LOAD TIMESERIES WITH A HALO OF DJ GRID CELLS. 
#TO MAKE THINGS EASY SIMPLY LOAD THE FULL X DIMENSION BUT ONLY PART OF THE Y
#
#AFTER LOADING CALCULATE
#STANDARD DEVIATION, LOCAL CORRELATION (IN DJ,DI SPACE), AND A AUTOCORRELATION (WITH SAY 30-90 DAY LAGS)

#
#this is low res, download higher res
#
blknum=0;
lag=0
r2=0.25
ecco=False
if ecco:
  Data_directory  = '/home/anummel1/move_data/'
  Data_directory_clim  = '/home/anummel1/move_data/'
  plot_path       = '/home/anummel1/move_plots/'
  cor_path        = '/home/anummel1/move_data/'
  plot_name       = 'ecco_spatial_decorrelation'
  cor_file        = 'ecco_spatial_decorrelation_daily.npz'
  cor_axis_file   = 'ecco_spatial_decorrelation_axis_daily.npz'
  File_names      = ['THETA.0001.surface.nc']
  File_names_clim = ['THETA.0001.surface_clim.nc']
  Field_cdf_name=var='THETA'
  cmatrix_name    = 'corr_matrix_glob'
else:
  Data_directory  = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/annual_files/'
  #Data_directory  = '/datascope/hainegroup/anummel1/Projects/MicroInv/ssh_data/annual_files/'
  plot_path       = '/home/anummel1/move_plots/'
  cor_path        = '/datascope/hainegroup/anummel1/Projects/MicroInv/'
  Field_cdf_name=var='sst' #'sla' #sla
  plot_name       = 'obs_spatial_decorrelation_new_lag'+str(lag)+'_'+str(int(r2*100))+'_'+var+'highpass_y4.0deg_x8.0deg'
  if var in ['sla']:
    cor_file      = 'spatial_decorrelation_daily_new_lag'+str(lag)+'_'+var+'.npz'
  else:
    cor_file      = 'spatial_decorrelation_daily_new_lag'+str(lag)+'_highpass_y4.0deg_x8.0deg.npz'
  cor_axis_file   = 'spatial_decorrelation_axis_daily_new_lag'+str(lag)+'_'+str(int(r2*100))+var+'_highpass_y4.0deg_x8.0deg.npz'
  cor_lag_analysis_file = 'spatial_decorrelation_lag_analysis_'+var+'_r2_variable.npz'
  cmatrix_name    = 'corr_matrix_glob'
  #Field_cdf_name=var='sla' #'sst'
  File_names    = os.listdir(Data_directory)
  File_names.sort()
  print var, lag

calculate_cor=False
calculate_cor_axis=False
lag_analysis=False
plotting_load=False
plotting=True
#
if plotting:
 import matplotlib.pyplot as plt
 from matplotlib.colors import BoundaryNorm, LogNorm, SymLogNorm, from_levels_and_colors
 import matplotlib as mpl
if plotting or calculate_cor_axis:
 from mpl_toolkits.basemap import Basemap, addcyclic, interp, maskoceans      

if calculate_cor and not ecco: 
 dt=1
 Partition_rows  = 32
 Partition_cols  = 1
 ny=720
 nx=1440
 Block_row_size = int(np.ceil(ny/Partition_rows));
 Block_col_size = int(np.ceil(nx/Partition_cols));
 #
 dj=30;di=30
 corr_matrix_glob = np.zeros((ny,nx,dj,di))
 data0=Dataset(Data_directory+File_names[0])
 mask=data0[var][0,:,:].mask
if calculate_cor and ecco:
 dt=30
 Partition_rows  = 32
 Partition_cols  = 1
 ny=360
 nx=720
 Block_row_size = int(np.ceil(ny/Partition_rows));
 Block_col_size = int(np.ceil(nx/Partition_cols));
 #
 dj=15;di=15
 corr_matrix_glob = np.zeros((ny,nx,dj,di))
#
def main_loop(j,dj,di,corr_matrix2_mm,sst_woa25_anom_mm):
    """ """
    print j
    for i in range(nx):
     if np.isfinite(np.sum(sst_woa25_anom_mm[:,j,i])):
       jinds=np.arange(j-dj/2,j+dj/2);
       iinds=np.arange(i-dj/2,i+dj/2);
       jinds[np.where(jinds>=ny)]=ny-1
       jinds[np.where(jinds<0)]=0
       iinds[ma.where(iinds>=nx)]=iinds[ma.where(iinds>=nx)]-nx
       iinds[ma.where(iinds<0)]=iinds[ma.where(iinds<0)]+nx
       #ind=ma.where(np.isfinite(np.sum(sst_woa25_anom_mm[:,jinds,iinds],0)))[0]
       for j1,jj in enumerate(jinds): #[ind]):
         for i1,ii in enumerate(iinds): #[ind]):
             ind=np.where(np.isfinite(sst_woa25_anom_mm[:,jj,ii]))[0]
             if len(ind)>0:
               #r,p=pearsonr(sst_woa25_anom_mm[ind,j,i],sst_woa25_anom_mm[ind,jj,ii])
               r,p=pearsonr(sst_woa25_anom_mm[ind,j,i][:len(ind)-lag],sst_woa25_anom_mm[ind,jj,ii][lag:])
               corr_matrix2_mm[j,i,j1,i1]=r #ind[j1],ind[i1]]=r

if calculate_cor:
 for b_row in range(Partition_rows):
  rowStart = b_row*Block_row_size;
  for b_col in range(Partition_cols):
    colStart  = b_col*Block_col_size
    blknum=blknum+1;
    print 'calculating block '+str(blknum)+' of '+str(Partition_rows*Partition_cols)+' rows '+str(rowStart)+'-'+str(rowStart+Block_row_size)+ ' ,cols '+str(colStart)+'-'+str(colStart+Block_col_size)
    #
    block_rows      = np.arange(rowStart-dj/2,rowStart+Block_row_size+dj/2).astype('int')
    block_cols      = np.arange(colStart,colStart+Block_col_size).astype('int')
    block_cols[ma.where(block_cols<0)] = block_cols[ma.where(block_cols<0)]+nx
    block_cols[ma.where(block_cols>nx-1)] = block_cols[ma.where(block_cols>nx-1)]-nx
    block_rows[ma.where(block_rows<0)] = 0
    block_rows[ma.where(block_rows>ny-1)] = ny-1
    #
    iinds,jinds=np.meshgrid(block_cols,block_rows)
    iinds=iinds.flatten()
    jinds=jinds.flatten()
    if np.sum(1-mask[jinds,iinds])==0:
       continue
    num_cores=18; dim4D=True; sum_over_depth=False; depth_lim=0; depth_lim0=0; remclim=True; model_data=False
    if Field_cdf_name in ['sla']:
       remclim=False
    #
    if ecco:
      data1=Dataset(Data_directory+File_names[0])
      data2=Dataset(Data_directory_clim+File_names_clim[0])
      #nt=data1.variables['THETA'].shape[0]
      #sst_woa25=np.zeros((nt,len(jinds)))
      #for p in range(nt):
      #  print p
      sst_woa25=data1.variables['THETA'][:,0,block_rows,:].squeeze()
      sst_woa25_clim=data2.variables['THETA'][:,0,block_rows,:].squeeze()
      sst_woa25=np.reshape(sst_woa25,(sst_woa25.shape[0],-1))
      sst_woa25_clim=np.reshape(sst_woa25_clim,(sst_woa25_clim.shape[0],-1))
      print sst_woa25.shape
      print sst_woa25_clim.shape      
      sst_woa25=sst_woa25-np.tile(sst_woa25_clim,(sst_woa25.shape[0]/12,1))
    else:
      sst_woa25=mutils.load_data(Data_directory, File_names, jinds, iinds, Field_cdf_name, num_cores, dim4D, sum_over_depth, depth_lim, model_data=model_data,remove_clim=remclim,dt=dt, depth_lim0=depth_lim0)
      #
      if True:
        #Here is a way to high pass filter the data
        Data_directory_smooth='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/smooth_annual_files_y4.0deg_x8.0deg/'
        File_names_smooth=os.listdir(Data_directory_smooth)
        File_names_smooth.sort()
        sst_smooth=mutils.load_data(Data_directory_smooth, File_names_smooth, jinds, iinds, Field_cdf_name, num_cores, dim4D, sum_over_depth, depth_lim, model_data=model_data,remove_clim=remclim,dt=dt, depth_lim0=depth_lim0)
        sst_woa25=sst_woa25-sst_smooth
    #
    if not remclim:
      ninds=np.where(np.isfinite(np.sum(sst_woa25,0).squeeze()))[0]
      if len(ninds)<1:
        continue
      sst_woa25[:,ninds]=mutils.remove_climatology(sst_woa25[:,ninds],dt,num_cores=20)
    sst_woa25=np.reshape(sst_woa25,(sst_woa25.shape[0],len(block_rows),len(block_cols)))
    nt=sst_woa25.shape[0]
    ny2=sst_woa25.shape[1] 
    #observations
    #sst_woa25_anom=detrend(sst_woa25,axis=0)
    folder1 = tempfile.mkdtemp()
    #path1 =  os.path.join(folder1, 'mask.mmap')
    path2 =  os.path.join(folder1, 'corr_matrix2.mmap')
    path3 =  os.path.join(folder1, 'sst_woa25_anom.mmap')
    #mask = np.memmap(path1, dtype=float, shape=(ny2,nx), mode='w+')  
    corr_matrix2_mm = np.memmap(path2, dtype=float, shape=(ny2,nx,dj,di), mode='w+')
    sst_woa25_anom_mm =np.memmap(path3, dtype=float, shape=(nt,ny2,nx), mode='w+')
    #mask[:]=sst_woa25.mask[0,:,:]
    corr_matrix2_mm[:]=np.zeros((ny2,nx,dj,di))
    sst_woa25_anom_mm[:]=sst_woa25[:]
    #
    num_cores=30
    print 'calculating correlation...'
    Parallel(n_jobs=num_cores)(delayed(main_loop)(j,dj,di,corr_matrix2_mm,sst_woa25_anom_mm) for j in range(dj/2,ny2-dj/2))
    corr_matrix_glob[block_rows[dj/2:ny2-dj/2],:,:,:]=np.asarray(corr_matrix2_mm[dj/2:ny2-dj/2,:,:,:])
    try:
      shutil.rmtree(folder1)
    except OSError:
      pass
 
 print 'done - saving the data'
 np.savez(cor_path+cor_file,corr_matrix_glob=corr_matrix_glob)

if calculate_cor_axis:
  data=Dataset(Data_directory+File_names[0])
  lon=data.variables['lon'][:]
  lat=data.variables['lat'][:]
  cdat=np.load(cor_path+cor_file)
  #corr_matrix2=cdat['corr_matrix2'][:]
  ny,nx,dj,di=cdat[cmatrix_name].shape
  if len(lon.shape)<2:
    lon2,lat2=np.meshgrid(lon,lat)
  else:
    lon2=lon; lat2=lat
  #minor=np.zeros(lon2.shape);
  #major=np.zeros(lon2.shape);
  #angle=np.zeros(lon2.shape);
  #
  folder1 = tempfile.mkdtemp()
  path1 =  os.path.join(folder1, 'minor.mmap')  
  path2 =  os.path.join(folder1, 'major.mmap')
  path3 =  os.path.join(folder1, 'angle.mmap')
  path4 =  os.path.join(folder1, 'corr_matrix2.mmap')
  minor = np.memmap(path1, dtype=float, shape=(ny,nx), mode='w+')
  major = np.memmap(path2, dtype=float, shape=(ny,nx), mode='w+')
  angle = np.memmap(path3, dtype=float, shape=(ny,nx), mode='w+')
  corr_matrix2= np.memmap(path4, dtype=float, shape=(ny,nx,dj,di), mode='w+')
  minor[:]=np.zeros(lon2.shape);
  major[:]=np.zeros(lon2.shape);
  angle[:]=np.zeros(lon2.shape);
  corr_matrix2[:]=cdat[cmatrix_name][:]
  #for j in range(ny):
  def main_loop_ax(j,dj,di,corr_matrix2,minor,major,angle):
    print j
    for i in range(nx):
     if corr_matrix2[j,i,dj/2,di/2]:
       jinds0,iinds0=ma.where(corr_matrix2[j,i,:,:]*abs(corr_matrix2[j,i,:,:])>=r2)
       jinds1,iinds1=ma.where(corr_matrix2[j,i,:,:]*abs(corr_matrix2[j,i,:,:])<=r2)
       if len(jinds0)==0:
          jinds0=jinds1; iinds0=iinds1
       ind0=ma.where(corr_matrix2[j,i,jinds0,iinds0]>0)[0]
       #ind1=ma.where(corr_matrix2[j,i,jinds1,iinds1]>0)[0]
       #
       jinds0=jinds0[ind0]; iinds0=iinds0[ind0];
       #jinds1=jinds1[ind1]; iinds1=iinds1[ind1];
       #HERE IS ANOTHER APPROACH - DEFINE CONVEX HULL OF THE INNER POINTS AND FIT AN ELLIPSE
       if len(jinds0)>2 and not (ma.mean(jinds0)==ma.max(jinds0) or ma.mean(iinds0)==ma.max(iinds0)):
         hull = ConvexHull(np.reshape(np.concatenate([jinds0,iinds0]),(2,len(jinds0))).T)
         jinds0=jinds0[hull.vertices]
         iinds0=iinds0[hull.vertices]
       #
       cj,ci=dj/2,di/2 #central points #ma.where(corr_matrix2[j,i,:,:]*abs(corr_matrix2[j,i,:,:])==1)
       #make jinds,iinds to range from 0-ny, 0-nx, so that we can pick up the corresponding lon, lat
       jinds0=jinds0-cj+j
       iinds0=iinds0-ci+i
       #jinds1=jinds1-cj+j
       #iinds1=iinds1-ci+i
       #boundary points
       jinds0[np.where(jinds0>=ny)]=ny-1; jinds0[np.where(jinds0<0)]=0
       iinds0[ma.where(iinds0>=nx)]=iinds0[ma.where(iinds0>=nx)]-nx; iinds0[ma.where(iinds0<0)]=iinds0[ma.where(iinds0<0)]+nx
       #
       dest0=(lat2[jinds0,iinds0],lon2[jinds0,iinds0])  
       d0=dist.distance((np.ones(len(dest0[0]))*lat2[j,i],np.ones(len(dest0[0]))*lon2[j,i]),dest0)
       if len(d0)<=4 or ma.mean(jinds0)==ma.max(jinds0) or ma.mean(iinds0)==ma.max(iinds0):
         minor[j,i]=d0.max()
         major[j,i]=d0.max()
         dm=ma.where(d0==d0.max())[0][0]
         dx=dist.distance((lat2[j,i],lon2[j,i]),(lat2[j,i],lon2[jinds0,iinds0][dm]))*np.sign(lon2[jinds0,iinds0][dm]-lon2[j,i])
         dy=dist.distance((lat2[j,i],lon2[j,i]),(lat2[jinds0,iinds0][dm],lon2[j,i]))*np.sign(lat2[jinds0,iinds0][dm]-lat2[j,i])
         angle[j,i]=np.arctan2(dx,dy)*180/np.pi
       else:
         #m = Basemap(width=np.ceil(d0.max())*2,height=np.ceil(d0.max())*2,projection='aeqd',lat_0=lat2[j,i],lon_0=lon2[j,i])
         destY      = (lat2[jinds0,iinds0],np.ones(len(jinds0))*lon2[j,i])
         destX      = (np.ones(len(jinds0))*lat2[j,i],lon2[jinds0,iinds0])
         y          = np.sign(destY[0]-lat2[j,i])*dist.distance((np.ones(len(destY[0]))*lat2[j,i],np.ones(len(destY[0]))*lon2[j,i]),destY)
         x          = np.sign(destX[1]-lon2[j,i])*dist.distance((np.ones(len(destX[0]))*lat2[j,i],np.ones(len(destX[0]))*lon2[j,i]),destX)
         #x,y=m(lon2[jinds0,iinds0],lat2[jinds0,iinds0])
         x          = x-np.mean(x)
         y          = y-np.mean(y)
         points     = np.zeros((len(x),2)); points[:,0]=x; points[:,1]=y;
         ellipse    = cv2.fitEllipse((np.round(points)).astype('int'))
         minor[j,i] = ellipse[1][0]
         major[j,i] = ellipse[1][1]
         angle[j,i] = ellipse[2]
       #  a = mutils.fitEllipse(x,y)
       #  mj,mi=mutils.ellipse_axis_length(a)/1E3
       #  minor[j,i]=mi
       #  major[j,i]=mi
       #  angle[j,i]=mutils.ellipse_angle_of_rotation2(a)
       ##ind=ma.where(1-mask[jinds0,iinds0])[0]; jinds0=jinds0[ind]; iinds0=iinds0[ind]
       #
       #jinds1[np.where(jinds1>=ny)]=ny-1; jinds1[np.where(jinds1<0)]=0
       #iinds1[ma.where(iinds1>=nx)]=iinds1[ma.where(iinds1>=nx)]-nx; iinds1[ma.where(iinds1<0)]=iinds1[ma.where(iinds1<0)]+nx
       ##ind=ma.where(1-mask[jinds1,iinds1])[0]; jinds1=jinds1[ind]; iinds1=iinds1[ind]
       ##calculate the distance to the central point
       #dest0=(lat2[jinds0,iinds0],lon2[jinds0,iinds0])
       #dest1=(lat2[jinds1,iinds1],lon2[jinds1,iinds1])
       #d0=dist.distance((np.ones(len(dest0[0]))*lat2[j,i],np.ones(len(dest0[0]))*lon2[j,i]),dest0)
       #d1=dist.distance((np.ones(len(dest1[0]))*lat2[j,i],np.ones(len(dest1[0]))*lon2[j,i]),dest1)
       ##calculate the angle of the major axis
       #dm=ma.where(d0==d0.max())[0][0]
       #dx=dist.distance((lat2[j,i],lon2[j,i]),(lat2[j,i],lon2[jinds0,iinds0][dm]))*np.sign(lon2[jinds0,iinds0][dm]-lon2[j,i])
       #dy=dist.distance((lat2[j,i],lon2[j,i]),(lat2[jinds0,iinds0][dm],lon2[j,i]))*np.sign(lat2[jinds0,iinds0][dm]-lat2[j,i])
       ##minor-minimum of 0.7 contour; major:maximum of 0.7 contour
       #major[j,i]=d0.max();
       #if len(d1)==0:
       #    minor[j,i]=d0.max();
       #else:
       #    minor[j,i]=d1.min();
       #if minor[j,i]>major[j,i]:
       #   minor[j,i]=major[j,i]
       ##  minor[np.where(minor>major)]=major[np.where(minor>major)] #this can happen if the correlation ellipe is pretty much a circle
       ## 
       #angle[j,i]=np.arctan2(dx,dy)*180/np.pi
  #
  num_cores=6
  Parallel(n_jobs=num_cores)(delayed(main_loop_ax)(j,dj,di,corr_matrix2,minor,major,angle) for j in range(ny))
  minor=np.asarray(minor)
  major=np.asarray(major)
  angle=np.asarray(angle)
  corr_matrix2=np.asarray(corr_matrix2)
  try:
    shutil.rmtree(folder1)
  except OSError:
    pass
  #
  np.savez(cor_path+cor_axis_file,minor=minor,major=major,angle=angle,lat2=lat2,lon2=lon2)

if lag_analysis:
  print 'starting analysis'
  data=Dataset(Data_directory+File_names[0])
  #
  cor_file1        = 'spatial_decorrelation_daily_new.npz'
  cor_file2        = 'spatial_decorrelation_daily_new_lag'+str(1)+'.npz'
  cor_file3        = 'spatial_decorrelation_daily_new_lag'+str(2)+'.npz'
  cor_file4        = 'spatial_decorrelation_daily_new_lag'+str(3)+'.npz'
  #
  print 'load files'
  lon=data.variables['lon'][:]
  lat=data.variables['lat'][:]
  cdat1=np.load(cor_path+cor_file1)
  cdat2=np.load(cor_path+cor_file2)
  cdat3=np.load(cor_path+cor_file3)
  cdat4=np.load(cor_path+cor_file4)
  #
  ny,nx,dj,di=cdat1[cmatrix_name].shape
  if len(lon.shape)<2:
    lon2,lat2=np.meshgrid(lon,lat)
  else:
    lon2=lon; lat2=lat
  print 'create variables'
  folder1 = tempfile.mkdtemp()
  path1 =  os.path.join(folder1, 'area1.mmap')
  path2 =  os.path.join(folder1, 'area2.mmap')
  path3 =  os.path.join(folder1, 'area3.mmap')
  path4 =  os.path.join(folder1, 'area4.mmap')
  path5 =  os.path.join(folder1, 'clims5.mmap')
  folder2 = tempfile.mkdtemp()
  path21 =  os.path.join(folder2, 'corr_matrix1.mmap')
  path22 =  os.path.join(folder2, 'corr_matrix2.mmap')
  path23 =  os.path.join(folder2, 'corr_matrix3.mmap')
  path24 =  os.path.join(folder2, 'corr_matrix4.mmap')
  #
  print 'create mmap variables'
  area1 = np.memmap(path1, dtype=float, shape=(ny,nx), mode='w+')
  area2 = np.memmap(path2, dtype=float, shape=(ny,nx), mode='w+')
  area3 = np.memmap(path3, dtype=float, shape=(ny,nx), mode='w+')
  area4 = np.memmap(path4, dtype=float, shape=(ny,nx), mode='w+')
  clims = np.memmap(path5, dtype=float, shape=(ny,nx), mode='w+')
  corr_matrix1= np.memmap(path21, dtype=float, shape=(ny,nx,dj,di), mode='w+')
  corr_matrix2= np.memmap(path22, dtype=float, shape=(ny,nx,dj,di), mode='w+')
  corr_matrix3= np.memmap(path23, dtype=float, shape=(ny,nx,dj,di), mode='w+')
  corr_matrix4= np.memmap(path24, dtype=float, shape=(ny,nx,dj,di), mode='w+')
  #
  area1[:]=np.zeros(lon2.shape);
  area2[:]=np.zeros(lon2.shape);
  area3[:]=np.zeros(lon2.shape);
  area3[:]=np.zeros(lon2.shape);
  #
  clims[:]=np.ones(lon2.shape);
  #
  corr_matrix1[:]=cdat1[cmatrix_name][:]
  corr_matrix2[:]=cdat2[cmatrix_name][:]
  corr_matrix3[:]=cdat3[cmatrix_name][:]
  corr_matrix4[:]=cdat4[cmatrix_name][:]
  #
  def main_loop_lag_an(j,corr_matrix,area,clims,lat2):
    print j
    for i in range(nx):
     corr_m=corr_matrix[j,i,:,:]/np.max(corr_matrix[j,i,:,:])
     if np.max(corr_m):
       if clims[j,i]==1:
         jinds0=[]
         while len(jinds0)<9:
           jinds0,iinds0=ma.where(corr_m*abs(corr_m)>=clims[j,i])
           clims[j,i]=clims[j,i]-0.01
           if clims[j,i]<0.5:
              break
       else:
         jinds0,iinds0=ma.where(corr_m*abs(corr_m)>=clims[j,i])
       #jinds1,iinds1=ma.where(corr_m*abs(corr_m)<=0.8)
       #if len(jinds0)==0:
       #   jinds0=jinds1; iinds0=iinds1
       ind0=ma.where(corr_matrix2[j,i,jinds0,iinds0]>0)[0]
       jinds0=jinds0[ind0]; iinds0=iinds0[ind0];
       #sum over all the grid cells
       area[j,i]=np.sum((6371E3*0.25*np.pi/180.)*(6371E3*np.cos(lat2[jinds0,iinds0]*np.pi/180.)*0.25*np.pi/180.))
  #
  num_cores=15
  print 'loop'
  Parallel(n_jobs=num_cores)(delayed(main_loop_lag_an)(j,corr_matrix1,area1,clims,lat2) for j in range(ny))
  Parallel(n_jobs=num_cores)(delayed(main_loop_lag_an)(j,corr_matrix2,area2,clims,lat2) for j in range(ny))
  Parallel(n_jobs=num_cores)(delayed(main_loop_lag_an)(j,corr_matrix3,area3,clims,lat2) for j in range(ny))
  Parallel(n_jobs=num_cores)(delayed(main_loop_lag_an)(j,corr_matrix4,area4,clims,lat2) for j in range(ny))
  #
  print 'done with the loop'
  area1=np.asarray(area1)
  area2=np.asarray(area2)
  area3=np.asarray(area3)
  area4=np.asarray(area4)
  clims=np.asarray(clims)
  print 'save the data'
  np.savez(cor_path+cor_lag_analysis_file,area1=area1,area2=area2,area3=area3,area4=area4,clims=clims,lat2=lat2)
  #
  try:
    shutil.rmtree(folder1)
  except OSError:
    pass
  try:
    shutil.rmtree(folder2)
  except OSError:
    pass

if plotting_load:
  dat=np.load(cor_path+cor_axis_file)
  for var in dat.keys():
     exec(var+'=dat[var][:]')
  cdat=np.load(cor_path+cor_file)
  corr_matrix2=cdat[cmatrix_name][:]
  ny,nx,dj,di=corr_matrix2.shape
  #corr_matrix2mask=np.ones(corr_matrix2.shape);corr_matrix2mask[np.where(corr_matrix2>0)]=0
if plotting_load:
  mask=np.ones((ny,nx))
  mask[np.where(minor>0)]=0
  jinds,iinds=ma.where(1-mask)
  med_corr=np.zeros(mask.shape)
  for k in range(len(jinds)):
     submask=1-np.ceil(corr_matrix2[jinds[k],iinds[k],13:18,11:20].squeeze())
     med_corr[jinds[k],iinds[k]]=ma.median(ma.masked_array(corr_matrix2[jinds[k],iinds[k],13:18,11:20],mask=submask).flatten())
     #submask=1-np.ceil(corr_matrix2[jinds[k],iinds[k],:,:].squeeze())
     #med_corr[jinds[k],iinds[k]]=ma.median(ma.masked_array(corr_matrix2[jinds[k],iinds[k],:,:],mask=submask).flatten())
  #corr_matrix2mask=1-np.ceil(corr_matrix2);
  #med_corr=ma.median(ma.reshape(ma.masked_array(corr_matrix2,corr_matrix2mask),(ny,nx,dj*di)),-1)
if plotting:
 icedata=np.load('/home/anummel1/Projects/MicroInv/icedata.npz')
 icemask=np.round(icedata['icetot25'][:].T)
 lon2[ma.where(lon2>180)]=lon2[ma.where(lon2>180)]-360
 for ext in ['1','2','3']:
  print 'plotting'
  #Figure
  cmaps=[plt.cm.OrRd,plt.cm.OrRd,plt.cm.OrRd,plt.cm.OrRd]
  #
  if ext in ['1']:
    #levs=[np.array([0,0.2,0.4,0.6,0.7,0.8,0.9,1]),np.array([0,50,100,150,200,300,400,500]),np.array([0,50,100,150,200,300,400,500]),np.array([-180,-135.,-90.,-45.,0,45.,90.,135.,180.])]
    levs=[np.array([0,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,1]),np.array([0,50,100,150,200,250,300]),np.array([0,50,75,100,125,150,175,200]),np.array([-180,-135.,-90.,-45.,0,45.,90.,135.,180.])]
    titles=['Median Local Spatial Correlation', 'Major', 'Minor','Angle']
    clabs=['r', 'Distance [km]','Distance [km]','Angle to x [$degree$]']
    variables=[med_corr,major,minor,abs(angle)]
  #
  elif ext in ['2']:
    #levs=[np.array([0,0.2,0.4,0.6,0.7,0.8,0.9,1]),np.array([0,50,100,150,200,300,400,500]),np.array([0,50,100,150,200,300,400,500]), np.array([0.0,0.2,0.4,0.6,0.8,0.9,0.95,1.0])] #,np.array([0.,0.2,0.4,0.5,0.6,0.8,1.0])]
    levs=[np.array([0,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,1]),np.array([0,50,100,150,200,250,300]),np.array([0,50,75,100,125,150,175,200]),np.array([0.0,0.2,0.4,0.6,0.7,0.8,0.9,1.0])]
    titles=['Median Local Spatial Correlation', 'Major', 'Minor','Minor/Major']
    clabs=['r', 'Distance [km]','Distance [km]','Minor/Major [ratio 0-1]']
    variables=[med_corr,major,minor,minor/major]
  elif ext in ['3']:
    #levs=[np.array([0,0.2,0.4,0.6,0.7,0.8,0.9,1]),np.array([0,50,100,150,200,300,400,500]),np.array([0,50,100,150,200,300,400,500]),np.array([0.,25,50,75,100,250,500,750,1000])]
    levs=[np.array([0,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,1]),np.array([0,50,100,150,200,250,300]),np.array([0,50,75,100,125,150,175,200]),np.array([0.,10,20,30,50,100,150,200])]
    titles=['Median Local Spatial Correlation', 'Major', 'Minor','Major-Minor']
    clabs=['r', 'Distance [km]','Distance [km]','Distance [km]']
    variables=[med_corr,major,minor,major-minor]
  #
  fig,axes=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(20,10))
  extra_artists=[]
  for j, ax in enumerate(axes.flatten()):
    print j
    levels=levs[j]
    cmap=cmaps[j]
    cmlist=[];
    if j==1 or j==2:
      for cl in np.linspace(0,252,len(levels)): cmlist.append(int(cl))
      cmap, norm = from_levels_and_colors(levels,cmap(cmlist),extend='max');
    else:
      for cl in np.linspace(0,252,len(levels)-1): cmlist.append(int(cl))
      cmap, norm = from_levels_and_colors(levels,cmap(cmlist));
    #cmap.set_bad([.5,.5,.5])
    ax.set_rasterization_zorder(1);
    m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='i',ax=ax)
    m.fillcontinents(color='gray',lake_color='gray',zorder=0)
    m.drawparallels(np.arange(-90.,91.,30.))
    m.drawmeridians(np.arange(-180.,181.,60.))
    #m.drawmapboundary(fill_color='gray',zorder=0)
    ax.set_title(titles[j], fontsize=20)
    if ext in ['2'] and j==3:
      variables[j][ma.where(variables[j]>1)]=1
    c=m.pcolormesh(lon2,lat2,ma.masked_array(variables[j],mask),cmap=cmap,norm=norm,zorder=0,latlon=True)
    cbar=m.colorbar(mappable=c,ax=ax)
    txt=cbar.ax.set_ylabel(clabs[j], fontsize=20)
    cice=m.pcolormesh(lon2,lat2,ma.masked_array(icemask,1-icemask),cmap=plt.cm.Set2_r,latlon=True,rasterized=True)
    extra_artists.append(txt)
    ax.set_ylim(-80,80)
    ax.set_xlim(-180,180)
  #
  plt.savefig(plot_path+plot_name+ext+'.png',format='png', dpi=300, bbox_inches='tight', bbox_extra_artists=extra_artists)
  plt.savefig(plot_path+plot_name+ext+'.pdf',format='pdf', dpi=300, bbox_inches='tight', bbox_extra_artists=extra_artists)
  plt.close('all')
