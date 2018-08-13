import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
#import micro_inverse_utils as mutils
import sys
sys.path.append('/home/anummel1/Projects/MicroInv/MicroInverse/')
from MicroInverse import MicroInverse_utils as mutils
#
import os
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import glob
#
#this script will first calculate smoothing weights and then apply them and create new set of smoothed files
#
calc_weights=False
apply_weights=True
model_data=False
smooth_highpass=False
#load a dummy field
if not model_data:
 filepath = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/amsr_avhrr/annual_files/'
 wpath    = '/datascope/hainegroup/anummel1/Projects/MicroInv/smoothing_weights/'
 outpath  = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/amsr_avhrr/'
elif not model_data and not smooth_highpass:
 filepath = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/annual_files/'
 wpath    = '/datascope/hainegroup/anummel1/Projects/MicroInv/smoothing_weights/'
 outpath  = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/'
elif model_data:
 #model_k='800'
 var      = 'sss'
 print(var)
 wpath    = '/datascope/hainegroup/anummel1/Projects/MicroInv/model_smoothing_weights/'
 outpath  = '/datascope/hainegroup/anummel1/Projects/MicroInv/model_data/newCO2_control_'+model_k+'_daily_smooth_'+var+'/'
 filepath = ''
 fnames   = glob.glob('/datascope/gnana_esms/newCO2_control_'+model_k+'_daily/history/*.ocean_daily.nc')
 fnames.sort()
elif smooth_highpass:
 print('smoothing highpass filtered data...')
 filepath = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/smooth_annual_files_y5.0deg_x10.0deg/'
 wpath    = '/datascope/hainegroup/anummel1/Projects/MicroInv/smoothing_weights/'
 outpath  = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/'
else:
 filepath = '/export/scratch/anummel1/sst_data/annual_files/'
 wpath    = '/export/scratch/anummel1/smoothing_weights/'
 outpath  = '/export/scratch/anummel1/sst_data/'

if not model_data:
  var='sst'
  lat_name='lat'
  lon_name='lon'
  fnames=os.listdir(filepath);
  fnames.sort()
  d1=Dataset(filepath+fnames[0])
  sst=d1.variables[var][0,:,:].squeeze()
  lat=d1.variables[lat_name][:]
  lon=d1.variables[lon_name][:]
if model_data:
  lat_name='yt_ocean'
  lon_name='xt_ocean'
  d1=Dataset(filepath+fnames[0])
  sst=d1.variables[var][0,:,:]
  lat=d1.variables[lat_name][:]
  lon=d1.variables[lon_name][:]
  lon[np.where(lon<-180)]=lon[np.where(lon<-180)]+360
  lon[np.where(lon>180)]=lon[np.where(lon>180)]-360
  lon,lat=np.meshgrid(lon,lat)

#first calculate weights
if calc_weights:
  for n in [1]:#[2,4]: #[8,10]: #[2,4,6]: #,10,15]:
     print(n)
     dum,weights_out=mutils.smooth2D_parallel(lon,lat,sst,n=n,num_cores=30,use_weights=True,weights_only=True,use_median=False,save_weights=True,save_path=wpath)

def save_smooth(lonin,latin,timein,data_smooth,var,outpath,outfile):
    """save data_smooth into a outfile """
    tlen,ylen,xlen = data_smooth.shape
    print(outpath+outfile)
    ncfile = Dataset(outpath+outfile, 'w', format='NETCDF4')
    #
    t = ncfile.createDimension('time', None)
    y = ncfile.createDimension('y', ylen)
    x = ncfile.createDimension('x', xlen)
    #
    lat    = ncfile.createVariable(lat_name,'f4',('y',))
    lon    = ncfile.createVariable(lon_name,'f4',('x',))
    time   = ncfile.createVariable('time','f8',('time',))
    nc_var = ncfile.createVariable(var,'f4',('time','y','x',))
    #
    lat[:]    = latin[:]
    lon[:]    = lonin[:]
    time[:]   = timein[:]
    nc_var[:] = data_smooth[:]
    #
    ncfile.close()

def smooth_file(ff,n,filepath,fname,outpath,t_inds2,weights_out,var='sst',model_data=None,smooth_highpass=None):
    #print(outpath+'smooth_highpass_y5deg_x10deg_annual_files_y'+str(2*n*0.25)+'deg_x'+str(2*2*n*0.25)+'deg/','year'+str(ff).zfill(2)+'_'+str(2*n*0.25)+'deg.nc')
    #print n, fname
    print(smooth_highpass)
    f0=Dataset(filepath+fname)
    data=f0.variables[var][:].squeeze()
    if smooth_highpass:
       filepath1='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/amsr_avhrr/annual_files/' 
       #'/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/annual_files/'
       fnames1=os.listdir(filepath1);
       fnames1.sort()
       f1=Dataset(filepath1+fnames1[ff])
       data1=f1.variables[var][:]
       data=data1-data
    #
    data_smooth=ma.masked_array(np.zeros(data.shape),mask=data.mask)
    jind,iind=ma.where(1-data[0,:,:].mask);
    for j in range(data.shape[0]):
       if n>10:
         #spread the calculation so less memory is used
         nn=len(jind)
         data_smooth[j,jind[:nn/4],iind[:nn/4]]=ma.sum(data[j,:,:].ravel()[list(t_inds2[:nn/4,:])]*weights_out[:nn/4,:,0],-1)
         data_smooth[j,jind[nn/4:nn/2],iind[nn/4:nn/2]]=ma.sum(data[j,:,:].ravel()[list(t_inds2[nn/4:nn/2])]*weights_out[nn/4:nn/2,:,0],-1)
         data_smooth[j,jind[nn/2:3*nn/4],iind[nn/2:3*nn/4]]=ma.sum(data[j,:,:].ravel()[list(t_inds2[nn/2:3*nn/4])]*weights_out[nn/2:3*nn/4,:,0],-1)
         data_smooth[j,jind[3*nn/4:],iind[3*nn/4:]]=ma.sum(data[j,:,:].ravel()[list(t_inds2[3*nn/4:])]*weights_out[3*nn/4:,:,0],-1)
       else:
         data_smooth[j,jind,iind]=ma.sum(data[j,:,:].ravel()[list(t_inds2)]*weights_out[:,:,0],-1)
    #
    if not model_data and not smooth_highpass:
      save_smooth(f0.variables[lon_name][:],f0.variables[lat_name][:],f0.variables['time'][:],data_smooth,var,outpath+'smooth_annual_files_y'+str(2*n*0.25)+'deg_x'+str(2*2*n*0.25)+'deg/',fname[:-3]+'_'+str(2*n*0.25)+'deg.nc')
    elif smooth_highpass:
      'saving...'
      #this is for smoothing highpass filtered data
      save_smooth(f0.variables[lon_name][:],f0.variables[lat_name][:],f0.variables['time'][:],data_smooth,var,outpath+'smooth_highpass_y5deg_x10deg_annual_files_y'+str(2*n*0.25)+'deg_x'+str(2*2*n*0.25)+'deg/','year'+str(ff).zfill(2)+'_'+str(2*n*0.25)+'deg.nc')
    else:
      save_smooth(f0.variables[lon_name][:],f0.variables[lat_name][:],f0.variables['time'][:],data_smooth,var,outpath+'smooth_annual_files_y'+str(2*n*0.25)+'deg_x'+str(2*2*n*0.25)+'deg/','year'+str(ff).zfill(2)+'_'+str(2*n*0.25)+'deg.nc')
    f0.close()
        
if apply_weights:
 for n in [8]:
 #for n in [16]: #[2,4]: #[8,10]: #[2,4,6]: #]
 #for n in [15]: #,10,15]:
 #for [5,6,10,15]:
   print(n)
   if n<6:
     n_cores=12
   elif n<10:
     n_cores=8
   elif n<14:
     n_cores=4
   else:
     n_cores=4
   #
   d2=np.load(wpath+str(n)+'_degree_smoothing_weights_coslat_y'+str(n)+'_x'+str(2*n)+'.npz')
   #weights_out=d2['weights_out'][:]
   #jind=d2['jind'][:]
   #iind=d2['iind'][:]
   #
   t_inds=ma.reshape(np.arange(sst.ravel().shape[0]),(sst.shape[0],sst.shape[1])) #create array of indices                                                                    
   #t_inds2=t_inds[list(weights_out[:,:,1]),list(weights_out[:,:,2])] 
   #
   folder1 = tempfile.mkdtemp()
   path1 =  os.path.join(folder1, 'weights_out.mmap')
   path2 =  os.path.join(folder1, 't_inds2.mmap')
   #
   weights_out=np.memmap(path1, dtype=float, shape=d2['weights_out'].shape, mode='w+')
   t_inds2=np.memmap(path2, dtype=int, shape=d2['weights_out'].shape[:2], mode='w+')
   #
   weights_out[:]=d2['weights_out'][:]
   t_inds2[:]=t_inds[weights_out[:,:,1].astype('int'),weights_out[:,:,2].astype('int')]
   #this will give you [len(inds),n**2] shaped array of indices matching to corresponding points in data.ravel()
   if n==15:
     Parallel(n_jobs=n_cores)(delayed(smooth_file)(ff,n,filepath,fname,outpath,t_inds2,weights_out,var=var,model_data=model_data,smooth_highpass=smooth_highpass) for ff,fname in enumerate(fnames[10:]))
   else:
     Parallel(n_jobs=n_cores)(delayed(smooth_file)(ff,n,filepath,fname,outpath,t_inds2,weights_out,var=var,model_data=model_data,smooth_highpass=smooth_highpass) for ff,fname in enumerate(fnames))
   try:
    shutil.rmtree(folder1)
   except OSError:
    pass 

    
