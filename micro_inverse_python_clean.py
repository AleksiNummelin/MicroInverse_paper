import numpy as np
import numpy.ma as ma
import os
import sys
from scipy.signal import detrend, butter, filtfilt
# here I am using the most recent version of MicroInverse on my disk
# Please install it from github
sys.path.append('/home/anummel1/Projects/MicroInv/MicroInverse/MicroInverse/')
import MicroInverse_utils as mutils
import xarray as xr
import datetime
import glob
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                                        %
#%               NOTES                    %
#%                                        %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                                        %
#%              SETTINGS                  %
#%                                        %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
if True:
    GO_PARALLEL     = 1
    GO_SAVE         = 1            #save the results            
#
if not coarse_ave:
    ndp='0.25'

#constant values
Sec_day         = 60*60*24     #seconds in a day
Day_yr          = 365          #days in a year

#directory and file names
if OSTIA:
  Data_directory  = '/export/scratch/anummel1/OSTIA/GulfStream_subset/'
elif var in ['oa_sss']:
  Data_directory  = '/export/scratch/anummel1/sss_data/SMOS/annual_files/'
  dt=1
elif var in ['sss'] and not model_data:
  Data_directory  = '/export/scratch/anummel1/sss_data/Aquarius/'
  dt=7
elif amsre_avhrr_data:
  Data_directory  = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/amsr_avhrr/annual_files/' #'/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/annual_files/'
elif not model_data and not coarse_ave and not profiles and not tau_test and not smooth and (var not in ['sla']):
  #Data_directory  = '/export/scratch/anummel1/sst_data/annual_files/' # '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/annual_files_5deg/'
  Data_directory  = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/annual_files/'
elif var in ['sla']:
  Data_directory  = '/export/scratch/anummel1/ssh_data/annual_files/' #/datascope/hainegroup/anummel1/Projects/MicroInv/ssh_data/annual_files/'
elif coarse_ave:
  Data_directory  = '/export/scratch/anummel1/sst_data/annual_files_'+ndp+'deg/'
elif profiles:
  Data_directory  = '/datascope/hainegroup/anummel1/Projects/MicroInv/profile_data/ARMOR3D/'
  #Data_directory  = '/export/scratch/anummel1/temperature_data/ARMOR3D_10_15/'
  #Data_directory  = '/export/scratch/anummel1/temperature_data/ARMOR3D_0_5/'
  #Data_directory = '/export/scratch/anummel1/temperature_data/ARMOR3D/'
elif smooth:
  #Data_directory = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/smooth_annual_files_y'+smt+'deg_x'+str(2*float(smt))+'deg/'
  Data_directory = '/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/smooth_highpass_y5deg_x10deg_annual_files_y'+smt+'deg_x'+str(2*float(smt))+'deg/'
elif model_data:
  #model_k='400' #'aber2d'
  Data_directory = ''      
  File_names  = glob.glob('/datascope/gnana_esms/newCO2_control_'+model_k+'_daily/history/*.ocean_daily.nc')
  File_names.sort()

#
if profiles:
  outpath='/datascope/hainegroup/anummel1/Projects/MicroInv/seamounts/'
elif var in ['sst'] and not amsre_avhrr and not amsre_avhrr_data:
  outpath='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/final_output/' #
elif var in ['sst'] and (amsre_avhrr or amsre_avhrr_data):
  outpath='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/'
else:
  #outpath='/datascope/hainegroup/anummel1/Projects/MicroInv/'+var+'_data/'
  outpath='/export/scratch/anummel1/ssh_data/'
#outpath='/datascope/hainegroup/anummel1/Projects/MicroInv/ssh_data/'
#outpath='/export/scratch/anummel1/sst_data/'
if smooth:
  #Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_SpatialLowPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_Tau'+str(Tau)+'.npz';
  Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_smooth_highpass_4deg_x8deg_y'+smt+'deg_x'+str(2*float(smt))+'deg_Tau'+str(Tau)+'.npz';
  #Ukvr_data_fn    = outpath+'uvkr_data_python_smooth_'+smt+'deg_Tau3_norot.npz';
elif time_smooth:
  Ukvr_data_fn    = outpath+'uvkr_data_python_time_smooth_win_len'+str(win_len)+'_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_SpatialHighPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_TauIsFilterLength.npz';
elif coarse_ave:
  Ukvr_data_fn    = outpath+'uvkr_data_python_coarse_ave_'+str(ndp)+'_integral_1982_2016_Tau3_remclim_rotated.npz';
elif coarse:
  Ukvr_data_fn    = outpath+'uvkr_data_python_coarse_'+str(ndp)+'.npz';
elif decadal:
  Ukvr_data_fn    = outpath+'uvkr_data_python'+extens+'.npz';
elif time_ave:
  Ukvr_data_fn    = outpath+'uvkr_data_python_ave_time_ave_dt'+str(dt)+'_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_SpatialHighPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_Tau'+str(Tau)+'.npz';
elif model_data:
  Ukvr_data_fn    = outpath+'uvkr_data_python_ave_model_k'+model_k+'_'+var+'_integral_Tau'+str(Tau)+'_highpass.npz';
elif profiles:
  #Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_between_'+str(depth_lim0)+'_'+str(depth_lim)+'_remclim_norot_Tdz.npz';
  #Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_between_'+str(depth_lim0)+'_'+str(depth_lim)+'_remclim_norot_Tmean.npz';
  #Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_between_'+str(depth_lim0)+'_'+str(depth_lim)+'_remclim_norot_'+var[0].upper()+'mean.npz';
  Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_between_'+str(depth_lim0)+'_'+str(depth_lim)+'_remclim_norot_'+var[0].upper()+'mean_2004_2014_saveB_Tau'+str(Tau)+'.npz';
elif tau_test:
  #Ukvr_data_fn    = outpath+'uvkr_data_python_tau_'+str(Tau)+'_remclim_norot_DerivativeMethod.npz';
  Ukvr_data_fn    = outpath+'uvkr_data_python_tau_'+str(Tau)+'_remclim_norot_Method3.npz';
elif timeseries_sensitivity:
  Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_method1_timeseries_sensitivity_'+str(tts/365)+'year_'+str(tt0).zfill(2)+'_saveB_SpatialHighPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_Tau'+str(Tau).zfill(2)+'.npz';
elif OSTIA:
  Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_integral_OSTIA_GulfStream_norot.npz';
elif spatial_high_pass and not model_data and not amsre_avhrr and not amsre_avhrr_data:
  #Ukvr_data_fn=outpath+'uvkr_data_python_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_SpatialHighPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_Tau'+str(Tau)+'_rotated.npz';
  Ukvr_data_fn=outpath+'uvkr_data_python_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_SpatialHighPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_OptimizedTau_rotated.npz';
elif spatial_high_pass and amsre_avhrr:
  Ukvr_data_fn=outpath+'uvkr_data_python_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_SpatialHighPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_Tau'+str(Tau)+'_2003_2010.npz';
elif spatial_high_pass and amsre_avhrr_data:
  Ukvr_data_fn=outpath+'uvkr_data_python_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_SpatialHighPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_Tau'+str(Tau)+'_2003_2010_amsr_avhrr.npz';
#elif integral_method:
#  Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_integral_70S_80N_norot_saveB.npz';
#elif not integral_method:
#  Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_method3.npz';
else:
  Ukvr_data_fn    = outpath+'uvkr_data_python_'+var+'_'+inversion_method+'_70S_80N_norot_saveB_Tau'+str(Tau)+'.npz';
  #Ukvr_data_fn    = outpath+'/uvkr_data_python_'+var+'.npz';

#
if model_data:
  Lat_cdf_name    = 'yt_ocean'        #lat name in cdf file
  Lon_cdf_name    = 'xt_ocean'        #lat name in cdf file
elif (profiles and ndp in ['0.25']) or var in ['sss']:
  Lat_cdf_name    = 'latitude'        #lat name in cdf file
  Lon_cdf_name    = 'longitude'
else:
  Lat_cdf_name    = 'lat'        #lat name in cdf file
  Lon_cdf_name    = 'lon'        #lat name in cdf file

Time_cdf_name   = 'time'       #time name in cdf file
#
#chosen parameter values
if coarse_ave or model_data:
  Partition_rows  = 1            #row partions of global data
  Partition_cols  = 1            #col partions of global data
else:
  Partition_rows  = 4            #row partions of global data
  Partition_cols  = 4            #col partions of global data
if time_ave:
  Tau           = int(np.ceil(3./dt))
elif profiles:
  Tau           = Tau
elif tau_test:
  Tau           = Tau
elif time_smooth:
  Tau           = int(win_len)
elif spatial_high_pass:
  Tau           = Tau
elif model_data:
  Tau           = Tau
else:
  Tau           = Tau # assumed forcing decorelation time (samples) - 
#
if b_9point:
 Stencil_size    = 9
 Stencil_size    = 4
else:
 Stencil_size    = 5 #size of stencil 
 Stencil_center  = 2 #target element in neighbor stencil vector list - python is 0 based so 2 instead of 3
if model_data:
  Lon_range     = np.array([-300, 80])
elif (profiles and ndp in ['1.00']) or coarse_ave or OSTIA or var in ['oa_sss','sss']:
  Lon_range     = np.array([-180, 180])
else:
  Lon_range     = np.array([0, 360]) #np.array([260, 360])#np.array([260, 360])    #(NA large) range of longitude to use
Lat_range       = np.array([-70, 80]) #np.array([-70, 80]) #np.array([20, 50]) #np.array([0, 80])       #(NA lage) range of latitude to use
# --------------------------------
#this is a test for gulf stream area
#Lon_range       = np.array([275, 300])
#Lat_range       = np.array([25, 45])
#
y0=1980; m0=1; d0=1
Day_start       = datetime.datetime(y0,m0,d0) #0            #First day to use
Day_finish      = datetime.datetime(2020,m0,d0) #9999999      #Last day to use
#%Day_start       = 66480       #First day to use
#%Day_finish      = 67201       #Last day to use

#operations to perform
#if True: #not time_ave:
#  GO_PARALLEL     = 1
#  GO_SAVE         = 1            #save the results

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                                        %
#%            INITIALIZATION              %
#%                                        %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# load data filenames from data directory
if (not decadal) and (not time_ave) and (not model_data) and (not coarse_ave):
  File_names      = os.listdir(Data_directory) #os.listdir(Data_directory)   
  File_names.sort()
  if profiles:
    for kk in range(len(File_names)):
      if int(File_names[kk][-20:-16])>=2004: #1996:
        break
    File_names    = File_names[kk:]
  elif timeseries_sensitivity:
    #test timeseries sensitivity with a timeseries that is 30 year long, because that breaks down to many exact fractions (1,2,3,5,6,10,15; unlike 35 which has deviators 1,5,7) - actually use 32 year long timeseries
    for kk in range(len(File_names)):
      if int(File_names[kk][-10:-6])>=1985:
        break
    File_names    = File_names[kk:]
  if amsre_avhrr:
     File_names    = File_names[21:-6]
elif not model_data:
    File_names    = os.listdir(Data_directory)
    File_names.sort()
    #File_names = File_names[2:]
Num_data_files  = len(File_names)

if spatial_high_pass and not model_data:
  if amsre_avhrr_data:
     Data_directory_spatial_high_pass='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/amsr_avhrr/smooth_annual_files_y'+smt+'deg_x'+str(2*float(smt))+'deg/'
  else:
     Data_directory_spatial_high_pass='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/smooth_annual_files_y'+smt+'deg_x'+str(2*float(smt))+'deg/'
  File_names_spatial_high_pass=os.listdir(Data_directory_spatial_high_pass)
  File_names_spatial_high_pass.sort()
  if timeseries_sensitivity:
    for kk in range(len(File_names_spatial_high_pass)):
      if int(File_names_spatial_high_pass[kk][-17:-13])>=1985:
        break
    File_names_spatial_high_pass    = File_names_spatial_high_pass[kk:]
  if amsre_avhrr:
     File_names_spatial_high_pass    = File_names_spatial_high_pass[21:-6]
elif spatial_high_pass and model_data:
  Data_directory_spatial_high_pass='/datascope/hainegroup/anummel1/Projects/MicroInv/model_data/newCO2_control_'+model_k+'_daily_smooth_'+var+'/smooth_annual_files_y'+smt+'deg_x'+str(2*float(smt))+'deg/'
  File_names_spatial_high_pass=os.listdir(Data_directory_spatial_high_pass)
  File_names_spatial_high_pass.sort()
#
# load latitude, longitude information from first file
if var in['sss'] and not model_data:
  first_file      = xr.open_dataset(Data_directory+File_names[0],decode_times=False)
else:
  first_file      = xr.open_dataset(Data_directory+File_names[0]) #Dataset(Data_directory+File_names[0])
if coarse:
  Lat_vector_glob = first_file[Lat_cdf_name].values[::ndp] #first_file.variables[Lat_cdf_name][::ndp]
  Lon_vector_glob = first_file[Lon_cdf_name].values[::ndp] #first_file.variables[Lon_cdf_name][::ndp]
else:
  Lat_vector_glob = first_file[Lat_cdf_name].values #first_file.variables[Lat_cdf_name][:]
  Lon_vector_glob = first_file[Lon_cdf_name].values #first_file.variables[Lon_cdf_name][:]
if (coarse_ave and ndp not in ['0.25']) or (profiles and ndp not in ['0.25']):
  Lat_vector_glob=Lat_vector_glob[:,0]
  Lon_vector_glob=Lon_vector_glob[0,:]
if profiles:
 Lat_vector      = Lat_vector_glob[ma.where(ma.logical_and(Lat_vector_glob>=Lat_range[0],Lat_vector_glob<=Lat_range[1]))]
 Lon_vector      = Lon_vector_glob[ma.where(ma.logical_and(Lon_vector_glob>=Lon_range[0],Lon_vector_glob<=Lon_range[1]))]
else:
 Lat_vector      = Lat_vector_glob[ma.where(ma.logical_and(Lat_vector_glob>Lat_range[0],Lat_vector_glob<Lat_range[1]))]
 Lon_vector      = Lon_vector_glob[ma.where(ma.logical_and(Lon_vector_glob>Lon_range[0],Lon_vector_glob<Lon_range[1]))]
#
Num_lats_global = len(Lat_vector);
Num_lons_global = len(Lon_vector);
Num_pts_global  = Num_lats_global * Num_lons_global;
Lat_grid,Lon_grid = np.meshgrid(Lat_vector,Lon_vector);
Lon_grid        = Lon_grid.T     # transpose to suit rows/cols matrix format
Lat_grid        = Lat_grid.T     # note: field is upside down for image plots
#
# load time info from first file
if not profiles and not coarse_ave and not smooth and var not in ['sss']:
 Time_vector    = first_file[Time_cdf_name].values #first_file.variables[Time_cdf_name][:]
 Dt_days        = np.timedelta64(Time_vector[1]-Time_vector[0],'D').astype(int);
elif coarse_ave or smooth:
 Time_vector    = first_file[Time_cdf_name].values
 Dt_days        = Time_vector[1]-Time_vector[0]
elif profiles or var in ['sss']:
 Time_vector    = np.arange(1)
 Dt_days        = 7

if time_ave:
  Dt_secs       = Sec_day*dt;
else:
  Dt_secs       = Sec_day*(Dt_days); #Dt in sec

Samps_per_file = len(Time_vector);
# set partition sizes
Block_row_size = np.ceil(Num_lats_global/Partition_rows);
Block_col_size = np.ceil(Num_lons_global/Partition_cols);
#
# allocation for global sized variables
if timeseries_sensitivity:
 #Mn_global      = np.zeros((7,Num_lats_global,Num_lons_global));   #global mean of field
 U_global       = np.zeros((4,Num_lats_global,Num_lons_global));   #global east-west velocity (m/s)
 V_global       = np.zeros((4,Num_lats_global,Num_lons_global));   #global north-south velocity (m/s)
 Kx_global      = np.zeros((4,Num_lats_global,Num_lons_global));   #global east-west diffusivity (m^2/s)
 Ky_global      = np.zeros((4,Num_lats_global,Num_lons_global));   #global north-south diffusivity (m^2/s)
 Kxy_global     = np.zeros((4,Num_lats_global,Num_lons_global));   #global northeast-southwest diffusivity (m^2/s)
 Kyx_global     = np.zeros((4,Num_lats_global,Num_lons_global));   #global northwest-southeast diffusivity (m^2/s)
 R_global       = np.zeros((4,Num_lats_global,Num_lons_global));   #global horizontal velocity (sec)
 B_global       = np.zeros((4,5,Num_lats_global,Num_lons_global));
 #Mask_global    = np.zeros((7,Num_lats_global,Num_lons_global));   #global land/ice mask
else:
 Mn_global      = np.zeros((Num_lats_global,Num_lons_global));   #global mean of field 
 U_global       = np.zeros((Num_lats_global,Num_lons_global));   #global east-west velocity (m/s)
 V_global       = np.zeros((Num_lats_global,Num_lons_global));   #global north-south velocity (m/s)
 Kx_global      = np.zeros((Num_lats_global,Num_lons_global));   #global east-west diffusivity (m^2/s)
 Ky_global      = np.zeros((Num_lats_global,Num_lons_global));   #global north-south diffusivity (m^2/s)
 Kxy_global     = np.zeros((Num_lats_global,Num_lons_global));   #global northeast-southwest diffusivity (m^2/s)
 Kyx_global     = np.zeros((Num_lats_global,Num_lons_global));   #global northwest-southeast diffusivity (m^2/s)
 R_global       = np.zeros((Num_lats_global,Num_lons_global));   #global horizontal velocity (sec)
 Mask_global    = np.zeros((Num_lats_global,Num_lons_global));    #global land/ice mask 
 Inst_global    = np.zeros((Num_lats_global,Num_lons_global));
 B_global       = np.zeros((Stencil_size,Num_lats_global,Num_lons_global));
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                                        %
#%              MASTER LOOP               %
#%                                        %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%loop over partitioned rows
blknum=0;
print(dt, Dt_secs, Dt_days, Tau, Sec_day)
for b_row in range(Partition_rows):
  rowStart = b_row*Block_row_size;     #%increment row number
  #
  #%loop over partitioned cols
  for b_col in range(Partition_cols):
    colStart  = b_col*Block_col_size     #%increment column number
    #
    #display progress
    blknum=blknum+1;
    print('inverting block '+str(blknum)+' of '+str(Partition_rows*Partition_cols)+' rows '+str(rowStart)+'-'+str(rowStart+Block_row_size)+ ' ,cols '+str(colStart)+'-'+str(colStart+Block_col_size))
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%                                        %
    #%            BLOCK SETUP                 %
    #%                                        %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%identify block rows
    if True:
      block_rows      = np.arange(rowStart-1,rowStart+Block_row_size+1).astype('int')
      block_cols      = np.arange(colStart-1,colStart+Block_col_size+1).astype('int')
      #%check for borders
      block_cols[ma.where(block_cols<0)] = block_cols[ma.where(block_cols<0)]+Num_lons_global #returns index starting from the last index
      block_cols[ma.where(block_cols>Num_lons_global-1)] = block_cols[ma.where(block_cols>Num_lons_global-1)]-Num_lons_global #returs index starting from 0 ->
      block_rows[ma.where(block_rows<0)] = 0 #no looping over in y
      block_rows[ma.where(block_rows>Num_lats_global-1)] = Num_lats_global-1 #no looping over in y
      #
    #%identify lats, lons, times, and sizes
    block_rows      = np.unique(block_rows)
    block_lon       = Lon_grid[block_rows,:][:,block_cols]
    block_lat       = Lat_grid[block_rows,:][:,block_cols]
    block_num_lats  = len(block_rows);
    block_num_lons  = len(block_cols);
    block_num_points= block_num_lons*block_num_lats;
    block_num_samp  = np.min([Samps_per_file*Num_data_files, np.ceil((np.timedelta64(Day_finish-Day_start,'D').astype(int)+1)/Dt_days)]);
    ###########################################
    #
    #            MAIN CALCULATION
    #
    ###########################################
    if GO_PARALLEL:
      #
      if profiles:
        lat_ind= ma.where(ma.logical_and(Lat_vector_glob>=Lat_range[0],Lat_vector_glob<=Lat_range[1]))[0][block_rows]
        lon_ind= ma.where(ma.logical_and(Lon_vector_glob>=Lon_range[0],Lon_vector_glob<=Lon_range[1]))[0][block_cols]
      else:
        lat_ind= ma.where(ma.logical_and(Lat_vector_glob>Lat_range[0],Lat_vector_glob<Lat_range[1]))[0][block_rows]
        lon_ind= ma.where(ma.logical_and(Lon_vector_glob>Lon_range[0],Lon_vector_glob<Lon_range[1]))[0][block_cols]
      #
      iinds,jinds=np.meshgrid(lon_ind,lat_ind)
      iinds=iinds.flatten()
      jinds=jinds.flatten()
      #
      if profiles:
        num_cores=18; dim4D=False; sum_over_depth=True; #depth_lim=13;
      elif model_data:
        num_cores=12; dim4D=True; sum_over_depth=False; depth_lim=13; depth_lim0=0
      elif var in ['sss']:
        num_cores=12; dim4D=False; sum_over_depth=False; depth_lim=13; depth_lim0=0
      else:
        #num_cores=18
        num_cores=12; dim4D=True; sum_over_depth=False; depth_lim=13; depth_lim0=0
      #LOAD DATA AND REMOVE CLIMATOLOGY
      if OSTIA or profiles or var in ['oa_sss','sss','sla'] or spatial_high_pass:
        remclim=False
      else:
        remclim=True
      x_grid,x_clim=mutils.load_data(Data_directory, File_names, jinds, iinds, Field_cdf_name, num_cores, dim4D, sum_over_depth, depth_lim, model_data,remove_clim=remclim,dt=dt, depth_lim0=depth_lim0)
      #
      if spatial_high_pass:
         x_grid_filt,x_filt_clim=mutils.load_data(Data_directory_spatial_high_pass, File_names_spatial_high_pass, jinds, iinds, Field_cdf_name, num_cores, dim4D, sum_over_depth, depth_lim, model_data,remove_clim=remclim,dt=dt, depth_lim0=depth_lim0)
         x_grid=x_grid-x_grid_filt
      if not remclim and False:
        #remove mean
        xmean=np.nanmean(x_grid,0)
        x_grid=x_grid-xmean
      elif not remclim and False:
        ninds=np.where(np.isfinite(np.sum(x_grid,0).squeeze()))[0]
        if len(ninds)>0:
          x_grid[:,ninds]=mutils.remove_climatology(x_grid[:,ninds],dt,num_cores=18)
        else:
          break
      elif not remclim and True:
        ninds=np.where(np.isfinite(np.sum(x_grid,0).squeeze()))[0]
        if len(ninds)>0:
          b,a,=butter(3,1/360.,btype='highpass')
          x_grid[:,ninds] = filtfilt(b, a, x_grid[:,ninds], axis=0)
          x_grid=x_grid[180:-180,:]
          block_num_samp = x_grid.shape[0]
      if time_smooth:
        ninds=np.where(np.isfinite(np.sum(x_grid,0).squeeze()))[0]
        b,a=butter(3,2./win_len)
        x_grid[:,ninds] = filtfilt(b, a, x_grid[:,ninds], axis=0)
        dum=int(np.ceil(win_len*0.5))
        x_grid=x_grid[dum:-dum,:]
        x_grid[:,ninds]=detrend(x_grid[:,ninds],axis=0,type='linear')
        block_num_samp = x_grid.shape[0]
      #reshape
      x_grid=np.reshape(x_grid,(x_grid.shape[0],len(block_rows),len(block_cols)))
      #swapaxes
      x_grid=np.swapaxes(np.swapaxes(x_grid,0,2),0,1)
      #
      #adjust length of time series
      block_num_samp = x_grid.shape[-1]
      #
      if model_data and False:
         x_grid2=np.ones(x_grid.shape)
         dum=x_grid[:,:,0];
         mask=np.zeros(dum.shape); mask[np.where(np.isfinite(dum))]=0
         dum=ma.masked_array(dum,mask)
         Lon_vector2=Lon_vector.copy()
         Lon_vector2[np.where(Lon_vector>180)]=Lon_vector2[np.where(Lon_vector>180)]-360
         lon2,lat2=np.meshgrid(Lon_vector2,Lat_vector)
         dum,weights_out=mutils.smooth2D_parallel(lon2,lat2,dum,n=4,num_cores=15,use_weights=True,weights_only=True,use_median=False,save_weights=False,save_path='')
         jind,iind=ma.where(mask);
         x_grid2=ma.sum(ma.reshape(x_grid,(x_grid.shape[0]*x_grid.shape[1],x_grid[3]))[list(t_inds2),:]*weights_out[:,:,0],0) #ma.sum(x_grid.ravel()[list(t_inds2)]*weights_out[:,:,0],-1)
      #
      if time_ave and dt!=1:
         x_grid=utils.timeMean(x_grid.T,year0=year0,xtype=xtype,dt=dt)
         x_grid=x_grid.T
         jinds,iinds=ma.where(~np.isnan(np.sum(x_grid,-1)))
         x_grid[jinds,iinds,:]=detrend(x_grid[jinds,iinds,:],axis=-1,type='linear')
         block_num_samp = x_grid.shape[-1] #this needs to be calculated again
      #INVERT!
      if timeseries_sensitivity:
        #tts=365*7
        #for tt,tts in enumerate(np.array([365,365*2,365*3,365*5,365*10,365*20])):
        for tt in range(tt0,tt0+U_global.shape[0]): #if done in multiple steps i.e 2 and 1 years
        #for tt in range(x_grid.shape[-1]/tts):
            print(tt*tts/365,(tt*tts+tts)/365)
            if tt*tts+tts>x_grid.shape[-1]:
               'exiting'
               break
            block_num_samp=tts
            #U_block,V_block,Kx_block,Ky_block,Kxy_block,Kyx_block,R_block=mutils.inversion(x_grid[:,:,:tts],block_rows,block_cols,block_lon,block_lat,block_num_lons,block_num_lats,block_num_samp,Stencil_center,Stencil_size,integral_method,Tau,Dt_secs)
            U_block,V_block,Kx_block,Ky_block,Kxy_block,Kyx_block,R_block,B_block=mutils.inversion(x_grid[:,:,tt*tts:tt*tts+tts],block_rows,block_cols,block_lon,block_lat,block_num_lons,block_num_lats,block_num_samp,Stencil_center,Stencil_size,Tau,Dt_secs,inversion_method=inversion_method)
            Browsp = block_rows[1:-1];
            Bcolsp = block_cols[1:-1];
            #
            U_global[tt-tt0,ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=U_block
            V_global[tt-tt0,ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=V_block
            Kx_global[tt-tt0,ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Kx_block
            Ky_global[tt-tt0,ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Ky_block
            R_global[tt-tt0,ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=R_block
            #
            B_global[tt-tt0,:,ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=B_block 
      else:
        print('invert')
        U_block,V_block,Kx_block,Ky_block,Kxy_block,Kyx_block,R_block,B_block=mutils.inversion(x_grid,block_rows,block_cols,block_lon,block_lat,block_num_lons,block_num_lats,block_num_samp,Stencil_center,Stencil_size,Tau,Dt_secs,inversion_method=inversion_method)
        Browsp = block_rows[1:-1];
        Bcolsp = block_cols[1:-1];
        #
        U_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=U_block
        V_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=V_block
        Kx_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Kx_block
        Ky_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Ky_block
        Kxy_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Kxy_block
        Kyx_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=Kyx_block
        R_global[ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=R_block
        B_global[:,ma.min(Browsp):ma.max(Browsp)+1,ma.min(Bcolsp):ma.max(Bcolsp)+1]=B_block

if GO_SAVE:
  #SAVE THE DATA
  print('save')
  np.savez(Ukvr_data_fn,U_global=U_global,V_global=V_global,Kx_global=Kx_global,Ky_global=Ky_global,Kxy_global=Kxy_global,Kyx_global=Kyx_global,R_global=R_global,B_global=B_global,Lat_vector=Lat_vector,Lon_vector=Lon_vector);
