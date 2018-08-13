import numpy as np
import numpy.ma as ma
from matplotlib.colors import BoundaryNorm, LogNorm, SymLogNorm, from_levels_and_colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap, addcyclic, interp, maskoceans
import matplotlib.pyplot as plt
import NorESM_utils as utils
import esmf_utils as eutils
from netCDF4 import Dataset
import ESMF
from esmf_utils import grid_create, grid_create_periodic
from wquantiles import *
import string
#
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
#
def load_files(filepath,fnames):
  #load files given filepath and list of filenames#
  dataout={}
  for e,filename in enumerate(fnames):
    #print filename
    datain=np.load(filepath+filename)
    if e==0:
      variables=datain.keys()
      variables.sort()
      lmask=np.zeros(datain['V_global'].shape)
      lmask[ma.where(datain['V_global'][:]==0)]=1
    for var in variables:
      if var in ['Kx_global','Ky_global','R_global','U_global','V_global']:
        if e==0:
          exec('dataout["'+var+'"]=ma.masked_array(datain["'+var+'"][:],mask=lmask)')
        elif e>0:
          exec('dataout["'+var+'"]=ma.concatenate([dataout["'+var+'"],ma.masked_array(datain["'+var+'"][:],mask=lmask)],axis=0)')
      else:
        if e==0:
          exec('dataout["'+var+'"]=datain["'+var+'"][:]')
        elif e>0:
          exec('dataout["'+var+'"]=ma.concatenate([dataout["'+var+'"],datain["'+var+'"][:]],axis=0)')
  #
  return dataout

def combine_Taus(data,weight_coslat,Taus,K_lim=True):
   """Use the CFL criteria to limit Tau, at first choose the min dt (max spped) for each location."""
   #
   for key in ['U_global','V_global','Kx_global','Ky_global','R_global']:
      data[key]=np.reshape(data[key][:],(len(Taus),data[key].shape[0]/len(Taus),-1)).squeeze()
   #
   lmask=data['R_global'][0,:,:].mask
   #
   dt=ma.min((1/(abs(data['U_global'][:])/(weight_coslat*111E3*0.25)+abs(data['V_global'][:])/(111E3*0.25)))/(3600*24.),0)
   #make dt's to be integers
   dt[np.where(dt<Taus[0])]=Taus[0]
   for t,tau in enumerate(Taus[:-2]):
     dt[np.where(ma.logical_and(dt>tau,dt<=Taus[t+1]))]=Taus[t+1];
   dt[np.where(dt>Taus[-2])]=Taus[-1];
   if K_lim:
    c=0
    while c<max(Taus):
     c=c+1
     for t,tau in enumerate(Taus):
       dx=(weight_coslat*111E3*0.25)
       dy=111E3*0.25
       jinds,iinds=np.where(dt.squeeze()==tau)
       if len(jinds)>1:
         jindsX=np.where(data['Kx_global'][t,jinds,iinds].squeeze()*tau*3600*24/dx[jinds,iinds]**2>1)[0]
         if len(jindsX)>1:
           dt[jinds[jindsX],iinds[jindsX]]=Taus[max(t-1,0)]
         jindsY=np.where(data['Ky_global'][t,jinds,iinds].squeeze()*tau*3600*24/dy**2>1)[0]
         if len(jindsY)>1:
           dt[jinds[jindsY],iinds[jindsY]]=Taus[max(t-1,0)]
   #use dt and pick up each field given the location specific Tau
   for key in ['U_global','V_global','Kx_global','Ky_global','R_global']:
       dum2=np.zeros(data[key][0,:,:].shape)
       for j,ext in enumerate(Taus):
           jinds,iinds=np.where(dt.squeeze()==ext)
           dum2[jinds,iinds]=data[key][j,jinds,iinds].squeeze()
       data[key]=ma.masked_array(dum2,lmask)
   #
   return data

filepath='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/final_output/'
#
titles_0=['$\kappa_x$','$\kappa_y$','r','u','v','$\sqrt{u^2+v^2}$']
#
if True:
 #stypes=['smooth2','smooth'] #
 #titles=['Low Pass','High pass']
 stypes=['smooth','smooth2', 'smooth3'] #OSM2018 setup
 titles=['High Pass','Low Pass', 'High'+r'${\rightarrow}$'+'Low Pass']#OSM2018 setup
 #Taus=[3,5,7,10]
 Taus=[2,4,6,8,10]
 scales=[1E3,1E3,1,1E-2,1E-2,1E-2]
 minlims=[0.1,0.1,0,-15,-15,0]
 maxlims=[150,150,40,15,15,20]
 xlims=[0.0,5.25]
elif False:
 stypes=['time_ave','time_smooth1']
 titles=['Time average','Time Low Pass']
 Taus=[1]
 scales=[1E3,1E3,1,1E-2,1E-2,1E-2]
 minlims=[0,0,0,-5,-5,0]
 maxlims=[2,2,120,5,5,7.5]
 xlims=[0.0,30]
else:
 stypes=['tau_test','tau_test2']
 titles=['Spatial High Pass','Original data']
 Taus=[1]
 scales=[1E3,1E3,1,1E-2,1E-2,1E-2]
 minlims=[0,0,0,-5,-5,0]
 maxlims=[3,3,60,5,5,7.5]
 xlims=[0.0,30]
#
fig,axes=plt.subplots(nrows=2,ncols=3,sharex=True,figsize=(25,10))
#fig,axes=plt.subplots(nrows=2,ncols=3,sharex=False,figsize=(25,10)) #this is for OSM2018
#
icedata=np.load('/home/anummel1/Projects/MicroInv/icedata.npz')
icemask25=np.round(icedata['icetot25'][:,80:-40].T)
icemask1=np.round(icedata['icetot1'][:,20:-10].T)
icemask2=np.round(icedata['icetot2'][:,10:-5].T)
icemask3=np.round(icedata['icetot3'][:,7:-3].T)
icemask5=np.round(icedata['icetot5'][:,4:-2].T)
#
ylabs=['Diffusivity [1000 m$^2$ s$^{-1}$]','Diffusivity [1000 m$^2$ s$^{-1}$]','Decay [days]','Velocity [cm s$^{-1}$]','Velocity [cm s$^{-1}$]','Speed [cm s$^{-1}$]']
cols=['C1','C2','C3','C4']
#
#for tt,tau in enumerate(Taus):
for tt, stype in enumerate(stypes):
  print stype
  for var in ['Kx','Ky','R','U','V']:
    exec(var+'={}')
  weight_coslats={}
  if stype in ['coarse_ave']:
    exts=['0.25','1','2','3','5']; lbext='deg'
  if stype in ['time_ave']:
    exts=['1','5','10','15','20','30']; lbext='days'
  if stype in ['time_smooth1','time_smooth2']:
    exts=['1','5','10','15','20','30']; lbext='days'
  if stype in ['tau_test']:
    exts=['2','3','4','5','6','7','8','10','15','20','30']; lbext='days'
  if stype in ['tau_test2']:
    exts=['2','3','4','5','6','7','8','10','15','20','30']; lbext='days'
  if stype in ['smooth']:
    exts=['1.0','2.0','3.0','4.0','5.0']; lbext='deg'
  if stype in ['smooth2']:
    exts=['0.25','1.0','2.0','3.0','4.0','5.0']; lbext='deg'
  if stype in ['smooth3']:
    exts=['0.5','1.0','2.0','3.0']; lbext='deg'
  #for tau in Taus:
  #     meanfiles.append('uvkr_data_python_sst_method1_timeseries_sensitivity_30year_00_saveB_SpatialHighPass_y4.0deg_x8.0deg_Tau'+str(tau).zfill(2)+'.npz')
  for ee,ext in enumerate(exts):
    print ext
    files=[]
    if stype in ['coarse_ave','smooth','smooth2']:
       Taus=[3,5,7,10]
    elif stype in ['smooth3']:
       Taus=[2,4,6,8,10]
    elif stype in  ['time_ave','time_smooth1','time_smooth2'] and ext in ['1']:
       Taus=[3,5,7,10]
    elif stype in  ['time_ave','time_smooth1','time_smooth2','tau_test','tau_test2']:
       Taus=[1]
    for tau in Taus:
      if (stype in ['coarse_ave','smooth','smooth2'] and ext in ['0.25']): #or (stype in ['time_ave','time_smooth1','time_smooth2','tau_test'] and ext in ['1']):
        filename='uvkr_data_python_sst_integral_70S_80N_norot_saveB_Tau'+str(tau)+'.npz' #'uvkr_data_python_tau_'+str(tau)+'_remclim_norot.npz'
      elif (stype in ['time_ave','time_smooth1','time_smooth2'] and ext in ['1']):
        filename='uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_Tau'+str(tau)+'.npz'   
      elif stype in ['tau_test']:
        filename='uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_Tau'+ext+'.npz'
      elif stype in ['tau_test2']:
        filename='uvkr_data_python_sst_integral_70S_80N_norot_saveB_Tau'+ext+'.npz' #'uvkr_data_python_tau_'+ext+'_remclim_norot.npz' 
      elif stype in ['smooth']:
        filename='uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y'+ext+'deg_x'+str(2*float(ext))+'deg_Tau'+str(tau)+'.npz'
      elif stype in ['smooth2']:
        filename='uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialLowPass_y'+ext+'deg_x'+str(2*float(ext))+'deg_Tau'+str(tau)+'.npz'
      elif stype in ['smooth3']:
        filename='uvkr_data_python_sst_integral_70S_80N_norot_saveB_smooth_highpass_4deg_x8deg_y'+ext+'deg_x'+str(2*float(ext))+'deg_Tau'+str(tau)+'.npz'
      elif stype in ['time_ave']:
        filename='uvkr_data_python_ave_time_ave_dt'+ext+'_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_Tau1.npz'
      elif stype in ['time_smooth1']:
        filename='uvkr_data_python_time_smooth_win_len'+ext+'_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_TauIsFilterLength.npz'
      files.append(filename)
    #load data
    data=load_files(filepath,files)
    #combine Taus
    Lon_vector=data['Lon_vector'][:];
    Lat_vector=data['Lat_vector'][:];
    if len(Taus)>1:
       Lon_vector=Lon_vector[:np.where(np.diff(Lon_vector)<0)[0][0]+1]
       Lat_vector=Lat_vector[:np.where(np.diff(Lat_vector)<0)[0][0]+1]
    #
    weight_coslat=np.tile(np.cos(Lat_vector*np.pi/180.),(Lon_vector.shape[-1],1)).T
    if len(Taus)>1:
      A2=combine_Taus(data,weight_coslat,Taus)
    else:
      A2=data.copy()
    #
    lmask=A2['R_global'].mask
    #
    variables=A2.keys()
    variables.sort()
    for var in variables:
      exec(var+'=A2["'+var+'"][:]')
    if (stype in ['coarse_ave'] and ext in ['0.25']) or (stype not in ['coarse_ave']):
      lmask[ma.where(icemask25)]=1
    else:
      exec('lmask[ma.where(icemask'+ext+')]=1')
    #combine across
    for var in ['Kx','Ky','R','U','V']:
      if var in ['R']:
        exec(var+'["'+var+'_'+ext+'"]=(abs(ma.masked_array('+var+'_global,mask=lmask)/(3600*24)))')
      else:
        exec(var+'["'+var+'_'+ext+'"]=ma.masked_array('+var+'_global,mask=lmask)')
    #
    #weight_coslat=np.tile(np.cos(Lat_vector*np.pi/180.),(V_global.shape[-1],1)).T
    weight_coslats[ext]=weight_coslat #[ma.where(1-lmask)]
  if tt==0 and stype in ['smooth','smooth2','smooth3', 'coarse_ave']:
     xlab=fig.text(0.5,0.01,'Spatial filter [$\degree$]',fontsize=20,ha='center')
  elif tt==0 and stype in ['time_ave', 'time_smooth1','time_smooth2']:
     xlab=fig.text(0.5,0.01,'Time average/Time filter',fontsize=20,ha='center')
  elif tt==0 and stype in ['tau_test','tau_test2']:
     xlab=fig.text(0.5,0.01,'Tau [days]', fontsize=20,ha='center')
  extra_artists=[xlab]
  for j,var in enumerate(['Kx','Ky','R','U','V','UV']):
     ax=axes.flatten()[j]
     ax.set_title(titles_0[j],fontsize=20)
     txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=20)
     y=[];x=[]; y25=[]; y75=[]
     #
     for ee,ext in enumerate(exts):
       x.append(float(ext))
       exec('mask2=np.zeros(Kx["Kx_'+ext+'"].shape)')
       exec('mask2[np.where(Kx["Kx_'+ext+'"]<0)]=1')
       exec('mask2[np.where(Ky["Ky_'+ext+'"]<0)]=1')
       #
       if var in ['UV']:
         exec('mask=U["U_'+ext+'"].mask')
         mask=mask+mask2; mask[np.where(mask>1)]=1
         exec('u1=U["U_'+ext+'"][:][ma.where(1-mask)]')
         exec('v1=V["V_'+ext+'"][:][ma.where(1-mask)]')
         y.append(quantile_1D(ma.sqrt(u1**2+v1**2)/scales[j],weight_coslats[ext][ma.where(1-mask)],0.5))
         y25.append(quantile_1D(ma.sqrt(u1**2+v1**2)/scales[j],weight_coslats[ext][ma.where(1-mask)],0.25))
         y75.append(quantile_1D(ma.sqrt(u1**2+v1**2)/scales[j],weight_coslats[ext][ma.where(1-mask)],0.75))
       else:
         exec('mask='+var+'["'+var+'_'+ext+'"].mask')
         #if var in ['Kx','Ky']:
         #exec('mask2=np.zeros('+var+'["'+var+'_'+ext+'"].shape)')
         #exec('mask2[np.where('+var+'["'+var+'_'+ext+'"]<0)]=1')
         mask=mask+mask2; mask[np.where(mask>1)]=1
         exec('y.append(quantile_1D('+var+'["'+var+'_'+ext+'"][:][ma.where(1-mask)],weight_coslats[ext][ma.where(1-mask)],0.5)/scales[j])')
         exec('y25.append(quantile_1D('+var+'["'+var+'_'+ext+'"][:][ma.where(1-mask)],weight_coslats[ext][ma.where(1-mask)],0.25)/scales[j])')
         exec('y75.append(quantile_1D('+var+'["'+var+'_'+ext+'"][:][ma.where(1-mask)],weight_coslats[ext][ma.where(1-mask)],0.75)/scales[j])')
     if stype in ['coarse_ave','smooth','smooth2'] and var in ['Kx','Ky']:
       ax.semilogy(x,y,'-o',color=cols[tt],lw=3,label=titles[tt])
       ax.fill_between(x,y25,y75,color=cols[tt],alpha=0.3)
     else:
       ax.plot(x,y,'-o',color=cols[tt],lw=3,label=titles[tt])
       ax.fill_between(x,y25,y75,color=cols[tt],alpha=0.3)
     ax.set_ylim([minlims[j],maxlims[j]])
     ax.set_xlim(xlims)
     if j in [0,2,3,5]:
       ylab=ax.set_ylabel(ylabs[j],fontsize=20)
       if j in [2,5]:
         ax.yaxis.set_label_position("right")
         ax.yaxis.tick_right()
     extra_artists.append(ylab)
     extra_artists.append(txt1)

for j in range(2):
 for yy in [1E0,1E1,1E2]:
  axes.flatten()[j].axhline(y=yy,lw=2,color='gray',ls='--')

#execfile('load_ecco_diffusivities.py')
#axes.flatten()[0].plot(np.ones(3),Kredi_ECCO/1000,'ok')
#axes.flatten()[1].plot(np.ones(3),Kredi_ECCO/1000,'ok')
#
plt.legend()

if stype in ['time_ave', 'time_smooth1','time_smooth2']:
 plt.savefig('/home/anummel1/move_plots/medians_time_effects_new.png',format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
elif stype in ['tau_test','tau_test2']:
 plt.savefig('/home/anummel1/move_plots/medians_tau_effects_new_OSM2018.png',format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
else:
 plt.savefig('/home/anummel1/move_plots/medians_spatial_effects_new_v3.png',format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)


