import numpy as np
import numpy.ma as ma
import matplotlib as mpl
#mpl.use('Agg') #use this if running in the background                             
#from matplotlib.colors import BoundaryNorm, LogNorm, SymLogNorm, 
from matplotlib.colors import from_levels_and_colors
import matplotlib.pyplot as plt
#import matplotlib as mpl
from mpl_toolkits.basemap import Basemap #, addcyclic, interp, maskoceans
import matplotlib.pyplot as plt
#import NorESM_utils as utils
#import esmf_utils as eutils
#from netCDF4 import Dataset
import string
from matplotlib.patches import Ellipse
from scipy.interpolate import griddata
#
def load_files(filepath,fnames):
  #load files given filepath and list of filenames
  dataout={}
  for e,filename in enumerate(fnames):
    print filename
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
   for key in ['U_global','V_global','Kx_global','Ky_global','R_global']:
      data[key]=np.reshape(data[key][:],(len(Taus),data[key].shape[0]/len(Taus),-1)).squeeze()
   lmask=data['R_global'][0,:,:].mask
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
           dt[jinds[jindsX],iinds[jindsX]]=Taus[max(t-1,0)] #np.max([dt[jinds[jindsX],iinds[jindsX]]-1,np.ones(len(jindsX))*Taus[0]],axis=0)
         jindsY=np.where(data['Ky_global'][t,jinds,iinds].squeeze()*tau*3600*24/dy**2>1)[0]
         if len(jindsY)>1:
           dt[jinds[jindsY],iinds[jindsY]]=Taus[max(t-1,0)] #np.max([dt[jinds[jindsY],iinds[jindsY]]-1,np.ones(len(jindsY))*Taus[0]],axis=0)
   #use dt and pick up each field given the location specific Tau
   for key in ['U_global','V_global','Kx_global','Ky_global','R_global']:
       dum2=np.zeros(data[key][0,:,:].shape)
       for j,ext in enumerate(Taus):
           jinds,iinds=np.where(dt.astype('int')==ext)
           dum2[jinds,iinds]=data[key][j,jinds,iinds].squeeze()
           #dum2[jinds,iinds]=ma.max(data[key][:j+1,jinds,iinds].squeeze(),0)
       data[key]=ma.masked_array(dum2,lmask)
   #
   return data, ma.masked_array(dt.astype('int'),lmask)

def add_ellipse(ax,x,y,kx,ky,mask,Kxrange,Kyrange,color='k'):
    ''' '''
    if (not np.isnan(kx)) and (not np.isnan(ky)) and (not mask) and ma.logical_and(ky>Kyrange[0],ky<Kyrange[-1]) and ma.logical_and(kx>Kxrange[0],kx<Kxrange[-1]): # and abs(y)<60:
       wax=2.5 #*np.cos(y*np.pi/180.)
       hax=wax*ky/kx
       #if hax>1:
       #   hax=hax/np.log(hax)
       #   wax=wax/np.log(hax)
       if hax>2.0*wax:
          hax=2.0*wax #*hax/hax
          wax=hax*kx/ky #2.0*wax/hax
       #angle=np.arctan(ky/kx)*180/np.pi-90
       angle=0
       e=Ellipse(xy=(x,y),width=wax,height=hax,angle=angle,facecolor='none',edgecolor=color,rasterized=True)
       ax.add_artist(e)
       ax.plot(x,y,'.k',markersize=1)
    
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
cmaps=[plt.cm.RdBu_r,plt.cm.Reds,plt.cm.Reds,plt.cm.Reds,plt.cm.RdBu_r,plt.cm.RdBu_r,plt.cm.RdBu_r]
#
execfile('plot_autocorrelation.py')
plt.close('all')
t1=load_autocorrelation_data('/datascope/hainegroup/anummel1/Projects/MicroInv/',['autocorrelation_dt_1_highpass.npz'],[1])
t1=t1.squeeze()
t1mask=t1.copy(); t1mask[np.where(t1mask<=2)]=0; t1mask[np.where(t1mask!=0)]=1; t1mask=1-t1mask
t1mask=ma.masked_array(t1mask,mask=t1.mask)
t1mask=t1mask[80:-40,:]
#
y0=1982 #1982
ye=2016 #2012
plot_speed=False
plot_diffu=False
plot_decay=False
saveplot=False
#
dt=1
coarse_ave=False
model_data=False
profiles=False
smooth=False
dyears=35
vari='sst'
smt='4.0'
plotname='/home/anummel1/move_plots/uvkr_data_python_sst_integral_70S_80N_norot_SpatialHighPass_y'+smt+'deg_x'+str(2*float(smt))+'deg_over_all_Tau'
filepath='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/final_output/'
Taus=[2,3,4,5,6,7,8,10,15,20,30]
#
icedata=np.load('/home/anummel1/Projects/MicroInv/icedata.npz')
icemask=np.round(icedata['icetot25'][:,80:-40].T)
#
#load data
#Kx=np.ones((4,600,1440))
#Ky=np.ones((4,600,1440))
#U=np.ones((4,600,1440))
#V=np.ones((4,600,1440))
#R=np.ones((4,600,1440))
##
fnames=[]
for j,ext in enumerate(Taus):
  fnames.append('uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_Tau'+str(ext)+'.npz')
  #data=np.load('/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/final_output/uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_Tau'+str(ext)+'.npz')
  #Kx[j,:,:]=data['Kx_global'][:]
  #Ky[j,:,:]=data['Ky_global'][:]
  #U[j,:,:]=data['U_global'][:]
  #V[j,:,:]=data['V_global'][:]
  #R[j,:,:]=data['R_global'][:]
  #
data=load_files(filepath,fnames)
Lon_vector=data['Lon_vector'][:];
Lat_vector=data['Lat_vector'][:];
#
Lon_vector=Lon_vector[:np.where(np.diff(Lon_vector)<0)[0][0]+1]
Lat_vector=Lat_vector[:np.where(np.diff(Lat_vector)<0)[0][0]+1]
#
weight_coslat=np.tile(np.cos(Lat_vector*np.pi/180.),(Lon_vector.shape[-1],1)).T
A2,optimized_tau=combine_Taus(data,weight_coslat,Taus)
#Lon_vector=data['Lon_vector'][:]
#Lat_vector=data['Lat_vector'][:]
#
Lon_vector[ma.where(Lon_vector>180)]=Lon_vector[ma.where(Lon_vector>180)]-360
Lon_vector[ma.where(Lon_vector<-180)]=Lon_vector[ma.where(Lon_vector<-180)]+360
#
for var in ['Kx','Ky','R','U','V']:
  exec(var+'=A2["'+var+'_global"][:]')
#
lmask=np.zeros(V.shape)
lmask[ma.where(V==0)]=1
for var in ['Kx','Ky','R','U','V']:
  exec(var+'=ma.masked_array('+var+',mask=lmask)')

if True:
 fig,axes=plt.subplots(nrows=3,ncols=1,sharex=True,sharey=True,figsize=(25,30))
 if plot_speed:
  #################
  # --- SPEED --- #
  #################
  print 'plotting speed...'
  #fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
  ax=axes.flatten()[0]
  ax.set_rasterization_zorder(1)
  cbarloc='right'
  if dt>360:
    levels=np.array([0,0.0025,0.005,0.01,0.02,0.03])
  elif coarse_ave:
    levels=np.array([0,0.01,0.02,0.03,0.04,0.05,0.1,0.25,0.5,0.75,1])
  else:
    levels=np.array([0,0.01,0.02,0.03,0.04,0.05,0.1,0.25,0.5])*100
  cmap=cmaps[1]
  cmlist=[];
  for cl in np.linspace(0,252,len(levels)): cmlist.append(int(cl))
  cmap, norm = from_levels_and_colors(levels,cmap(cmlist),extend='max');
  cmap.set_bad([.5,.5,.5])
  #
  m1 = Basemap(projection='cyl',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax)
  m1.drawmapboundary()
  p1=m1.drawparallels(np.arange(-60.,61,20.),labels=[True,False,False,False],linewidth=0.25,fontsize=20);
  p2=m1.drawmeridians(np.arange(60.,360.,60.),labels=[False,False,False,True],linewidth=0.25,fontsize=20);
  m1.fillcontinents(color='DarkGray',lake_color='LightGray',zorder=0);
  speed=ma.sqrt(ma.sum([U**2,V**2],0)) #ma.max(ma.sqrt(ma.sum([U**2,V**2],0)),0)
  #
  u=U.copy();  u[:,:720]=U[:,720:]; u[:,720:]=U[:,:720];
  v=V.copy();  v[:,:720]=V[:,720:]; v[:,720:]=V[:,:720];
  mask2=lmask+icemask #+t1mask.data; 
  mask2[np.where(mask2>1)]=1
  mask=mask2.copy(); mask[:,:720]=mask2[:,720:]; mask[:,720:]=mask2[:,:720]
  u=ma.masked_array(u,mask)
  v=ma.masked_array(v,mask)
  lon=Lon_vector.copy(); lon[:720]=Lon_vector[720:]; lon[720:]=Lon_vector[:720]
  lat=Lat_vector.copy();
  lon,lat=np.meshgrid(lon,lat)
  x,y=m1(lon,lat)
  #
  c1=m1.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),speed*100,cmap=cmap,norm=norm,latlon=True,rasterized=True)
  lon2,lat2=np.meshgrid(Lon_vector.squeeze(),Lat_vector.squeeze())
  x2,y2=m1(lon2[ma.where(t1mask)],lat2[ma.where(t1mask)])
  m1.plot(x2,y2, '.',color='gray', markersize=0.2, rasterized=True)
  ax.streamplot(x,y,u,v,color='k',linewidth=1,density=3) #,arrowstyle='->')
  cice=m1.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),ma.masked_array(icemask,1-icemask),cmap=plt.cm.Set2_r,latlon=True,rasterized=True)
  if model_data:
    ndx=4; ndy=4
  else:
    ndx=20; ndy=20
  scaling=np.sqrt(speed)/2.
  #m1.quiver(Lon_vector.squeeze()[::ndx],Lat_vector.squeeze()[::ndy],(U_global/scaling)[::ndx,::ndy],(V_global/scaling)[::ndx,::ndy],latlon=True,pivot='mid',scale=7,units='inches',rasterized=True,headlength=2,headwidth=2)
  cbar1=m1.colorbar(mappable=c1,location=cbarloc)
  ylab=cbar1.ax.set_ylabel('Speed [cm s$^{-1}$]', fontsize=25)
  ttl=ax.set_title(str(y0)+'-'+str(ye),fontsize=25)
  extra_artists=[ylab,ttl]
  for p in p1.iterkeys():
   du=p1[p][1];
   for jj in du:
    extra_artists.extend([jj]);
  for p in p2.iterkeys():
   du=p2[p][1];
   for jj in du:
    extra_artists.extend([jj]);
  
  #plt.savefig(plotname+'_speed.png',format='png',dpi=600,bbox_inches='tight',bbox_extra_artists=extra_artists)
  #plt.savefig(plotname+'_speed.pdf',format='pdf',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
  #plt.savefig(plotname+'_speed.pdf',dpi=300)
 if plot_diffu:
  #######################
  # --- DIFFUSIVITY --- #
  #######################
  print 'plotting diffusivity...'
  #fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
  ax=axes.flatten()[1]
  ax.set_rasterization_zorder(1)
  cbarloc='right'
  if vari in ['ssh']:
    levels=np.array([-0.5,-0.25,-0.1,0.0,0.1,0.25,0.5])
    cmap=cmaps[0]
  elif model_data:
    levels=np.array([-15,-10,-5,-1,1,5,10,25,50])
    cmap=cmaps[0]
  elif profiles:
    levels=np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.6,0.8,1,2,4])
    cmap=cmaps[1]
  elif smooth or coarse_ave:
    levels=np.array([0.0,0.01,0.1,0.25,0.5,0.75,1,2,4,6,8,10,15,20])
    cmap=cmaps[1]
  else:
    levels=np.array([0.0,0.1,0.25,0.5,0.75,1,2,3])
    #levels=np.array([0.0,0.01,0.1,0.25,0.5,0.75,1,1.5,2,3])#
    #levels=np.array([0.0,0.01,0.1,0.25,0.5,0.75,1,2,3,4,6,8,10,15,20])
    cmap=cmaps[1]
  cmlist=[];
  for cl in np.linspace(0,252,len(levels)): cmlist.append(int(cl))
  cmap, norm = from_levels_and_colors(levels,cmap(cmlist),extend='max');
  cmap.set_bad([.5,.5,.5])
  #
  #Kb = 1/2.*(Kx_global+Ky_global)/1000.
  #Kb = 1/2.*np.sign(ma.max(Kx,0)+ma.max(Ky,0))*np.sqrt((ma.max(Kx,0)**2+ma.max(Ky,0)**2))/1000.
  Kb = 1/2.*np.sign(Kx+Ky)*np.sqrt((Kx**2+Ky**2))/1000.
  m2 = Basemap(projection='cyl',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax)
  m2.drawmapboundary()
  p1=m2.drawparallels(np.arange(-60.,61,20.),labels=[True,False,False,False],linewidth=0.25,fontsize=20);
  p2=m2.drawmeridians(np.arange(60.,360.,60.),labels=[False,False,False,True],linewidth=0.25,fontsize=20);
  m2.fillcontinents(color='DarkGray',lake_color='LightGray',zorder=0);
  #c2=m2.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),Kb,cmap=cmap,norm=norm,latlon=True,rasterized=True)
  #m2.plot(x2,y2, '.',color='gray', markersize=0.2, rasterized=True)
  #
  lo,la=np.meshgrid(Lon_vector.squeeze(),Lat_vector.squeeze())
  #x3,y3=m2(lo,la)
  Kxrange=np.percentile(Kx[np.where(Kx>10)],[1,99])
  Kyrange=np.percentile(Ky[np.where(Ky>10)],[1,99])
  jinds,iinds=np.where(ma.logical_and(Kx>Kxrange[0],Kx<Kxrange[1]))
  Kx2=griddata((lo[jinds,iinds],la[jinds,iinds]),Kx[jinds,iinds],(lo,la))
  jinds,iinds=np.where(ma.logical_and(Ky>Kyrange[0],Ky<Kyrange[1]))
  Ky2=griddata((lo[jinds,iinds],la[jinds,iinds]),Ky[jinds,iinds],(lo,la))
  newmask=Kb.mask.copy()
  newmask2=Kb.mask.copy()
  newmask3=Kb.mask.copy()
  newmask2[np.where(Kx<Kxrange[0])]=1;# newmask2[np.where(Kx>Kxrange[1])]=1
  newmask2[np.where(Ky<Kyrange[0])]=1;# newmask2[np.where(Ky>Kyrange[1])]=1
  newmask2[np.where(icemask)]=1
  newmask[np.where(icemask)]=1
  #newmask3[np.where(Kx<0)]=1;
  #newmask3[np.where(Ky<0)]=1;
  n=4
  if False:
     import micro_inverse_utils as mutils
     dum,weights_out1=mutils.smooth2D_parallel(Lon_vector,Lat_vector,ma.masked_array(Ky2,mask=newmask2),n=n,num_cores=15,use_weights=True,weights_only=True,use_median=False,save_weights=False,save_path='')
     dum,weights_out2=mutils.smooth2D_parallel(Lon_vector,Lat_vector,ma.masked_array(newmask2,mask=newmask),n=n,num_cores=15,use_weights=True,weights_only=True,use_median=False,save_weights=False,save_path='')
     np.savez('/export/scratch/anummel1/smoothing_weights/KyKx_smooth_'+str(n)+'.npz',weights_out1=weights_out1)
     np.savez('/export/scratch/anummel1/smoothing_weights/newmask_smooth_'+str(n)+'.npz',weights_out2=weights_out2)
  #
  weight_data1=np.load('/export/scratch/anummel1/smoothing_weights/KyKx_smooth_'+str(n)+'.npz')
  weight_data2=np.load('/export/scratch/anummel1/smoothing_weights/newmask_smooth_'+str(n)+'.npz')
  weights_out1=weight_data1['weights_out1'][:]
  weights_out2=weight_data2['weights_out2'][:]
  t_inds=ma.reshape(np.arange(Ky.ravel().shape[0]),(Ky.shape[0],Ky.shape[1]))
  t_inds2=t_inds[weights_out1[:,:,1].astype('int'),weights_out1[:,:,2].astype('int')]
  Ky2_smooth=ma.masked_array(np.zeros(Ky2.shape),mask=newmask2)
  Kx2_smooth=ma.masked_array(np.zeros(Ky2.shape),mask=newmask2)
  mask_smooth=ma.masked_array(np.zeros(Ky2.shape),mask=newmask)
  jind,iind=ma.where(1-newmask2);
  Ky2_smooth[jind,iind]=ma.sum(Ky2.ravel()[list(t_inds2)]*weights_out1[:,:,0],-1)
  Kx2_smooth[jind,iind]=ma.sum(Kx2.ravel()[list(t_inds2)]*weights_out1[:,:,0],-1)
  jind,iind=ma.where(1-newmask);
  t_inds2=t_inds[weights_out2[:,:,1].astype('int'),weights_out2[:,:,2].astype('int')]
  mask_smooth[jind,iind]=ma.sum(newmask2.ravel()[list(t_inds2)]*weights_out2[:,:,0],-1)
  #
  #print Kxrange, Kyrange
  #for j in xrange(0,Kx.shape[0],30):
  #    for i in xrange(10,Kx.shape[1],30):
  #        add_ellipse(ax,x3[j,i],y3[j,i],Kx2[j,i],Ky2[j,i],newmask[j,i],Kxrange,Kyrange,color='k')
  Kymask=Kb.mask.copy()
  Kxmask=Kb.mask.copy()
  Kymask[np.where(Ky<0)]=1;
  Kxmask[np.where(Kx<0)]=1;
  Ky2Kx2_smooth=Ky2_smooth/Kx2_smooth
  Ky2Kx2_smooth[np.where(np.round(mask_smooth))]=np.nan
  c2=m2.pcolormesh(lo,la,ma.masked_array(ma.mean([ma.masked_array(Ky,Kymask),ma.masked_array(Kx,Kxmask)],0).data/1E3,mask=newmask),cmap=cmap,norm=norm,latlon=True,rasterized=True)
  m2.plot(x2,y2, '.',color='gray', markersize=0.2, rasterized=True)
  m2.contour(lo,la,ma.masked_array(Ky2_smooth/Kx2_smooth,mask=np.round(mask_smooth)),levels=[2.0],colors='k',linestyles='-',linewidths=1.5,latlon=True,rasterized=True)
  m2.contour(lo,la,ma.masked_array(Ky2_smooth/Kx2_smooth,mask=np.round(mask_smooth)),levels=[0.5],colors='k',linestyles='-',linewidths=1.5,latlon=True,rasterized=True)
  m2.contour(lo,la,ma.masked_array(Ky2_smooth/Kx2_smooth,mask=np.round(mask_smooth)),levels=[2.0],colors='C4',linestyles='-',linewidths=.75,latlon=True,rasterized=True)
  m2.contour(lo,la,ma.masked_array(Ky2_smooth/Kx2_smooth,mask=np.round(mask_smooth)),levels=[0.5],colors='C9',linestyles='-',linewidths=.75,latlon=True,rasterized=True)
  cice=m2.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),ma.masked_array(icemask,1-icemask),cmap=plt.cm.Set2_r,latlon=True,rasterized=True)
  cbar2=m2.colorbar(mappable=c2,location=cbarloc)
  ylab=cbar2.ax.set_ylabel('Diffusivity [1000 m$^2$ s$^{-1}$]',fontsize=25)
  ttl=ax.set_title(str(y0)+'-'+str(ye),fontsize=25)
  extra_artists.extend([ylab,ttl])
  for p in p1.iterkeys():
   du=p1[p][1];
   for jj in du:
    extra_artists.extend([jj]);
  for p in p2.iterkeys():
   du=p2[p][1];
   for jj in du:
    extra_artists.extend([jj]);
 
 if plot_decay:
  ###################
  # ---- DECAY ---- #
  ###################
  print 'plotting decay...'
  #fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
  ax=axes.flatten()[2]
  ax.set_rasterization_zorder(1)
  cbarloc='right'
  if vari in ['ssh']:
    levels=np.array([-120,-90,-60,-30,-10,10,30,60,90,120])
    levels=np.sign(levels)*np.sqrt(abs(levels))
    cmap=cmaps[0]
  else:
    if dt<360 and not profiles:
      levels=np.sqrt(np.array([0,2.5,5,10,15,20,30,40,50,60]))
    elif profiles:
      levels=np.sqrt(np.array([0,30,60,90,120,150,180,210,270,360,420,480]))
    else:
      levels=np.sqrt(np.array([0,180,360,540,720,900,1080]))
    cmap=cmaps[1]
  cmlist=[];
  for cl in np.linspace(0,252,len(levels)+1): cmlist.append(int(cl))
  cmap, norm = from_levels_and_colors(levels,cmap(cmlist),extend='both');
  cmap.set_bad([.5,.5,.5])
  #
  m3 = Basemap(projection='cyl',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax)
  m3.drawmapboundary()
  p1=m3.drawparallels(np.arange(-60.,61,20.),labels=[True,False,False,False],linewidth=0.25,fontsize=20);
  p2=m3.drawmeridians(np.arange(60.,360.,60.),labels=[False,False,False,True],linewidth=0.25,fontsize=20);
  m3.fillcontinents(color='DarkGray',lake_color='LightGray',zorder=0);
  c3=m3.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),np.sign(R)*ma.sqrt(abs(R/(24*3600.))),cmap=cmap,norm=norm,latlon=True,rasterized=True)
  m3.plot(x2,y2, '.',color='gray', markersize=0.2, rasterized=True)
  cice=m3.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),ma.masked_array(icemask,1-icemask),cmap=plt.cm.Set2_r,latlon=True,rasterized=True)
  cbar3=m3.colorbar(mappable=c3,location=cbarloc)
  ticklabs=[]
  for t,tt in enumerate(levels):
    ticklabs.append(str(int(tt**2)))
  #
  print ticklabs
  cbar3.ax.set_yticklabels(ticklabs)
  ylab=cbar3.ax.set_ylabel('Decay [days]',fontsize=25)
  ttl=ax.set_title(str(y0)+'-'+str(ye),fontsize=25)
  #
  extra_artists.extend([ylab,ttl])
  for p in p1.iterkeys():
   du=p1[p][1];
   for jj in du:
    extra_artists.extend([jj]);
  for p in p2.iterkeys():
   du=p2[p][1];
   for jj in du:
    extra_artists.extend([jj]);
  
  for j,ax in enumerate(axes.flatten()):
    txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=20)
    extra_artists.append(txt1)    
  #plt.savefig(plotname+'_decay.png',format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
  #plt.savefig(plotname+'_decay.pdf',format='pdf',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
  #plt.savefig(plotname+'_decay.pdf',dpi=300)
  if saveplot:
    plt.savefig(plotname+'_all_in_one.png',format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
    #plt.savefig(plotname+'_all_in_one.pdf',format='pdf',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists) 
    print(plotname+'_all_in_one.png')
    plt.close('all')


if False:
  extra_artists=[]
  fig,ax=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(25,30/3.))
  ax.set_rasterization_zorder(1)
  cbarloc='right'
  levels=np.array([2,3,4,5,6,7,8,10,15,20,30])#
  cmap=plt.cm.Reds
  cmlist=[];
  for cl in np.linspace(0,252,len(levels)): cmlist.append(int(cl))
  cmap, norm = from_levels_and_colors(levels,cmap(cmlist),extend='max');
  cmap.set_bad([.5,.5,.5])
  m2 = Basemap(projection='cyl',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax)
  m2.drawmapboundary()
  p1=m2.drawparallels(np.arange(-60.,61,20.),labels=[True,False,False,False],linewidth=0.25);
  p2=m2.drawmeridians(np.arange(60.,360.,60.),labels=[False,False,False,True],linewidth=0.25);
  m2.fillcontinents(color='DarkGray',lake_color='LightGray',zorder=0);
  c2=m2.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),optimized_tau,cmap=cmap,norm=norm,latlon=True,rasterized=True)
  cice=m2.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),ma.masked_array(icemask,1-icemask),cmap=plt.cm.Set2_r,latlon=True,rasterized=True)
  cbar2=m2.colorbar(mappable=c2,location=cbarloc)
  ylab=cbar2.ax.set_ylabel('Optimized $\\tau$ [days]',fontsize=25)
  #ttl=ax.set_title('1982-2016', fontsize=25)
  extra_artists.extend([ylab])
  for p in p1.iterkeys():
   du=p1[p][1];
   for jj in du:
    extra_artists.extend([jj]);
  for p in p2.iterkeys():
   du=p2[p][1];
   for jj in du:
    extra_artists.extend([jj]);
  
  plt.savefig('/home/anummel1/move_plots/uvkr_data_python_sst_integral_70S_80N_norot_SpatialHighPass_y4.0deg_x8.0deg_optimized_tau.png',format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)


############################
# Correcting diffusivities #
if False:
 def funce(x, a, c, d, e):
     return a*np.exp(-c*x)+d*x+e
 
 
 #
 data=np.load('/export/scratch/anummel1/inverted_fipy_data.npz')
 for var in data.keys():
     exec(var+'=data["'+var+'"][:]')
 Ds0=dds
 #
 lon,lat=np.meshgrid(Lon_vector,Lat_vector)
 dy=0.25*111E3
 dx=dy*np.cos(lat*np.pi/180)
 testx=[];testy=[]
 for d1,dd in enumerate(Ds0): y=np.array(vels)*5E3/dd; x=(1/(np.array(rs)*3600.*24.))/(dd/(5E3)**2); yy=(Dsx_all[0,:,d1,:,1].ravel()-dd)/dd; yy[np.where(abs(yy)==1)]=0;testx.extend(np.tile(np.log10(y),(len(x),1)).T.ravel()); testy.extend(100*yy)
 x1=[]
 y1=[]
 inds=np.argsort(testx)
 for i in inds: x1.append(testx[i]); y1.append(testy[i])
 x2=x1[np.where(x1>np.log10(0.5))[0][0]:]
 y2=y1[np.where(x1>np.log10(0.5))[0][0]:]
 #popt,pcov=curve_fit(funce,testx,testy,p0=(1,-1,.1,0))
 popt,pcov=curve_fit(funce,x2,y2,p0=(1,-1,.1,0))
 Kxcor=funce(np.log10(abs(U)*dx/Kx),*popt)
 Kxcor[ma.where(abs(U)*dx/Kx<0.5)]=0
 Kycor=funce(np.log10(abs(V)*dy/Ky),*popt)
 Kycor[ma.where(abs(V)*dy/Ky<0.5)]=0
 Kxnew=Kx-0.01*Kxcor*Kx
 Kynew=Ky-0.01*Kycor*Ky
 plt.figure();plt.pcolormesh(ma.mean([Ky,Kx],0)/1E3,cmap=cmap,norm=norm)
 plt.figure();plt.pcolormesh(ma.mean([Kynew,Kxnew],0)/1E3,cmap=cmap,norm=norm)
 #
 levels=np.array([0.0,0.01,0.1,0.25,0.5,0.75,1,1.5,2,3])
 cmap=cmaps[1]
 cmlist=[];
 for cl in np.linspace(0,252,len(levels)+1): cmlist.append(int(cl))
 cmap, norm = from_levels_and_colors(levels,cmap(cmlist),extend='both');
 cmap.set_bad([.5,.5,.5])
 #
 Kvals=[Kxnew/1E3,Kynew/1E3,Kx/1E3,Ky/1E3]
 Ktitles=['$\\kappa_x$ corrected','$\\kappa_y$ corrected','$\\kappa_x$ original','$\\kappa_y$ original']
 #
 fig,axes=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(20,10))
 for j,ax in enumerate(axes.flatten()):
    m2 = Basemap(projection='cyl',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax)
    m2.drawmapboundary()
    p1=m2.drawparallels(np.arange(-60.,61,20.),labels=[True,False,False,False],linewidth=0.25,fontsize=20);
    p2=m2.drawmeridians(np.arange(60.,360.,60.),labels=[False,False,False,True],linewidth=0.25,fontsize=20);
    m2.fillcontinents(color='DarkGray',lake_color='LightGray',zorder=0);
    c2=m2.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),Kvals[j],cmap=cmap,norm=norm,latlon=True,rasterized=True)
    cice=m2.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),ma.masked_array(icemask,1-icemask),cmap=plt.cm.Set2_r,latlon=True,rasterized=True)
    cbar2=m2.colorbar(mappable=c2,location='right')
    ylab=cbar2.ax.set_ylabel('Diffusivity [days]',fontsize=25)
    ax.set_title(Ktitles[j],fontsize=20)
