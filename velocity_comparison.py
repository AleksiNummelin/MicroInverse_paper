import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.colors import BoundaryNorm, LogNorm, SymLogNorm, from_levels_and_colors
from mpl_toolkits.basemap import Basemap, addcyclic, interp, maskoceans
import string
#
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
#
ArrowStyle='fancy'
#
def load_files(filepath,fnames):
  #
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
   #
   for key in ['U_global','V_global','Kx_global','Ky_global','R_global']:
      data[key]=np.reshape(data[key][:],(len(Taus),data[key].shape[0]/len(Taus),-1)).squeeze()
   lmask=np.zeros(data['V_global'][0,:,:].shape); lmask[np.where(data['V_global'][0,:,:]==0)]=0
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


#
execfile('plot_autocorrelation.py')
t1=load_autocorrelation_data('/datascope/hainegroup/anummel1/Projects/MicroInv/',['autocorrelation_dt_1_highpass.npz'],[1])
t1=t1.squeeze()
t1mask=t1.copy(); t1mask[np.where(t1mask<=2)]=0; t1mask[np.where(t1mask!=0)]=1; t1mask=1-t1mask
t1mask=ma.masked_array(t1mask,mask=t1.mask)
t1mask=t1mask[80:-40,:]
#
icedata=np.load('/home/anummel1/Projects/MicroInv/icedata.npz')
icemask=np.round(icedata['icetot25'][:,80:-40].T)
#
data_drift=Dataset('/export/scratch/anummel1/drifter_vel/drifter_annualmeans.nc')
data_oscar=Dataset('/export/scratch/anummel1/OSCAR_vel/oscar_vel_ave.nc')
la_d=data_drift.variables['Lat'][:]
lo_d=data_drift.variables['Lon'][:]
la_o=data_oscar.variables['latitude'][::-1]
lo_o=data_oscar.variables['longitude'][:]
lo_d,la_d=np.meshgrid(lo_d,la_d)
lo_o,la_o=np.meshgrid(lo_o,la_o)
#
U_drift=data_drift.variables['U'][:].T
V_drift=data_drift.variables['V'][:].T
U_oscar=data_oscar.variables['u'][:,:,::-1,:].squeeze()
V_oscar=data_oscar.variables['v'][:,:,::-1,:].squeeze()
#
filepath='/datascope/hainegroup/anummel1/Projects/MicroInv/sst_data/final_output/'
Taus=[2,3,4,5,6,7,8,10,15,20,30]
fnames=[]
for j,ext in enumerate(Taus):
  fnames.append('uvkr_data_python_sst_integral_70S_80N_norot_saveB_SpatialHighPass_y4.0deg_x8.0deg_Tau'+str(ext)+'.npz')

data=load_files(filepath,fnames)
#
Lon_vector=data['Lon_vector'][:];
Lat_vector=data['Lat_vector'][:];
Lon_vector=Lon_vector[:np.where(np.diff(Lon_vector)<0)[0][0]+1]
Lat_vector=Lat_vector[:np.where(np.diff(Lat_vector)<0)[0][0]+1]
Lon_vector[ma.where(Lon_vector>180)]=Lon_vector[ma.where(Lon_vector>180)]-360
Lon_vector[ma.where(Lon_vector<-180)]=Lon_vector[ma.where(Lon_vector<-180)]+360
#
lo_i,la_i=np.meshgrid(Lon_vector,Lat_vector)
#
weight_coslat=np.tile(np.cos(Lat_vector*np.pi/180.),(Lon_vector.shape[-1],1)).T
A2=combine_Taus(data,weight_coslat,Taus)
#
for var in ['Kx','Ky','R','U','V']:
  exec(var+'=A2["'+var+'_global"][:]')
#
lmask=np.zeros(V.shape)
lmask[ma.where(V==0)]=1
lmask[ma.where(icemask)]=1
#
U_inversion=ma.masked_array(U,lmask)
V_inversion=ma.masked_array(V,lmask)
#LOAD EDDY CORE DATA
#eddy_core=np.load('/home/anummel1/Projects/MicroInv/eddy_core_velocity.npz')
eddy_core=np.load('/home/anummel1/Projects/MicroInv/eddy_core_velocity_binned.npz')
grid_x=eddy_core['grid_x'][:]
grid_y=eddy_core['grid_y'][:]
u_grid=eddy_core['u_grid'][:]
v_grid=eddy_core['v_grid'][:]
eddy_mask=eddy_core['mask'][:]
eddy_core_count=eddy_core['core_count'][:]
eddy_core_count=ma.masked_array(eddy_core_count,mask=eddy_mask)
mask=maskoceans(grid_x, grid_y,np.zeros(grid_x.shape),inlands=False,resolution='c')
eddy_mask=eddy_mask+(1-mask.mask)
eddy_mask[np.where(eddy_mask>1)]=1
#eddy_icemask=eddy_core['icemask2'][:]
#
if False:
 fig,ax2=plt.figure()
 ax2.hist(V_inversion[ma.where(1-V_inversion.mask)],bins=nbins,range=x_range,normed=True,weights=np.cos(np.pi*la_i[ma.where(1-V_inversion.mask)]/180.),alpha=aa,label='M ethod 1 smooth');
 ax2.hist(V_drift[ma.where(np.isfinite(V_drift))],bins=nbins,range=x_range,normed=True,weights=np.cos(np.pi*la_d[ma.where(np.isfinite(V_drift))]/180.),alpha=aa,label='Drifter');
 ax2.hist(V_oscar[ma.where(1-V_oscar.mask)],bins=nbins,range=x_range,normed=True,weights=np.cos(np.pi*la_o[ma.where(1-V_oscar.mask)]/180.),alpha=aa,label='Oscar');
 #
 lgnd=ax1.legend(fontsize=20)
 xlab1=ax1.set_xlabel('Zonal velocity [m s$^{-1}$]',fontsize=20)
 xlab2=ax2.set_xlabel('Meridional velocity [m s$^{-1}$]',fontsize=20)
 ylab1=ax1.set_ylabel('Probability density distribution [%]',fontsize=20)
 extra_artists=[lgnd,xlab1,xlab2,ylab1]
 #
 #plt.savefig('/home/anummel1/move_plots/UV_comparison.png',format='png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
 #
 #
if True:
 titles=['Inversion','Eddy','Oscar','Drifter']
 fig,axes=plt.subplots(nrows=4,ncols=1,sharex=True,sharey=True,figsize=(25,40))
 fig.subplots_adjust(hspace=0.15)
 extra_artists=[]
 for j,ax in enumerate(axes.flatten()):
  print titles[j]
  ax.set_rasterization_zorder(1)
  cbarloc='right'
  levels=np.array([0,0.01,0.02,0.03,0.04,0.05,0.1,0.25,0.5])*100
  cmap=plt.cm.Reds
  cmlist=[];
  for cl in np.linspace(0,252,len(levels)): cmlist.append(int(cl))
  cmap, norm = from_levels_and_colors(levels,cmap(cmlist),extend='max');
  cmap.set_bad([.5,.5,.5])
  #
  if j==0:
    #speed=ma.sqrt(U_inversion**2+V_inversion**2)
    u=U_inversion.copy();  u[:,:720]=U_inversion[:,720:]; u[:,720:]=U_inversion[:,:720];
    v=V_inversion.copy();  v[:,:720]=V_inversion[:,720:]; v[:,720:]=V_inversion[:,:720];
    lon=Lon_vector.copy(); lon[:720]=Lon_vector[720:]; lon[720:]=Lon_vector[:720]
    lat=Lat_vector.copy(); 
    lon,lat=np.meshgrid(lon,lat)
  elif j==1:
    #speed=ma.sqrt(u_grid**2+v_grid**2)
    u=ma.masked_array(u_grid.copy(),mask=eddy_mask)
    v=ma.masked_array(v_grid.copy(),mask=eddy_mask)
    lon=grid_x.copy()
    lat=grid_y.copy()
  elif j==2:
    #speed=ma.sqrt(U_oscar**2+V_oscar**2)[:,:360]
    #fix the range to be -180:180 for plotting convenience
    u=U_oscar[:,:1080].copy(); u[:,:601]=U_oscar[:,480:1081]; u[:,600:]=U_oscar[:,:480];
    v=V_oscar[:,:1080].copy(); v[:,:601]=V_oscar[:,480:1081]; v[:,600:]=V_oscar[:,:480];
    lon=lo_o[:,:1080].copy();  lon[:,:601]=lo_o[:,480:1081]; lon[:,600:]=lo_o[:,:480];
    lat=la_o[:,:1080].copy();  lat[:,:601]=la_o[:,480:1081]; lat[:,600:]=la_o[:,:480];
    lon[np.where(lon>=180)]=lon[np.where(lon>=180)]-360
  elif j==3:
    #speed=ma.sqrt(U_drift**2+V_drift**2)
    u=U_drift.copy()
    v=V_drift.copy()
    lon=lo_d.copy()
    lat=la_d.copy()
  #
  speed=ma.sqrt(u**2+v**2)
  #
  m2 = Basemap(projection='cyl',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax)
  m2.drawmapboundary()
  p1=m2.drawparallels(np.arange(-60.,61,20.),labels=[True,False,False,False],linewidth=0.25,fontsize=20);
  p2=m2.drawmeridians(np.arange(60.,360.,60.),labels=[False,False,False,True],linewidth=0.25,fontsize=20);
  m2.fillcontinents(color='DarkGray',lake_color='LightGray',zorder=0);
  c2=m2.pcolormesh(lon,lat,100*speed,cmap=cmap,norm=norm,latlon=True,rasterized=True)
  x,y=m2(lon,lat)
  ax.streamplot(x,y,u,v,color='k',linewidth=1.5,density=3,arrowstyle=ArrowStyle)#,minlength=1.0) #,rasterized=True)
  cice=m2.pcolormesh(Lon_vector.squeeze(),Lat_vector.squeeze(),ma.masked_array(icemask,1-icemask),cmap=plt.cm.Set2_r,latlon=True,rasterized=True)
  if j==0:
    lon2,lat2=np.meshgrid(Lon_vector,Lat_vector)
    x2,y2=m2(lon2[ma.where(t1mask)],lat2[ma.where(t1mask)])
    m2.plot(x2, y2, '.', color='gray', markersize=0.5)
  elif j==1:
    x2,y2=m2(lon[ma.where(eddy_core_count<10)],lat[ma.where(eddy_core_count<10)])
    m2.plot(x2, y2, '.', color='gray', markersize=0.5)
  cbar2=m2.colorbar(mappable=c2,location=cbarloc)
  ylab=cbar2.ax.set_ylabel('Speed [cm s$^{-1}$]',fontsize=25)
  ttl=ax.set_title(titles[j], fontsize=25)
  txt1=ax.text(0.0, 1.02, string.ascii_lowercase[j],transform=ax.transAxes, fontsize=25)
  extra_artists.extend([ylab,ttl,txt1])
  for p in p1.iterkeys():
   du=p1[p][1];
   for jj in du:
    extra_artists.extend([jj]);
  for p in p2.iterkeys():
   du=p2[p][1];
   for jj in du:
    extra_artists.extend([jj]);
 #
 plt.savefig('/home/anummel1/move_plots/UV_comparison_map_v2.png',format='png',dpi=150,bbox_inches='tight',bbox_extra_artists=extra_artists)



if False:
 U_d=U_drift[12:-20,:]
 V_d=V_drift[12:-20,:]
 la_d2=la_d[12:-20,:]
 lo_d2=lo_d[12:-20,:]
 #
 i0=1160
 i1=1200
 #
 id0=440
 id1=480
 j=400
 dj=50
 di=40
 #
 fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(6,6))
 p1=np.percentile(Kx[j:j+dj,i0:i0+di],[25,50,75],axis=1)
 p2=np.percentile(Ky[j:j+dj,i0:i0+di],[25,50,75],axis=1)
 p3=np.percentile(ma.sqrt(U_d**2+V_d**2)[j:j+dj,id0:id0+di],[25,50,75],axis=1)
 l1=ax.plot(Lat_vector[j:j+dj],p1[1,:],color='C1',lw=2,label='$\kappa_x$')
 l2=ax.plot(Lat_vector[j:j+dj],p2[1,:],color='C2',lw=2,label='$\kappa_y$')
 ax.fill_between(Lat_vector[j:j+dj],p1[0,:],p1[2,:],color='C1',alpha=0.3)
 ax.fill_between(Lat_vector[j:j+dj],p2[0,:],p2[2,:],color='C2',alpha=0.3)
 ax2=ax.twinx()
 l3=ax2.plot(la_d2[j:j+dj,0],p3[1,:],color='C3',lw=2,label='Drifter |U|')
 ax2.fill_between(la_d2[j:j+dj,0],p3[0,:],p3[2,:],color='C3',alpha=0.3)
 ax.set_xlabel('Latitude [$\degree$ N]',fontsize=18)
 ax.set_ylabel('Diffusivity [m$^2$ s$^{-1}$]',fontsize=18)
 ax2.set_ylabel('Speed [m s$^{-1}$]',fontsize=18)
 ax.legend(loc=2,fontsize=15)
 ax2.legend(loc=1,fontsize=15)
 ax.set_ylim(0,1800)
 ax2.set_ylim(0,1)
 ax.set_xlim(32,42)
 plt.savefig('/home/anummel1/move_plots/Diffusivity_AccrossNAC_70W_60W.png',format='png',dpi=150,bbox_inches='tight',bbox_extra_artists=[xlab,ylab1,ylab2])
 #
 #ax2.plot(Lat_vector[j:j+dj],ma.median(ma.sqrt(U**2+V**2)[j:j+dj,i0:i0+di],-1),'--k')
 fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(6,6))
 p1=np.percentile(Kx[j:j+dj,i1:i1+di],[25,50,75],axis=1)
 p2=np.percentile(Ky[j:j+dj,i1:i1+di],[25,50,75],axis=1)
 p3=np.percentile(ma.sqrt(U_d**2+V_d**2)[j:j+dj,id1:id1+di],[25,50,75],axis=1)
 ax.plot(Lat_vector[j:j+dj],p1[1,:],color='C1',lw=2,label='$\kappa_x$')
 ax.plot(Lat_vector[j:j+dj],p2[1,:],color='C2',lw=2,label='$\kappa_y$')
 ax.fill_between(Lat_vector[j:j+dj],p1[0,:],p1[2,:],color='C1',alpha=0.3)
 ax.fill_between(Lat_vector[j:j+dj],p2[0,:],p2[2,:],color='C2',alpha=0.3)
 ax2=ax.twinx() 
 ax2.plot(la_d2[j:j+dj,0],p3[1,:],color='C3',lw=2,label='Drifter |U|')
 ax2.fill_between(la_d2[j:j+dj,0],p3[0,:],p3[2,:],color='C3',alpha=0.3)
 xlab=ax.set_xlabel('Latitude [$\degree$ N]',fontsize=18)
 ylab1=ax.set_ylabel('Diffusivity [m$^2$ s$^{-1}$]',fontsize=18)
 ylab2=ax2.set_ylabel('Speed [m s$^{-1}$]',fontsize=18)
 ax.legend(loc=2,fontsize=15)
 ax2.legend(loc=1,fontsize=15)
 ax.set_ylim(0,1800)
 ax2.set_ylim(0,1)
 ax.set_xlim(32,42)
 plt.savefig('/home/anummel1/move_plots/Diffusivity_AccrossNAC_60W_50W.png',format='png',dpi=150,bbox_inches='tight',bbox_extra_artists=[xlab,ylab1,ylab2])

if True:
    from scipy.interpolate import griddata
    method    = 'linear'
    #
    u=U_inversion.copy();  u[:,:720]=U_inversion[:,720:]; u[:,720:]=U_inversion[:,:720];
    v=V_inversion.copy();  v[:,:720]=V_inversion[:,720:]; v[:,720:]=V_inversion[:,:720];
    lon=Lon_vector.copy(); lon[:720]=Lon_vector[720:]; lon[720:]=Lon_vector[:720]
    lat=Lat_vector.copy();
    lon,lat=np.meshgrid(lon,lat)
    mask2=t1mask.data.copy(); mask2[:,:720]=t1mask[:,720:]; mask2[:,720:]=t1mask[:,:720];
    mask2[np.where(mask1)]=1
    #
    u1=ma.masked_array(u_grid.copy(),mask=eddy_mask)
    v1=ma.masked_array(v_grid.copy(),mask=eddy_mask)
    #
    print('grid eddy')
    u1[np.where(np.isnan(u1))]=0
    v1[np.where(np.isnan(v1))]=0
    u1 = griddata((grid_x.flatten(),grid_y.flatten()), u1.data.flatten(), (lon, lat), method=method)
    v1 = griddata((grid_x.flatten(),grid_y.flatten()), v1.data.flatten(), (lon, lat), method=method)
    #
    print('grid oscar')
    u2=U_oscar[:,:1080].copy(); u2[:,:601]=U_oscar[:,480:1081]; u2[:,600:]=U_oscar[:,:480];
    v2=V_oscar[:,:1080].copy(); v2[:,:601]=V_oscar[:,480:1081]; v2[:,600:]=V_oscar[:,:480];
    lon2=lo_o[:,:1080].copy();  lon2[:,:601]=lo_o[:,480:1081]; lon2[:,600:]=lo_o[:,:480];
    lat2=la_o[:,:1080].copy();  lat2[:,:601]=la_o[:,480:1081]; lat2[:,600:]=la_o[:,:480];
    lon2[np.where(lon2>180)]=lon2[np.where(lon2>180)]-360
    u2[np.where(np.isnan(u2))]=0
    v2[np.where(np.isnan(v2))]=0
    u2 = griddata((lon2.flatten(),lat2.flatten()), u2.data.flatten(), (lon, lat), method=method)
    v2 = griddata((lon2.flatten(),lat2.flatten()), v2.data.flatten(), (lon, lat), method=method)
    #
    print('grid drift')
    u3 = U_drift.copy()
    v3 = V_drift.copy()
    u3[np.where(np.isnan(u3))]=0
    v3[np.where(np.isnan(v3))]=0
    u3 = griddata((lo_d.flatten(),la_d.flatten()), u3.data.flatten(), (lon, lat), method=method)
    v3 = griddata((lo_d.flatten(),la_d.flatten()), v3.data.flatten(), (lon, lat), method=method)
    #lon=lo_d.copy()
    #lat=la_d.copy()
    #
    mask1=u.mask.copy(); mask1[260:300,:]=1
    #
    fig,axes=plt.subplots(nrows=3,ncols=2)
    axes.flatten()[0].pcolormesh(ma.masked_array(u-u1,mask=mask1),vmin=-0.1,vmax=0.1)
    axes.flatten()[1].pcolormesh(ma.masked_array(v-v1,mask=mask1),vmin=-0.1,vmax=0.1)
    #
    axes.flatten()[2].pcolormesh(u-u2,vmin=-0.1,vmax=0.1)
    axes.flatten()[3].pcolormesh(v-v2,vmin=-0.1,vmax=0.1)
    #
    axes.flatten()[4].pcolormesh(u-u3,vmin=-0.1,vmax=0.1)
    axes.flatten()[5].pcolormesh(v-v3,vmin=-0.1,vmax=0.1)
    #
    #
    mask1=u.mask.copy(); mask1[200:360,:]=1
    #
    rmse=np.zeros(6)
    velmeans=np.zeros(8)
    jj,ii=ma.where(1-mask2)
    mask3=mask2.copy(); mask3[260:300,:]=1
    #mask3[200:360,:]=1 #outside tropics
    mask3[:200,:]=1;mask3[360:,:]=1 #tropics only
    jj,ii=ma.where(1-mask3)
    rmse[0]=ma.sqrt(ma.sum(np.cos(lat*np.pi/180.)[jj,ii]*(u[jj,ii]-u1[jj,ii])**2)/ma.sum(np.cos(lat*np.pi/180.)[jj,ii]))
    rmse[1]=ma.sqrt(ma.sum(np.cos(lat*np.pi/180.)[jj,ii]*(v[jj,ii]-v1[jj,ii])**2)/ma.sum(np.cos(lat*np.pi/180.)[jj,ii]))
    #
    rmse[2]=ma.sqrt(np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(u[jj,ii]-u2[jj,ii])**2)/ma.sum(np.cos(lat*np.pi/180.)[jj,ii]))
    rmse[3]=ma.sqrt(np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(v[jj,ii]-v2[jj,ii])**2)/ma.sum(np.cos(lat*np.pi/180.)[jj,ii]))
    #
    rmse[4]=ma.sqrt(np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(u[jj,ii]-u3[jj,ii])**2)/ma.sum(np.cos(lat*np.pi/180.)[jj,ii]))
    rmse[5]=ma.sqrt(np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(v[jj,ii]-v3[jj,ii])**2)/ma.sum(np.cos(lat*np.pi/180.)[jj,ii]))
    #
    velmeans[0]=np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(u)[jj,ii])/ma.sum(np.cos(lat*np.pi/180.)[jj,ii])
    velmeans[1]=np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(u1)[jj,ii])/ma.sum(np.cos(lat*np.pi/180.)[jj,ii])
    velmeans[2]=np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(u2)[jj,ii])/ma.sum(np.cos(lat*np.pi/180.)[jj,ii])
    velmeans[3]=np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(u3)[jj,ii])/ma.sum(np.cos(lat*np.pi/180.)[jj,ii])
    #
    velmeans[4]=np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(v)[jj,ii])/ma.sum(np.cos(lat*np.pi/180.)[jj,ii])
    velmeans[5]=np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(v1)[jj,ii])/ma.sum(np.cos(lat*np.pi/180.)[jj,ii])
    velmeans[6]=np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(v2)[jj,ii])/ma.sum(np.cos(lat*np.pi/180.)[jj,ii])
    velmeans[7]=np.nansum(np.cos(lat*np.pi/180.)[jj,ii]*(v3)[jj,ii])/ma.sum(np.cos(lat*np.pi/180.)[jj,ii])
    mask1=np.ones(u.shape)
    mask1[200:260,:]=0
    mask1[300:360,:]=0
    mask1[np.where(u.mask)]=1
