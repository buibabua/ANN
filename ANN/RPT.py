import numpy as np
import math
import Kdry_func
import Mu_dry_func
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
# import convolution_common as kde_cm
import kernel_2d as kde_2d
Por=np.arange(0.05,0.35,0.01)
Sw=np.arange(0,1.2,0.01)
den_m=2.7639
Km=43
Mum=20.3
Kcri=6.4684
Mucri=5.6135
Phic=0.4
RhoBr=0.97
RhoOil=0.6
RhoGas=0.12
Kbr=2.498
Kgas=0.0432
Kfl_new=np.ones((len(Sw),len(Por)))
Kdry=np.ones((len(Sw),len(Por)))
Musat_rpt=np.ones((len(Sw),len(Por)))
den_fl=np.ones((len(Sw),len(Por)))
den_sat=np.ones((len(Sw),len(Por)))
Ksat_rpt=np.ones((len(Sw),len(Por)))
Vp_new=np.ones((len(Sw),len(Por)))
Vs_new=np.ones((len(Sw),len(Por)))
fl_V=np.ones((len(Sw),len(Por)))
Por_V=np.ones((len(Sw),len(Por)))
for i in range(len(Por)):
    for j in range(len(Sw)):
        Kfl_new[j,i]=(Kbr-Kgas)*Sw[j]**1+Kgas
        Kdry[j,i]=Kdry_func.Kdry_func(Km,Mum,Kcri,Mucri,Phic,Por[i],method='Hash')
        Musat_rpt[j,i]=Mu_dry_func.Mu_dry_func(Km,Mum,Kcri,Mucri,Phic,Por[i],method='Hash')
        Vbr=Sw[j]
        VOil=0
        VGas=(1-Sw[j])
        den_fl=RhoBr*Vbr+RhoOil*VOil+RhoGas*VGas
        den_sat[j,i]=den_fl*Por[i]+den_m*(1-Por[i])
        Ksat_rpt[j,i]=Kdry[j,i]+(1-Kdry[j,i]/Km)**2/(Por[i]/Kfl_new[j,i]+(1-Por[i])/Km-Kdry[j,i]/Km**2)
        Vp_new[j,i]=math.sqrt((Ksat_rpt[j,i]+4/3*Musat_rpt[j,i])*10**6/den_sat[j,i])
        Vs_new[j,i]=math.sqrt(Musat_rpt[j,i]/den_sat[j,i]*10**6)
        fl_V[j, i] = Sw[j]
        Por_V[j, i] = Por[i]

# plot Vp vs zp
df=pd.read_csv('D:\Tai Lieu\OneDrive - Hanoi University of Science and Technology\Desktop\data1.csv',header=0)
df['VpVs ratio']=df['P-wave']/df['S-wave']
df['Zp']=df['P-wave']*df['Density']
colors=df['Water Saturation']
plt.grid(linestyle='--',linewidth = 0.5)
plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],s=15,marker='o', edgecolors='black', linewidths=0.5)
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
cbar=plt.colorbar()
cbar.set_label('Water Saturation', rotation=270)
plt.plot(den_sat[-1,:]*Vp_new[-1,:],Vp_new[-1,:]/Vs_new[-1,:],'b-o',linewidth = '0.5',markersize=3)
for i in range(len(Por)):
    plt.plot(den_sat[:,i]*Vp_new[:,i],Vp_new[:,i]/Vs_new[:,i],'b-o',linewidth = '0.5',markersize=3)
plt.ylim([1.6,2.2])
plt.xlim([8000,12500])

# define x, y coordinates

# rpt data
Zp_rpt=(den_sat*Vp_new).reshape(1,-1)
VpVs_rpt=(Vp_new/Vs_new).reshape(1,-1)
x = np.linspace(np.min(Zp_rpt), np.max(Zp_rpt), 100)
# x = np.linspace(np.min(df['Zp']), np.max(df['Zp']), 100)
# y = np.linspace(np.min(df['VpVs ratio']), np.max(df['VpVs ratio']), 100)
y = np.linspace(np.min(VpVs_rpt), np.max(VpVs_rpt), 100)
XI, YI = np.meshgrid(x, y)
ZI=griddata((Zp_rpt[0],VpVs_rpt[0]),Por_V.reshape(1,-1)[0],(XI,YI),method='linear')
# ZI[np.isnan(ZI)]=np.nanmean(ZI)
# ax.plot_surface(XI,YI,ZI, cmap='viridis')
grady,gradx= np.gradient(ZI)
atanGrad=np.arctan(grady/gradx)
# atanGrad[np.isnan(atanGrad)==1]=np.nanmean(atanGrad)
a=kde_2d.Kernel()
grid=np.vstack([XI.ravel(),YI.ravel()])
hx = np.std(df['Zp']) / 2 / len(df['Zp']) ** (1 / 6)
hy = np.std(df['VpVs ratio']) / 2 / len(df['VpVs ratio']) ** (1 / 6)
a.fit(df[['Zp','VpVs ratio']].to_numpy())
result=np.array(a.kernel_2d(grid.T,1/(hx)**2,1/(hy)**2,atanGrad.ravel()/(hx/hy)))

# gradx[np.isnan(gradx)]=0
#
# grady[np.isnan(grady)]=0

# raw data
# x = np.linspace(np.min(df['Zp']), np.max(df['Zp']), 1000)
# y = np.linspace(np.min(df['VpVs ratio']), np.max(df['VpVs ratio']), 1000)
# XI, YI = np.meshgrid(x, y)
# ZI=griddata((df['Zp'],df['VpVs ratio']),df['Porosity'],(XI,YI),method='linear')
# gradx= np.gradient(ZI, axis=0)
# grady= np.gradient(ZI, axis=1)
# plot the vector field
plt.figure()

ax=plt.axes(projection='3d')
plt.quiver(XI, YI,0, gradx, grady,0)
atanGrad=np.arctan(grady/gradx)
atanGrad[np.isnan(atanGrad)==1]=np.nanmean(atanGrad)
# surf=ax.plot_surface(XI,YI,atanGrad,cmap='viridis')
# surf=ax.plot_surface(XI,YI,ZI,cmap='rainbow')
# plt.colorbar(surf)
ax.set_xlabel('Zp')
ax.set_ylabel('Vp Vs ratio')
ax.set_zlabel('Phie')

# plt.scatter(df['Zp'],df['VpVs ratio'],c=colors,cmap='rainbow',s=10)
# plt.contour(XI,YI,result.reshape(XI.shape))
# plt.plot(den_sat[-1,:]*Vp_new[-1,:],Vp_new[-1,:]/Vs_new[-1,:],'b-o',linewidth = '0.5',markersize=3)
# for i in range(len(Por)):
#     plt.plot(den_sat[:,i]*Vp_new[:,i],Vp_new[:,i]/Vs_new[:,i],'b-o',linewidth = '0.5',markersize=3)
#

# ax.plot_surface(XI,YI,atanGrad,cmap='viridis', vmin=-0.1, vmax=0.14)
# plt.quiver(XI, YI, gradx, grady)
# plt.scatter(XI.reshape(1,-1)[0],YI.reshape(1,-1)[0],np.arctan(grady/gradx).reshape(1,-1)[0])
ax.view_init(elev=90, azim=-90)

# colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']
ax.dist=7
# cmap = plt.cm.colors.ListedColormap(colors)
# bounds=[0,1,2,3,4,5,6,7,8,9,10]
# norm = mcolors.BoundaryNorm(bounds, cmap.N)
#
# plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],cmap=cmap,norm=norm,s=15,marker='o', edgecolors='black', linewidths=0.5)
# plt.xlabel('Zp')
# plt.ylabel('Vp Vs ratio')
# cbar=plt.colorbar()
# cbar.set_label('Lithology',rotation=270)

# plt.ylim([1.6,2.2])
# plt.xlim([8000,12500])
# ax.dist=7
plt.figure()
plt.scatter(XI,YI,c=atanGrad,marker='s')
cbar=plt.colorbar()
plt.quiver(XI, YI, gradx, grady)
# vector rpt
for i in range(len(Por)-1):
    for j in range(len(Sw)):
        if j%2==0 and i%2==0:
            plt.arrow(Zp_rpt.reshape(fl_V.shape)[j, i], VpVs_rpt.reshape(fl_V.shape)[j, i],
                      Zp_rpt.reshape(fl_V.shape)[j, i + 1] - Zp_rpt.reshape(fl_V.shape)[j, i],
                      VpVs_rpt.reshape(fl_V.shape)[j, i + 1] - VpVs_rpt.reshape(fl_V.shape)[j, i],
                      facecolor='red',
                      edgecolor='none')

# Zp_rpt.reshape(fl_V.shape)[]

#-----------------------
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
cbar.set_label('arctan(Fy/Fx)', rotation=270)
plt.figure()
plt.scatter(df['Zp'],df['VpVs ratio'],c=colors,cmap='rainbow',s=10)
plt.contour(XI,YI,result.reshape(XI.shape))
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')

cbar.set_label('arctan(Fx/Fy)', rotation=270)
plt.show()
