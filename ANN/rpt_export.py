import numpy as np
import math
import Kdry_func
import Mu_dry_func
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from scipy.signal import convolve,lfilter,convolve2d
import sklearn as sk
# import convolution_common as kde_cm
import kernel_2d as kde_2d
from sklearn.neural_network import  MLPRegressor
from sklearn.model_selection import train_test_split
import pickle
Por=np.arange(0.05,0.35,0.01)
Sw=np.arange(0,1.01,0.01)
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
# cbar.set_label('Water Saturation', rotation=270)
# plt.plot(den_sat[-1,:]*Vp_new[-1,:],Vp_new[-1,:]/Vs_new[-1,:],'b-o',linewidth = '0.5',markersize=3)
# for i in range(len(Por)):
#     plt.plot(den_sat[:,i]*Vp_new[:,i],Vp_new[:,i]/Vs_new[:,i],'b-o',linewidth = '0.5',markersize=3)
# plt.ylim([1.6,2.2])
# plt.xlim([8000,12500])

# define x, y coordinates

# rpt data
Zp_rpt=(den_sat*Vp_new)
VpVs_rpt=(Vp_new/Vs_new)


# gradient rpt
u=np.ones(Zp_rpt.shape)
v=np.ones(Zp_rpt.shape)
for i in range(len(Por)):
    for j in range(len(Sw)):
        try:
            u[j, i] = Zp_rpt[j, i + 1] - Zp_rpt[j, i]
            v[j, i] = VpVs_rpt[j, i + 1] - VpVs_rpt[j, i]
        except IndexError:
            u[j, i]=u[j, i-1]
            v[j, i]=v[j, i-1]

Zp_rpt=Zp_rpt.ravel()
VpVs_rpt=VpVs_rpt.ravel()
data=np.column_stack((Zp_rpt,VpVs_rpt))
y=np.arctan(v.ravel()/u.ravel())
# Save the variable to a file using pickle
my_var={'zp':Zp_rpt,'VpVs':VpVs_rpt,'actan':y,'Porosity':Por_V,'Sw':fl_V}
with open('RPT_variables.pickle', 'wb') as f:
    pickle.dump(my_var, f)

# plt.scatter(Zp_rpt,VpVs_rpt,c=y)
# plt.show()

# plt.scatter(Zp_rpt,VpVs_rpt,c=y)
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(data,y,test_size=0.1,random_state=42)
mlp_regressor=MLPRegressor(activation='tanh',solver='adam',hidden_layer_sizes=(1000,100,50,20,10,5),random_state=1, max_iter=500)
mlp_regressor.fit(data,y)
y_pred=mlp_regressor.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print(mlp_regressor.coefs_)
print(mlp_regressor.intercepts_)
print("Accuracy on training set: {:.2f}".format(mlp_regressor.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp_regressor.score(X_test, y_test)))
x = np.linspace(np.min(Zp_rpt), np.max(df['Zp']), 100)
y = np.linspace(np.min(df['VpVs ratio']), np.max(df['VpVs ratio']), 100)
y = np.linspace(np.min(df['VpVs ratio']), np.max(df['VpVs ratio']), 100)
XI, YI = np.meshgrid(x, y)
points = np.transpose(np.vstack((Zp_rpt, VpVs_rpt)))
gradx = griddata(points, u.ravel(), (XI, YI), method='linear')
grady = griddata(points, v.ravel(), (XI, YI), method='linear')

# ZI=griddata((Zp_rpt[0],VpVs_rpt[0]),Por_V.reshape(1,-1)[0],(XI,YI),method='linear')
# u_interp=griddata()
# ZI[np.isnan(ZI)]=np.nanmean(ZI)
# ax.plot_surface(XI,YI,ZI, cmap='viridis')
# gradx= np.gradient(ZI, axis=1) #Zp
# grady= np.gradient(ZI, axis=0) #Vp/Vs
atanGrad=np.arctan(grady/gradx)

atanGrad[np.isnan(atanGrad)]=0
windowSize = 30 # example window size
filter=np.ones((windowSize,windowSize))/windowSize**2
M=atanGrad.shape[0]
atanGrad = convolve2d(atanGrad,filter)
atanGrad = atanGrad[windowSize-1:M+windowSize,windowSize-1:M+windowSize]
mlp_regressor.fit(np.vstack((XI.ravel(),YI.ravel())).T,atanGrad.ravel())
# atanGrad[np.isnan(atanGrad)]=np.nanmean(0)

grid=np.vstack((XI.ravel(),YI.ravel()))
atanGrad=mlp_regressor.predict(grid.T).reshape(XI.shape)
plt.figure()
ax=plt.axes(projection='3d')
ax.plot_surface(XI,YI,atanGrad,cmap='rainbow')
plt.show()
hx = np.std(df['Zp']) / 2 / len(df['Zp']) ** (1 / 6)
hy = np.std(df['VpVs ratio']) / 2 / len(df['VpVs ratio']) ** (1 / 6)
data=df[['Zp','VpVs ratio']].to_numpy()
# tinh kernel
a=kde_2d.Kernel()
a.fit(data[df['Lithology']!=0])
result=np.array(a.kernel_2d(grid.T,1/(15*hx)**2,1/(hy)**2,-atanGrad.ravel()))
#-------------------------
# plt.figure()
# plt.scatter(XI[atanGrad[:,:]!=0],YI[atanGrad[:,:]!=0],c=atanGrad[atanGrad[:,:]!=0],marker='s',vmin=np.min(atanGrad),vmax=np.max(atanGrad))
# cbar=plt.colorbar()
# plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],s=15,marker='o', edgecolors='black', linewidths=0.5)
# plt.quiver(XI, YI, gradx, grady)
# # vector rpt
# for i in range(len(Por)):
#     for j in range(len(Sw)):
#         if j%2==0 and i%2==0:
#             plt.arrow(Zp_rpt.reshape(fl_V.shape)[j, i], VpVs_rpt.reshape(fl_V.shape)[j, i],
#                       Zp_rpt.reshape(fl_V.shape)[j, i + 1] - Zp_rpt.reshape(fl_V.shape)[j, i],
#                       VpVs_rpt.reshape(fl_V.shape)[j, i + 1] - VpVs_rpt.reshape(fl_V.shape)[j, i], facecolor='red',
#                       edgecolor='none')

# Zp_rpt.reshape(fl_V.shape)[]

#-----------------------
# plt.xlabel('Zp')
# plt.ylabel('Vp Vs ratio')
# cbar.set_label('arctan(Fy/Fx)', rotation=270)
# plt.figure()
# plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],s=15,marker='o', edgecolors='black', linewidths=0.5)
# plt.contour(XI,YI,result.reshape(XI.shape))
# plt.plot(den_sat[-1,:]*Vp_new[-1,:],Vp_new[-1,:]/Vs_new[-1,:],'b-o',linewidth = '0.5',markersize=3)
# for i in range(len(Por)):
#     plt.plot(den_sat[:,i]*Vp_new[:,i],Vp_new[:,i]/Vs_new[:,i],'b-o',linewidth = '0.5',markersize=3)
# plt.xlabel('Zp')
# plt.ylabel('Vp Vs ratio')
# cbar.set_label('arctan(Fy/Fx)', rotation=270)
# plt.show()
