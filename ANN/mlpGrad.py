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

Zp_rpt=Zp_rpt.reshape(1,-1)
VpVs_rpt=VpVs_rpt.reshape(1,-1)
data=np.vstack((Zp_rpt,VpVs_rpt)).T
y=np.arctan(v.reshape(1,-1)/u.reshape(1,-1)).T

X_train, X_test, y_train, y_test = train_test_split(data,y,test_size=0.1,random_state=42)
mlp_regressor=MLPRegressor(activation='tanh',solver='adam',hidden_layer_sizes=(1,),random_state=1, max_iter=500)
mlp_regressor.fit(X_train,y_train)
y_pred=mlp_regressor.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print(mlp_regressor.coefs_)
print(mlp_regressor.intercepts_)
print("Accuracy on training set: {:.2f}".format(mlp_regressor.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp_regressor.score(X_test, y_test)))