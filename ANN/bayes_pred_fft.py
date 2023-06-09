import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import pandas as pd
from kernel_2d import Kernel
from BayesMethod import BayesClass
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mlp
from kde_fft_nd import kde_N
df=pd.read_csv('D:\Tai Lieu\OneDrive - Hanoi University of Science and Technology\Desktop\data1.csv',header=0)

df['VpVs ratio']=df['P-wave']/df['S-wave']
df['Zp']=df['P-wave']*df['Density']
# Split the data into training and testing sets
X=df[['Zp','VpVs ratio']]
y=df['Lithology']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
test=BayesClass()
M=100
data=X.values
x_min,x_max=np.min(data[:,0]),np.max(data[:,0])
y_min,y_max=np.min(data[:,1]),np.max(data[:,1])
xx=np.linspace(x_min,x_max,M+1)
yy=np.linspace(y_min,y_max,M+1)
grid=xx,yy
test.fit(X_train.to_numpy(),y_train.to_numpy(),grid,M)
litho_pred_test=test.predict(X_test.to_numpy())
litho_pred=test.predict(X.to_numpy())
# predict result and compare with raw data
plt.subplot(1,2,1) #raw data
plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],cmap='rainbow',s=10)
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
cbar=plt.colorbar()
cbar.set_label('Litho', rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.grid()
plt.subplot(1,2,2) #predict data
plt.scatter(df['Zp'],df['VpVs ratio'],c=litho_pred,cmap='rainbow',s=10)
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
plt.grid()
cbar = plt.colorbar()
cbar.set_label('Litho', rotation=270)
cbar.ax.get_yaxis().labelpad = 15
accuracy = (litho_pred_test == y_test).sum() / len(y_test)
print(accuracy)
plt.show()
