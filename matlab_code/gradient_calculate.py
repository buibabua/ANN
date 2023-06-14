# %load gradient_calculate.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
import pandas as pd
import kernel_2d as kde_2d
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors
# define colormap for litho
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']
cmap = plt.cm.colors.ListedColormap(colors)
bounds=[0,1,2,3,4,5,6,7,8,9,10]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
#-------------------------------------------
df=pd.read_csv('data1.csv',header=0)
df['VpVs ratio']=df['P-wave']/df['S-wave']
df['Zp']=df['P-wave']*df['Density']
# colors=df['Water Saturation']
# plt.grid(linestyle='--',linewidth = 0.5)
# plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],s=15,marker='o', edgecolors='black', linewidths=0.5)
# load zp vp/vs of rpt
with open('RPT_variables.pickle', 'rb') as f:
    loaded_vars = pickle.load(f)
Zp=loaded_vars['zp']
VpVs=loaded_vars['VpVs']
arctan=loaded_vars['actan']
Por=loaded_vars['Porosity']
Sw=loaded_vars['Sw']

# Generate random data
y = arctan
Zp=Zp/10000
X= np.column_stack((Zp,VpVs))
plt.scatter(X[:,0]*10000,X[:,1],c=y)
plt.colorbar()
X_train, X_test, y_train, y_test = train_test_split(X,y*100000,test_size=0.1,random_state=42)
# Create an MLPRegressor object and fit it to the data
mlp = MLPRegressor(hidden_layer_sizes=(50,2),solver='lbfgs', activation='tanh',random_state=1, learning_rate_init=0.001, max_iter=1000)
mlp.fit(X_train, y_train)
coeff=mlp.coefs_
inter=mlp.intercepts_
y_pred=mlp.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print(mlp.coefs_)
print(mlp.intercepts_)
print("Accuracy on training set: {:.5f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.5f}".format(mlp.score(X_test, y_test)))
# Create a meshgrid of feature values
# x1_min, x1_max = X[:, 0].min() , X[:, 0].max()
# x2_min, x2_max = X[:, 1].min() , X[:, 1].max()
x1_min, x1_max = np.min(Zp) , np.max(df['Zp']/10000)
x2_min, x2_max = np.min(df['VpVs ratio']) , np.max(df['VpVs ratio'])
# x = np.linspace(np.min(Zp), np.max(df['Zp']), 100)
# y = np.linspace(np.min(df['VpVs ratio']), np.max(df['VpVs ratio']), 100)
# xx1,xx2= np.meshgrid(x, y)
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max+0.01, 0.005), np.arange(x2_min-0.05, x2_max, 0.005))
XX = np.column_stack((xx1.ravel(), xx2.ravel()))

# Predict the output for the meshgrid
y_pred = mlp.predict(XX)

# Reshape the predictions and meshgrid arrays
atanGrad = np.reshape(y_pred, xx1.shape)/100000
fig=plt.figure()
plt.scatter(xx1*10000,xx2,c=atanGrad,marker='s',cmap='viridis',vmin=np.min(y),vmax=np.max(y))
Zp=Zp*10000
Zp=np.reshape(Zp,Por.shape)
VpVs=np.reshape(VpVs,Por.shape)
# plt.plot(Zp[-1,:],VpVs[-1,:],'b-o',linewidth = '0.5',markersize=3)
# plt.scatter(X[:,0]*10000,X[:,1],c=y,edgecolors='black')
f1=plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],s=15,cmap=cmap,norm=norm,marker='o', edgecolors='black', linewidths=0.5)

for i in range(Sw.shape[0]):
    # plt.plot(Zp[:,i],VpVs[:,i],'b-o',linewidth = '0.5',markersize=3)
        if i%20==0:
            plt.plot(Zp[i, :], VpVs[i, :], 'b-', linewidth='0.5', markersize=3)
            f2=plt.scatter(Zp[i,:],VpVs[i,:],c=y.reshape(Zp.shape)[i,:],edgecolors='black',cmap='viridis',vmin=np.min(y),vmax=np.max(y))

plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
cbar=plt.colorbar(f2)
cbar.set_label('arctan(a)')
cbar1=plt.colorbar(f1)
cbar1.set_label('Lithology')

# Create a 3D plot of the predictions
# fig = plt.figure(figsize=(10, 8))
# ax = plt.axes(projection='3d')
# ax.plot_surface(xx1*10000, xx2, yy, cmap='viridis')
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
data=df[['Zp','VpVs ratio']].to_numpy()
a=kde_2d.Kernel()
hx = np.std(df['Zp']) / 2 / len(df['Zp']) ** (1 / 6)
hy = np.std(df['VpVs ratio']) / 2 / len(df['VpVs ratio']) ** (1 / 6)
grid=np.vstack((xx1.ravel()*10000,xx2.ravel()))
a.fit(data[df['Lithology']!=0])
result=np.array(a.kernel_2d(grid.T,1/(15*hx)**2,1/(hy)**2,-atanGrad.ravel()))
plt.figure()

plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],cmap=cmap,norm=norm,s=15,marker='o', edgecolors='black', linewidths=0.5)
plt.colorbar().set_label('Lithology')
plt.contour(xx1*10000,xx2,result.reshape(xx1.shape),cmap='Reds')
dk1=np.arange(0,Zp.shape[1],1)%5==1

plt.plot(Zp[-1,dk1],VpVs[-1,dk1],'b-o',linewidth = '0.5',markersize=3)

for i in range(Por.shape[1]):
    dk=np.arange(0,Zp.shape[0],1)%20==1
    dk[-1]=True
    if i%5==1:
        plt.plot(Zp[dk, i], VpVs[dk, i], 'b-o', linewidth='0.5', markersize=3)
    # plt.scatter(Zp[:,i],VpVs[:,i],c=y.reshape(Zp.shape)[:,i],edgecolors='black')
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
plt.grid()
# save gradient result
with open('output_grad_cal.pickle','wb') as f:
    pickle.dump({'xx':xx1*10000,'yy':xx2,'mlp':mlp},f)
# df1=pd.DataFrame({'coef':coeff,'inter':inter})
# df2=pd.DataFrame({'xx':xx1,'yy':xx2})
plt.show()
