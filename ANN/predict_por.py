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
# def custom_loss(y_true,y_pre):
#     loss= np.sum((y_true-y_pre)**2)
#     return loss
# def cost_function():
#     return np.sum(active_function())
# def active_function(x):

#     return np.tanh(x)
# def model():
#     return np.sum(active_function(x))
# def cost_function_grad(gradf,grad_data):
#     return np.sum((f))

# def objective_function(params,zp,vpvs):



# define colormap for litho
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']
cmap = plt.cm.colors.ListedColormap(colors)
bounds=[0,1,2,3,4,5,6,7,8,9,10]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
#-------------------------------------------
df=pd.read_csv('data1.csv',header=0)
df['VpVs ratio']=df['P-wave']/df['S-wave']
df['Zp']=df['P-wave']*df['Density']
with open('RPT_variables.pickle', 'rb') as f:
    loaded_vars = pickle.load(f)
Zp_rpt=loaded_vars['zp']
VpVs_rpt=loaded_vars['VpVs']
arctan_rpt=loaded_vars['actan']
Por_rpt=loaded_vars['Porosity']
Sw_rpt=loaded_vars['Sw']
with open('output_grad_cal.pickle','rb') as f:
    load_vars=pickle.load(f)
mlp_grad=load_vars['mlp']

Por=df['Porosity']
xx=load_vars['xx']*10000
yy=load_vars['yy']
grad_pre=mlp_grad.predict(np.column_stack((df['Zp'].to_numpy()/10000,df['VpVs ratio'].to_numpy())))
X=df[['Zp','VpVs ratio']]
Y=np.vstack((Por,grad_pre)).T
X.iloc[:,0]=X.iloc[:,0]/10000
#------- add data rpt----------
# X=np.vstack((np.column_stack((Zp_rpt/10000,VpVs_rpt)),X.values))
# Y=np.vstack((np.column_stack((Por_rpt.ravel(),arctan_rpt*100000)),Y))
#---------------------------
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=42)
mlp=MLPRegressor(hidden_layer_sizes=(5,),solver='adam', activation='tanh',random_state=1, learning_rate_init=0.001, max_iter=1000)
mlp.fit(X_train,y_train)
print("Accuracy on training set: {:.5f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.5f}".format(mlp.score(X_test, y_test)))
zz=mlp.predict(np.column_stack((xx.ravel()/10000,yy.ravel())))[:,0].reshape(xx.shape)
plt.figure()
# ax=plt.axes(projection='3d')
# ax.plot_surface(xx,yy,zz,cmap='rainbow')
plt.scatter(xx,yy,c=zz,marker='s',cmap='rainbow',vmin=zz.min(),vmax=zz.max())
plt.colorbar().set_label('Porosity')
# for i in range(Por_rpt.shape[1]):
#     plt.plot(Zp_rpt.reshape(Por_rpt.shape)[:,i],VpVs_rpt.reshape(Por_rpt.shape)[:,i],'b-o',linewidth = '0.5',markersize=3)
#     if i%2==0:
#         plt.plot(Zp_rpt.reshape(Por_rpt.shape)[:, i], VpVs_rpt.reshape(Por_rpt.shape)[:, i], 'b-', linewidth='0.5', markersize=3)
plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Porosity'],cmap='rainbow',s=15,marker='o', edgecolors='black', linewidths=0.5,vmin=zz.min(),vmax=zz.max())
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
plt.show()