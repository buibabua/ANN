import tensorflow as tf
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
from numpy.random import seed
import random as python_random
from sklearn.metrics import r2_score
# define colormap for litho
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']
cmap = plt.cm.colors.ListedColormap(colors)
bounds=[0,1,2,3,4,5,6,7,8,9,10]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
#-------------------------------------------
df=pd.read_csv('D:\Tai Lieu\OneDrive - Hanoi University of Science and Technology\Desktop\data1.csv',header=0)
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
# mlp=MLPRegressor(hidden_layer_sizes=(10,2),solver='lbfgs', activation='tanh',random_state=1, learning_rate_init=0.001, max_iter=1000)
# mlp.fit(X_train,y_train)
# print("Accuracy on training set: {:.5f}".format(mlp.score(X_train, y_train)))
# print("Accuracy on test set: {:.5f}".format(mlp.score(X_test, y_test)))
# zz=mlp.predict(np.column_stack((xx.ravel()/10000,yy.ravel())))[:,0].reshape(xx.shape)
plt.figure()
# ax=plt.axes(projection='3d')
# ax.plot_surface(xx,yy,zz,cmap='rainbow')
# plt.scatter(xx,yy,c=zz,marker='s',cmap='rainbow',vmin=zz.min(),vmax=zz.max())
# plt.colorbar().set_label('Porosity')
# for i in range(Por_rpt.shape[1]):
#     plt.plot(Zp_rpt.reshape(Por_rpt.shape)[:,i],VpVs_rpt.reshape(Por_rpt.shape)[:,i],'b-o',linewidth = '0.5',markersize=3)
#     if i%2==0:
#         plt.plot(Zp_rpt.reshape(Por_rpt.shape)[:, i], VpVs_rpt.reshape(Por_rpt.shape)[:, i], 'b-', linewidth='0.5', markersize=3)
# plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Porosity'],cmap='rainbow',s=15,marker='o', edgecolors='black', linewidths=0.5,vmin=zz.min(),vmax=zz.max())
# plt.xlabel('Zp')
# plt.ylabel('Vp Vs ratio')

# Define your MLP model
# tf.reset_default_graph()




# Set the random seed for NumPy
np.random.seed(1234)

# Set the random seed for Python's built-in random module
python_random.seed(1234)

# Set the random seed for TensorFlow
tf.random.set_seed(1234)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='tanh',input_shape=(2,)),
  tf.keras.layers.Dense(2, activation='tanh'),
  # tf.keras.layers.Dense(1)
])

# Define your custom loss function
def custom_loss(y_true, y_pred):
  # Calculate the sum of squared errors between the model function and the data
  # porosity data
  mse1 = tf.reduce_mean((y_true[:,0] - y_pred[:,0])**2)

  # Calculate the gradients of the model function
  with tf.GradientTape() as tape:
    tape.watch(y_pred[:,0])
    gradients = tape.gradient(y_pred[:,0], model.trainable_variables)

  # Calculate the sum of squared errors between the gradients of the model function and the gradients of the data
  mse2 = tf.reduce_mean((gradients - y_true[:,1])**2)
  # mse2=0
  # Combine the two loss terms
  loss = mse1 + mse2

  return loss
# def gradient
# Compile your model with your custom loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss=custom_loss, metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Train your model on your training data
model.fit(X_train, y_train, epochs=2000, batch_size=32,verbose=0)
# evaluate model
# predict on test data
y_pred = model.predict(X_test)

# compute R2 score
r2 = r2_score(y_test, y_pred)
print('R2 score:', r2)
loss, mae = model.evaluate(X_test, y_test)
print('Test accuracy:', mae)
zz1=model.predict(np.column_stack((xx.ravel()/10000,yy.ravel())))[:,0].reshape(xx.shape)
plt.scatter(xx,yy,c=zz1,marker='s',cmap='rainbow',vmin=df['Porosity'].min(),vmax=df['Porosity'].max())

for i in range(Por_rpt.shape[1]):
    plt.plot(Zp_rpt.reshape(Por_rpt.shape)[:,i],VpVs_rpt.reshape(Por_rpt.shape)[:,i],'b-o',linewidth = '0.5',markersize=3)
    if i%2==0:
        plt.plot(Zp_rpt.reshape(Por_rpt.shape)[:, i], VpVs_rpt.reshape(Por_rpt.shape)[:, i], 'b-', linewidth='0.5', markersize=3)
plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Porosity'],cmap='rainbow',s=15,marker='o', edgecolors='black', linewidths=0.5,vmin=df['Porosity'].min(),vmax=df['Porosity'].max())
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
plt.colorbar().set_label('Porosity')
plt.show()