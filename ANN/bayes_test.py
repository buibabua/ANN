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
df=pd.read_csv('data1.csv',header=0)

df['VpVs ratio']=df['P-wave']/df['S-wave']
df['Zp']=df['P-wave']*df['Density']
# Split the data into training and testing sets
X=df[['Zp','VpVs ratio']]
y=df['Lithology']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
test=BayesClass()
test.fit(X_train.to_numpy(),y_train.to_numpy())

# Tạo lưới điểm trên không gian 2D để tính toán mật độ xác suất

x_min, x_max = X.iloc[:,0].min(), X.iloc[:,0].max() # zp
y_min, y_max =X.iloc[:,1].min(), X.iloc[:,1].max() # vp/vs
xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
# xx, yy = np.mgrid[8000:10500:50j, 1.5:2:50j]
grid = np.vstack([xx.ravel(), yy.ravel()]).T
# plot real data

# plt.scatter(df.loc[df['Lithology']==2,['Zp']],df.loc[df['Lithology']==2,['VpVs ratio']],s=10)
# kde = Kernel()
# zz = np.exp(test.pdf(1,grid)).reshape(xx.shape)
# plt.xlabel('Zp')
# plt.ylabel('Vp Vs ratio')
# cbar=plt.colorbar()
# cbar.set_label('Litho', rotation=270)
# cbar.ax.get_yaxis().labelpad = 15
# plt.contour(xx, yy, zz,20)
#
# plt.grid()
# plt.show()
# # 3d
# fig=plt.figure()
# ax=fig.add_subplot(111,projection='3d')
# surf=ax.plot_surface(xx,yy,zz,cmap='coolwarm')
# plt.xlabel('Zp')
# plt.ylabel('Vp Vs ratio')
#
# plt.ylim(2,2.6)
# plt.xlim(8000,12000)
# plt.show()
# draw real data
plt.subplot(1,2,1)
# Tính toán mật độ xác suất trên lưới điểm
for idx,obj in enumerate(test.kde_c):
    zz = np.exp(test.pdf(idx,grid)).reshape(xx.shape)
    # plt.contour(xx, yy, zz,cmap='rainbow')

plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],cmap='rainbow',s=10)
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')
cbar=plt.colorbar()
cbar.set_label('Litho', rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.grid()
# plot visual data
# plt.subplot(1,2,2)
# litho_pred_test=test.predict(X_test.to_numpy())
# litho_pred=test.predict(X.to_numpy())
# # Tính toán mật độ xác suất trên lưới điểm
# # color=['red','blue','green']
# # for idx,obj in enumerate(test.kde_c):
# #     zz = np.exp(test.pdf(idx,grid)).reshape(xx.shape)
# #     plt.contour(xx, yy, zz,20,colors=color[idx])
#
# # Vẽ đồ thị mật độ xác suất
#
# plt.scatter(df['Zp'],df['VpVs ratio'],c=litho_pred,cmap='rainbow',s=10)
# plt.xlabel('Zp')
# plt.ylabel('Vp Vs ratio')
# plt.grid()
# cbar = plt.colorbar()
# cbar.set_label('Litho', rotation=270)
# cbar.ax.get_yaxis().labelpad = 15
# accuracy = (litho_pred_test == y_test).sum() / len(y_test)
# print(accuracy)
# plt.show()
#
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
color=['Reds','Greens','Blues']

# Create a colormap using the colors and values

for idx,obj in enumerate(test.kde_c):
    zz = np.exp(test.pdf(idx,grid)).reshape(xx.shape)
    cmap1=plt.get_cmap(color[idx])
    ax.contour(xx, yy, zz,cmap=cmap1, linewidths=1)

plt.show()

# Create a list of values to use as the colormap
values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create a list of colors to use for each value in the colormap
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']
# colors = ['red', 'green', 'blue']
# Create a colormap using the colors and values
cmap = plt.cm.colors.ListedColormap(colors)
# Create a colorbar to display the colormap
fig.colorbar(plt.imshow(np.arange(len(values)).reshape((1, len(values))), cmap=cmap, aspect='auto'),
             ticks=np.arange(len(values))+0.5, boundaries=np.arange(len(values)),
             label='Colorbar Title')
ax.scatter(df['Zp'], df['VpVs ratio'], c=df['Lithology'], cmap=cmap, s=15, marker='o', edgecolors='black',
               linewidths=0.5)
ax.set_xlabel('Zp')
ax.set_ylabel('Vp Vs ratio')
# plt.show()
# cbar=plt.colorbar()
# cbar.set_label('Litho', rotation=270)
# cbar.ax.get_yaxis().labelpad = 15
ax.grid()
plt.show()
# cm = confusion_matrix(y_test, litho_pred_test)
# cm[i,:]/np.sum(cm[i,:])
# import seaborn as sns
# # Create heatmap
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# # Add labels and title
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
# exit()