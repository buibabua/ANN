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
# data=X.values
litho=y.values
data=X.values
M=200
x_min,x_max=np.min(data[:,0]),np.max(data[:,0])
y_min,y_max=np.min(data[:,1]),np.max(data[:,1])
xx = np.linspace(x_min, x_max, M+1 )
yy = np.linspace(y_min, y_max, M+1 )
xx,yy=np.meshgrid(xx, yy)
grid = np.vstack([xx.ravel(), yy.ravel()])
litho_class=np.unique(y)

kde_c=[]

for i in litho_class:
    kde_c.append(data[litho == i])
color=['Reds','Greens','Blues']
color1=['red','green','blue']

xx=np.linspace(x_min,x_max,M+1)
yy=np.linspace(y_min,y_max,M+1)
grid=xx,yy
for i in range(0,3):
    kde = kde_N()
    kde.fit(kde_c[i],grid,M)

    cmap1 = plt.get_cmap(color[i])
    xx = np.linspace(x_min, x_max, M )
    yy = np.linspace(y_min, y_max, M )
    xx,yy=np.meshgrid(xx, yy)
    # xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    plt.contour(xx, yy, kde.result,cmap=cmap1, linewidths=1)
    plt.scatter(kde_c[i][:,0], kde_c[i][:,1], c=color1[i], marker='o', edgecolors='black',linewidths=0.5)
    print(kde.predict(np.array([[9000,1.8]])))

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']
# n_bins = [0, 1, 2, 3,4,5,6,7,8,9]
# # colors = ['red', 'green', 'blue']
# # Create a colormap using the colors and values
# cm = plt.LinearSegmentedColormap.from_list('custom', colors)
#
# plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],s=15,cmap='rainbow',marker='o', edgecolors='black', linewidths=0.5)
plt.grid()
plt.show()

