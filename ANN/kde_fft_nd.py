import numpy as np
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import time
class kde_N:
    def fit(self,data,grid,M):
        self.data=data
        self.grid=grid
        P = int(np.ceil(np.round(2 ** np.log2(3 * M - 1))))
        nd = data.shape[1]  # n dimesion
        data_matrix = np.zeros((P, P))
        kde_matrix = np.zeros_like(data_matrix)
        # set data
        # bins=[np.linspace(np.min(data[:,0]),np.max[:,1],M),np.linspace(np.min(data[:,0]),np.max[:,1],M)]
        binx = np.sort(grid[0].tolist())
        biny = np.sort(grid[1].tolist())
        data_binnied, edges = np.histogramdd(data, bins=(binx, biny))
        # create grid extent
        grid_width1 = (np.max(grid[0]) - np.min(grid[0])) / (M - 1)
        grid_width2 = (np.max(grid[1]) - np.min(grid[1])) / (M - 1)

        xrange = np.linspace(-(M - 1) * grid_width1, (M - 1) * grid_width1, 2 * M - 1)
        yrange = np.linspace(-(M - 1) * grid_width2, (M - 1) * grid_width2, 2 * M - 1)

        # xrange = np.linspace(-np.max(data[:,0]), np.max(data[:,0]), 2 * M )
        # yrange = np.linspace(-np.max(data[:,1]), np.max(data[:,1]), 2 * M )
        grid_extend = np.meshgrid(xrange, yrange)
        # insert to zeros matrix
        data_matrix[M:2 * M, M:2 * M] = data_binnied.T
        kde_matrix[0:2 * M - 1, 0:2 * M - 1] = 1 / len(data[:, 0]) * self.Gaussian(grid_extend)
        # fft
        fft_data = np.fft.fftn(data_matrix)
        fft_kde = np.fft.fftn(kde_matrix)
        #  invert fft
        kde = 1 / P ** 2 * np.real(np.fft.ifftn(fft_data * fft_kde))
        kde[2 * M - 1:3 * M - 1, 2 * M - 1:3 * M - 1]
        # plt.scatter(data[:,0],data[:,1])
        # xx, yy = np.mgrid[np.min(self.grid[0]):np.max(self.grid[0]):100j, np.min(self.grid[1]):np.max(self.grid[1]):100j]
        # plt.contour(xx, yy, kde[2 * M - 1:3 * M - 1, 2 * M - 1:3 * M - 1], linewidths=1)
        # plt.show()
        self.result = kde[2 * M - 1:3 * M - 1, 2 * M - 1:3 * M - 1]
        return self
    def Gaussian(self,data):
        X = data[0]
        Y = data[1]
        h1 = np.std(X) / 2 / len(X) ** (1 / 6)
        h2 = np.std(Y) / 2 / len(Y) ** (1 / 6)
        return 1 / (2 * np.pi * h1 * h2) * np.exp(-(0.5 * X ** 2 / h1 / h1 + 0.5 * Y ** 2 / h2 / h2))

    def predict(self,data):
        binx = np.sort(self.grid[0].tolist())
        biny = np.sort(self.grid[1].tolist())
        data_binnied, edges = np.histogramdd(data, bins=(binx, biny))
        data_binnied = data_binnied.T
        indices=np.where(data_binnied==1)
        return self.result[indices]


df=pd.read_csv('D:\Tai Lieu\OneDrive - Hanoi University of Science and Technology\Desktop\data1.csv',header=0)
df['VpVs ratio']=df['P-wave']/df['S-wave']
df['Zp']=df['P-wave']*df['Density']
# Split the data into training and testing sets
X=df[['Zp','VpVs ratio']]
y=df['Lithology']
#
data=X.values
litho=y.values
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
xx=np.linspace(x_min,x_max,M+1)
yy=np.linspace(y_min,y_max,M+1)
grid=xx,yy
for i in range(0,3):
    kde = kde_N()
    time.sleep(1)
    start_time = time.time()
    kde.fit(kde_c[i], grid, M)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, " seconds"," ",len(kde_c[i]),"points")

    cmap1 = plt.get_cmap(color[i])
    xx = np.linspace(x_min, x_max, M )
    yy = np.linspace(y_min, y_max, M )
    xx,yy=np.meshgrid(xx, yy)
    # xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    plt.contour(xx, yy, kde.result,cmap=cmap1, linewidths=1)
    # print(kde.predict(np.array([[9000,1.8]])))

# colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']
# n_bins = [0, 1, 2, 3,4,5,6,7,8,9]
# # colors = ['red', 'green', 'blue']
# # Create a colormap using the colors and values
# cm = LinearSegmentedColormap.from_list(
#         'cmap', colors, N=n_bins)
# cmap = plt.cm.colors.ListedColormap(colors)
plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Lithology'],s=15,marker='o', edgecolors='black', linewidths=0.5)
plt.grid
plt.show()

