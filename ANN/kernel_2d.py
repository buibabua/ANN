import numpy as np
import matplotlib.pyplot as plt
class Kernel:
    def fit(self,X):
        self.X=X
        return self
    def kernel_2d(self,grid, hx, hy,theta=0):
        X=self.X[:,0]
        Y=self.X[:,1]
        n = self.X.shape[0]
        result = []
        for i in range(grid.shape[0]):
            x,y=grid[i,:]

            K = 1 / (2 * np.pi * hx * hy)
            # f = 1 / n * np.sum(K * np.exp(-0.5 * (x * np.ones((1, len(X))) - X) ** 2 / hx ** 2 - 0.5 * (y * np.ones((1,len(Y))) - Y) ** 2 / hy ** 2))
            f=1/n*np.sum(self.rotated_gaussian(x * np.ones((1, len(X))) - X,y * np.ones((1,len(Y))) - Y,hx,hy,theta[i]))
            result.append(f)
        return result

    def rotated_gaussian(self, x_diff,y_diff, lambda1, lambda2, theta):

        # theta=0
        a = lambda1 * np.cos(theta) ** 2 + lambda2 * np.sin(theta) ** 2
        b = 1 / 2 * (-lambda1 * np.sin(2 * theta) + lambda2 * np.sin(2 * theta))
        c = lambda1 * np.sin(theta) ** 2 + lambda2 * np.cos(theta) ** 2
        det = 1 / lambda1 / lambda2
        return 2 * np.pi * det ** (-1 / 2) * np.exp(-1 / 2 * (a * x_diff ** 2 + 2 * b * x_diff * y_diff + c * y_diff ** 2))

#
# #example
# Generate some 2D data
# np.random.seed(0)
# X = np.random.randn(100, 2)
#
# # Create a grid over which to evaluate the KDE
# xmin, xmax = -3, 3
# ymin, ymax = -3, 3
#
# xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# grid = np.vstack([xx.ravel(), yy.ravel()]).T
# a = Kernel()
# a.fit(X)
# t=a.kernel_2d(grid,0.3,0.3)
# Z=np.exp(t)
# Z=Z.reshape(xx.shape)
# plt.contour(xx,yy,Z)
# plt.scatter(X[:,0],X[:,1])
# plt.show()
