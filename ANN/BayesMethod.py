import numpy as np
from kernel_2d import Kernel
from kde_fft_nd import kde_N
class BayesClass:

    def fit(self,X,y,grid,M):
        self.X = X
        self.y = y
        self.n_sample = X.shape[0]
        self.kde_c = []
        n_sample = X.shape[0]
        self.classes = np.unique(y)
        for c in self.classes:
            X_class= X[c == y]
            self.prior_prob = X_class.shape[0] / self.n_sample
            kde=kde_N()

            self.kde_c.append(kde.fit(X_class,grid,M))

    def predict(self,X):
        y_pred = []
        n_sample=len(self.y)
        classes=np.unique(self.y)
        # tinh xac suat co dieu kien cua feature voi tung class


        for x in X:
            posteriors = []
            x=np.array([x])
            for idx, c in enumerate(self.classes):
                posterior = np.log(self.prior_prob)+np.sum(np.log(self.pdf(idx,x)))
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return y_pred

    def pdf(self,idx,X):
        return self.kde_c[idx].predict(X)