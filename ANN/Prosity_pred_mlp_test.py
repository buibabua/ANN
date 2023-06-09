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
import scipy.optimize
from sklearn.metrics import r2_score
from scipy.interpolate import griddata
from sklearn import preprocessing
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size,random_state=1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.Ws = []
        self.bs = []
        np.random.seed(random_state)
        count=0
        self.W=None
        bias=None
        for i in range(len(hidden_sizes)):
            if i==0:
                count+=input_size*hidden_sizes[i]
                count+=hidden_sizes[i]
                # self.W=self.initialize_weights(self.input_size,hidden_sizes[i]).ravel()
                self.W=self._init_coef(self.input_size,hidden_sizes[i])[0].ravel()
                bias=self._init_coef(self.input_size,hidden_sizes[i])[1].ravel()
            if i!=0:
                count+=hidden_sizes[i-1]*hidden_sizes[i]
                count += hidden_sizes[i]
                self.W=np.concatenate((self.W,self._init_coef(self.hidden_sizes[i-1],hidden_sizes[i])[0].ravel()),axis=None)
                bias=np.concatenate((bias,self._init_coef(self.hidden_sizes[i-1],hidden_sizes[i])[1].ravel()),axis=None)
        self.W=np.concatenate((self.W,self._init_coef(self.hidden_sizes[i],output_size)[0].ravel()),axis=None)
        bias=np.concatenate((bias,self._init_coef(self.hidden_sizes[i],output_size)[1].ravel()))
        self.W=np.concatenate((self.W,bias),axis=0)
        count+=output_size*hidden_sizes[-1]
        count+=output_size
        # self.W = np.concatenate((self.W, 1*np.ones((1, count - len(self.W)))),axis=None)
        # self.W=np.random.randn(count)

        for i in range(len(hidden_sizes)):
            if i == 0:
                self.Ws.append(np.random.randn(self.input_size, self.hidden_sizes[i]))
            else:
                self.Ws.append(np.random.randn(self.hidden_sizes[i-1], self.hidden_sizes[i]))
            self.bs.append(np.zeros((1, self.hidden_sizes[i])))
        self.Ws.append(np.random.randn(self.hidden_sizes[-1], self.output_size))
        self.bs.append(np.zeros((1, self.output_size)))
        self.bias=bias

    def relu(self, x):
        return np.tanh(x)

    def linear(self, x):
        return x

    # def forward(self,Ws,bs,X):
    #     Ws=self.Ws
    #     bs=self.bs
    #     self.zs = []
    #     self.as_ = [X]
    #     for i in range(len(self.Ws)):
    #         z = np.dot(self.as_[-1], self.Ws[i]) + self.bs[i]
    #         if i == len(self.Ws) - 1:
    #             a = self.linear(z)
    #         else:
    #             a = self.relu(z)
    #         self.zs.append(z)
    #         self.as_.append(a)
    #     return self.as_[-1]
    def forward(self,X):
        self.zs = []
        self.as_ = [X]
        for i in range(len(self.Ws)):
            z = np.dot(self.as_[-1], self.Ws[i]) + self.bs[i]
            if i == len(self.Ws) - 1:
                a = self.linear(z)
            else:
                a = self.relu(z)
            self.zs.append(z)
            self.as_.append(a)
        # self.W=W
        return self.as_[-1]

    def mse_loss(self, y_true, y_pred):




        # gradient with model
        x, y = self.xx_norm, self.yy_norm
        # zz_model = griddata((self.xx.ravel(), self.yy.ravel()), self.forward(np.column_stack((self.xx.ravel(),self.yy.ravel())))[:,0],
        #                     (x,y), method='linear')
        # input_rpt=np.column_stack((self.rpt_zp_norm.ravel(), self.rpt_vpvs_norm.ravel()))
        input_rpt = np.column_stack((self.xx_norm.ravel(), self.yy_norm.ravel()))
        # input_rpt=np.column_stack((input_rpt, np.ones((input_rpt.shape[0],1))))
        zz_model = self.forward(input_rpt)[:, 0].reshape(x.shape)

        # zz_model =self.forward(input_rpt)[:,0]
        # zz_grid_model= griddata((self.rpt_zp_norm.ravel(), self.rpt_vpvs_norm.ravel()),zz_model,(x,y),method='linear')
        gradx = np.gradient(zz_model, axis=1)
        grady = np.gradient(zz_model, axis=0)
        arctan_model = np.arctan(grady / gradx)
        # arctan_model[self.arctan_rpt_norm==0]=0
        # y_grad_pred=griddata((x.ravel(),y.ravel()),arctan_model.ravel(),(self.X_norm),method='linear')
        # y_grad_pred[np.isnan(y_grad_pred)]=0
        # y_grad_true=griddata((x.ravel(),y.ravel()),self.arctan_rpt_norm.ravel(),(self.X_norm),method='linear')
        grad_model = np.sqrt(gradx ** 2 + grady ** 2)
        # grad_model[self.arctan_rpt_norm==0]=0

        # plt.scatter(x,y,c=self.grad_rpt,vmin=np.min(self.grad_rpt),vmax=np.max(self.grad_rpt))
        # plt.colorbar()
        # plt.show()
        # Add L2 regularization term to loss
        # zz_model[self.rpt_por_norm == 0] = 0
        values = 0
        loss=0
        n_samples=self.X.shape[0]
        for s in self.Ws:
            s = s.ravel()

            values += np.dot(s, s)

        loss += (0.5 * 0.0001) * values / n_samples
        loss = loss+ 10*np.mean((y_true - y_pred) ** 2) +15*np.mean((zz_model-self.rpt_por_norm)**2)
        loss+=0*np.mean((self.arctan_rpt_norm.ravel()-arctan_model.ravel())**2)+1000*np.mean((grad_model.ravel() - self.grad_amp_por_norm.ravel())**2)
        # loss=loss+np.mean((y_true-y_pred)**2)+np.mean((y_grad_pred-self.y_grad_true)**2)
        # self.zz=zz_model
        return loss

    def cost_function(self,W,X,y_true):
        Ws = []
        bs = []
        tmp = 0
        for i in range(len(self.Ws)):
            Ws.append(W[tmp:tmp + self.Ws[i].size].reshape(self.Ws[i].shape))
            tmp += self.Ws[i].size
        for i in range(len(self.bs)):
            bs.append(W[tmp:tmp + self.bs[i].size].reshape(self.bs[i].shape))
            tmp += self.bs[i].size
        self.Ws=Ws
        self.bs=bs
        self.W=W
        y_pred=self.forward(X)
        return self.mse_loss(y_true,y_pred)
    def update_weights(self,W):
        Ws = []
        bs = []
        tmp = 0
        for i in range(len(self.Ws)):
            Ws.append(W[tmp:tmp + self.Ws[i].size].reshape(self.Ws[i].shape))
            tmp += self.Ws[i].size
        for i in range(len(self.bs)):
            bs.append(W[tmp:tmp + self.bs[i].size].reshape(self.bs[i].shape))
            tmp += self.bs[i].size
        self.Ws = Ws
        self.bs = bs

    def backward(self, X, y):
        obj_func=lambda W: self.cost_function(W,X,y)
        # delta = self.as_[-1] - y
        # self.W=scipy.optimize.fmin_bfgs(obj_func,self.W)
        result= scipy.optimize.minimize(obj_func, self.W,method='L-BFGS-B')



        # result = scipy.optimize.minimize(obj_func, self.W, method='Nelder-Mead',options={'maxiter':1000})
        # minimizer_kwargs = {"method": "BFGS"}
        # result = scipy.optimize.basinhopping(obj_func, self.W,  minimizer_kwargs=minimizer_kwargs, niter=200)
        self.W=result.x
        self.update_weights(self.W)

    def fit(self, X, y,rpt_por,rpt_zp,rpt_vpvs,xx,yy, learning_rate=0.01, num_epochs=1, verbose=True):

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.X=X
        self.y=y
        self.xx=xx
        self.yy=yy
        self.rpt_por=rpt_por
        self.rpt_zp=rpt_zp
        self.rpt_vpvs=rpt_vpvs
        self.gradx_rpt = np.gradient(self.rpt_por, axis=1)
        self.grady_rpt = np.gradient(self.rpt_por, axis=0)
        self.arctan_rpt = np.arctan(self.grady_rpt / self.gradx_rpt)/np.linalg.norm(np.arctan(self.grady_rpt / self.gradx_rpt))
        self.arctan_rpt[np.isnan(self.arctan_rpt)] = 0
        self.grad_rpt = np.sqrt(self.gradx_rpt ** 2 + self.grady_rpt ** 2)/np.linalg.norm(np.sqrt(self.gradx_rpt ** 2 + self.grady_rpt ** 2))
        self.grad_rpt[np.isnan(self.grad_rpt)] = 0
        # normalize data
        X=X.to_numpy()
        self.X_norm=np.ones(X.shape)

        self.X_norm[:,0]=X[:,0]/10000
        self.X_norm[:,1]=X[:,1]

        self.y_norm = np.ones(y.shape)
        for i in range(y.shape[1]):
            self.y_norm[:,i]=y[:,i]#/np.linalg.norm(y[:,i])
        self.rpt_zp_norm=self.rpt_zp/np.linalg.norm(self.rpt_zp)
        self.rpt_vpvs_norm=self.rpt_vpvs/np.linalg.norm(self.rpt_vpvs)
        self.rpt_por_norm=self.rpt_por#/np.linalg.norm(self.y[:,0])
        self.gradx_rpt_norm = np.gradient(self.rpt_por_norm, axis=1)
        self.grady_rpt_norm = np.gradient(self.rpt_por_norm, axis=0)
        self.arctan_rpt_norm = np.arctan(self.grady_rpt_norm / self.gradx_rpt_norm)
        self.arctan_rpt_norm[np.isnan(self.arctan_rpt_norm)]=0
        self.grad_amp_por_norm=np.sqrt(self.gradx_rpt_norm**2+self.grady_rpt_norm**2)
        self.grad_amp_por_norm[np.isnan(self.grad_amp_por_norm)]=0
        self.rpt_por_norm[np.isnan(self.rpt_por_norm)]=0
        self.xx_norm = xx/10000 #/np.linalg.norm(X[:,0])
        self.yy_norm = yy #/np.linalg.norm(X[:,1])
        # self.y_grad_true = griddata((self.xx_norm.ravel(), self.yy_norm.ravel()), self.arctan_rpt_norm.ravel(), (self.X_norm), method='linear')
        # self.y_grad_true[np.isnan(self.y_grad_true)]=0
        for i in range(num_epochs):
            self.backward(self.X_norm, self.y_norm)
            y_pred = self.forward(self.X_norm)
            loss = self.mse_loss(self.y_norm, y_pred)
            if verbose:
                print('Epoch %d, loss = %.4f' % (i, loss))
        return self

    def predict(self, X):

        X[:,0]=X[:,0]/10000
        result=np.ones((X.shape[0],self.y.shape[1]))
        for i in range(self.y.shape[1]):
            result[:,i]=self.forward(X)[:,i]#*np.linalg.norm(self.y[:,i])
        return result

    def initialize_weights(self,n_in, n_out):
        variance = 6 / (n_in + n_out)
        stddev = np.sqrt(variance)
        weights = np.random.normal(loc=0.0, scale=stddev, size=(n_in, n_out))
        return weights
    def _init_coef(self, fan_in, fan_out):
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.0
        # if self.activation == "logistic":
        #     factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = np.random.uniform(-init_bound, init_bound, (fan_in, fan_out))
        intercept_init = np.random.uniform(-init_bound, init_bound, fan_out)
        return coef_init, intercept_init




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
# [df['Lithology']!=0]
# [df['Lithology']!=0]
xx=load_vars['xx']
yy=load_vars['yy']
# grad_pre=mlp_grad.predict(np.column_stack((df['Zp'].to_numpy()/10000,df['VpVs ratio'].to_numpy())))
X=df[['Zp','VpVs ratio']]
# [df['Lithology']!=0]
# [df['Lithology']!=0]
# Y=np.vstack((Por,grad_pre)).T
t=griddata((Zp_rpt,VpVs_rpt),Por_rpt.ravel(),(xx,yy), method='linear')
gradx = np.gradient(t, axis=1)
grady = np.gradient(t, axis=0)
grad_pred=np.arctan(grady/gradx)
grad_pred[np.isnan(grad_pred)]=0
grad_por_rpt=griddata((xx.ravel(),yy.ravel()),grad_pred.ravel(),(X.iloc[:,[0,1]]), method='linear')
grad_por_rpt[np.isnan(grad_por_rpt)]=0
Y=np.vstack((Por,grad_por_rpt)).T
# [(df['Lithology']!=0).to_numpy().any(axis=1)]
#------- add data rpt----------
# X=np.vstack((np.column_stack((Zp_rpt/10000,VpVs_rpt)),X.values))
# Y=np.vstack((np.column_stack((Por_rpt.ravel(),arctan_rpt*100000)),Y))
#---------------------------
X_train, X_test, y_train, y_test = train_test_split(X,Y[:,0][:,np.newaxis],test_size=0.1,random_state=42)
mlp=MLP(2,(5,),1,random_state=1)

t=griddata((Zp_rpt,VpVs_rpt),Por_rpt.ravel(),(xx,yy), method='linear')
mlp.fit(X_train,y_train,t,Zp_rpt,VpVs_rpt,xx,yy)
y_pred=mlp.predict(X_test.to_numpy())
r2 = r2_score(y_test[:,0], y_pred[:,0])
# r2_1=r2_score(y_test[:,1], y_pred[:,1])
print('R2 score prosity:', r2)
zz=mlp.predict(np.column_stack((xx.ravel(),yy.ravel()))).reshape(xx.shape)
plt.figure()
# ax=plt.axes(projection='3d')
# ax.plot_surface(xx,yy,zz,cmap='rainbow')
h1=plt.scatter(xx,yy,c=zz,marker='s',cmap='rainbow',vmin=df['Porosity'].min(),vmax=df['Porosity'].max())
plt.colorbar(h1).set_label('Porosity')
for i in range(Por_rpt.shape[1]):
    plt.plot(Zp_rpt.reshape(Por_rpt.shape)[:,i],VpVs_rpt.reshape(Por_rpt.shape)[:,i],'b-o',linewidth = '0.5',markersize=3)
    if i%2==0:
        plt.plot(Zp_rpt.reshape(Por_rpt.shape)[:, i], VpVs_rpt.reshape(Por_rpt.shape)[:, i], 'b-', linewidth='0.5', markersize=3)
h2=plt.scatter(df['Zp'],df['VpVs ratio'],c=df['Porosity'],cmap='rainbow',s=15,marker='o', edgecolors='black', linewidths=0.5,vmin=df['Porosity'].min(),vmax=df['Porosity'].max())
plt.colorbar(h2).set_label('Porosity')
plt.xlabel('Zp')
plt.ylabel('Vp Vs ratio')

plt.show()