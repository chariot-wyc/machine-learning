'''
NONLINEAR SUPPORT VECTOR MACHINE
'''
#非线性SVM分类器与数据可视化

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.datasets import make_moons
from pylab import mpl #解决中文和-显示问题
mpl.rcParams['font.sans-serif'] = ['FangSong'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

X,y = make_moons(n_samples=100, noise=0.1) #X为数据点，y为对应数据点的类别
print(X,y)
x0s = np.linspace(-1.5,2.5,100)
x1s = np.linspace(-1.0,1.5,100)
x0,x1 = np.meshgrid(x0s,x1s)
Xtest = np.c_[x0.ravel(),x1.ravel()]

#使用SVC建模
svc_clf = SVC(kernel='poly',degree=3,coef0=0.2) #kernel要选择核函数，poly为多项式核函数
                                                #degree为多项式核函数的次数
                                                #coef0为多项式核函数的常数项值
svc_clf.fit(X,y)
yPred1 = svc_clf.predict(Xtest).reshape(x0.shape) #预测分类结果
#使用NuSVC建模
nusvc_clf = NuSVC(kernel='rbf',gamma=1,nu=0.1) #rbf为高斯核函数
                                               #gamma与高斯核函数的参数sigma square成反比，过大容易过拟合，过小欠拟合（默认'scale'为1）
                                               #nu为错分率的百分比上限（支持向量的下限）
nusvc_clf.fit(X,y)
yPred2 = nusvc_clf.predict(Xtest).reshape(x0.shape) #预测分类结果

ax = plt.gca() #设置画布
plt.title("非线性支持向量机")
ax.contourf(x0,x1,yPred1,cmap=plt.cm.brg,alpha=0.1) #绘制模型1分类结果，cmap用于绘制等高线颜色，alpha用于调色
ax.contourf(x0,x1,yPred2,cmap='PuBuGn_r',alpha=0.1) #绘制模型2分类结果
ax.plot(X[:,0][y==0],X[:,1][y==0],"bo") #按类绘制数据样本点
ax.plot(X[:,0][y==1],X[:,1][y==1],"r^")
ax.grid(True,which='both')

plt.show()