'''
NAIVE BAYES ALGORITHM
'''
#朴素贝叶斯算法：用于特征之间是相互独立的情况，并假设样本集是按照一定分布产生的

import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB #三种朴素贝叶斯分类器：正态、多项式、两点
                                                                     #三种分类器代表特征服从什么分布
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pylab import mpl #解决中文和-显示问题
mpl.rcParams['font.sans-serif'] = ['FangSong'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

iris = load_iris()
data = iris.data[:,:2] #利用前两列的特征进行训练
target = iris.target

for method in [GaussianNB(),MultinomialNB(),BernoulliNB()]:

    method.fit(data,target)

    x = np.linspace(data[:,0].min(),data[:,0].max(),100)
    y = np.linspace(data[:,1].min(),data[:,1].max(),100)
    xx,yy = np.meshgrid(x,y)
    test_p = np.c_[xx.flatten(),yy.flatten()] #网格化画布，以两个特征作为两个做标注，用来观察算法的预测结果
    test_class = method.predict(test_p)

    cmap1 = ListedColormap(['r','g','b'])

    plt.scatter(test_p[:,0],test_p[:,1],c=test_class) #利用分类器绘制网格化数据的预测结果
    plt.scatter(data[:,0],data[:,1],c=target,cmap=cmap1) #绘制样本点
    plt.title(f"使用{method}方法进行分类的结果")

    plt.show()