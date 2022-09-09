'''
K-NEAREST NEIGHBOUR
'''
#基于kd树的KNN算法

from sklearn.datasets import load_iris #数据集
from sklearn.model_selection import cross_val_score #交叉验证方法
from sklearn.neighbors import KNeighborsClassifier #KNN核心函数
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pylab import mpl #解决中文和-显示问题
mpl.rcParams['font.sans-serif'] = ['FangSong'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

iris = load_iris() #加载鸢尾花数据集
x = iris.data #将数据点和其对应的类别分别储存
y = iris.target

k_range = range(1,31)
k_error = []
for k in k_range: #取k从1~30，查看当选择不同k值时的误分类情况
    knn = KNeighborsClassifier(n_neighbors=k, #n_neighbors参数决定KNN中的k选择多大的值
                               algorithm='kd_tree', #algorithm参数决定KNN中采用哪种算法（还有'brute'直接暴力计算,'ball_tree'是球状树）
                               leaf_size=50, #leaf_size参数设置叶子结点的最大数目
                               metric='euclidean') #metric参数设置距离度量方法（'euclidean','manhattan','chebyshev','minkowski'）
    scores = cross_val_score(knn,x,y,cv=6,scoring='precision_weighted') #cv参数决定S-fold验证中的S数值。这里6代表按照5:1划分训练集和测试集
                                                                        #scoring参数决定返回的是S-fold后每个子数据集正确分类的比例
    k_error.append(1 - scores.mean())

plt.plot(k_range,k_error) #作图观察：x轴为k值，y值为误差值
plt.plot(k_error.index(min(k_error)) + 1,k_error[k_error.index(min(k_error))],'ko')
plt.xlabel('k值')
plt.ylabel('误分类率')
plt.title('选择k值')
plt.show()
print(f"当k取{k_error.index(min(k_error)) + 1}时，效果最好")

k_value = 12

#-----------------------------------------------------------------------------------------------------------------------

X = iris.data[:,:2] #选择鸢尾花的两个特征
y = iris.target

cmap_light = ListedColormap(['orange','cyan','cornflowerblue']) #创建色彩图
cmap_bold = ListedColormap(['darkorange','c','darkblue'])

# 在两种权重下绘制图像
for weights in ['uniform','distance']: #两种表决方法。uniform为多数表决，distance为具有权重的多数表决（越近越重要）

    clf = KNeighborsClassifier(k_value,weights=weights)
    clf.fit(X,y)

    #网格化画布，便于绘制依据模型预测的结果
    x_min, x_max = X[:,0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20) #绘制训练集中的点
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"不同表决方法下的分类情况（k = {k_value}, weights = {weights}）" )

plt.show()