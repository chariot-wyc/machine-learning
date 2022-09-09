'''
LINEAR SUPPORT VECTOR MACHINE
'''
#线性SVM分类器与数据可视化

import numpy as np
import matplotlib.pyplot as plt #绘图所用的库
from sklearn import datasets #取得数据库
from sklearn.svm import SVC #核心函数，线性支持向量分类函数
from pylab import mpl #解决中文和-显示问题
mpl.rcParams['font.sans-serif'] = ['FangSong'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

iris = datasets.load_iris() #加载鸢尾花数据。共150个数据，列表示花的参数（比如花瓣长度等）。前面的数组是data部分，表示数据，后面的target部分
                            #表示该数据所代表花属于哪一类（用0、1、2三类来表示）
X = iris['data'][:,[2,3]] #取data部分的3、4列，分别代表petal length、petal width
X = (X - np.mean(X)) / np.std(X) #对数据进行标准化
y = (iris['target'] == 2) #将Virginica类的鸢尾花实例的类别设定为正类（True），其余类为负类（False）

clf = SVC(kernel='linear',C=2) #svc即support vector classification
                                 #C为惩罚参数，C越大，对于误分类的惩罚越大
clf.fit(X,y) #fit进数据集

print("\nSVM model: Y = w0 + w1*x1 + w2*x2") #分类超平面模型
print(f"截距: w0={clf.intercept_}")  #w0: 截距
print(f"系数: w1={clf.coef_}")  # w1,w2: 系数
print(f"分类准确度：{clf.score(X, y)}")  #对训练集的分类准确度

pred = clf.predict([[5.5,1.7]]) #试着使用得到的模型进行预测，判断新输入的数据是否属于第2类
print(pred)

ax = plt.gca() #设置画布
plt.title("线性支持向量机")

xx = np.linspace(-1,3,100)
yy = np.linspace(-1.5,0.5,50)
YY,XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

p01 = ax.scatter(X[:99,0],X[:99,1],color='r',s=4) #画0、1类的点，用红色注明
p2 = ax.scatter(X[100:149,0],X[100:149,1],color='b',s=4) #画2类的点，用蓝色注明
ax.contour(XX,YY,Z,colors='k',levels=[-1, 0, 1],alpha=0.5,linestyles=['--','-','--']) #画分割线、超平面
ax.scatter(clf.support_vectors_[:, 0],clf.support_vectors_[:, 1], s=100,linewidth=1, facecolors='none', edgecolors='k')  #绘制支持向量

plt.show() #这句用于显示图形