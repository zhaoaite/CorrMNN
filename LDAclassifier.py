#!/usr/bin/python
# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA 


def Iris_label(s):
    it={b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2 }
    return it[s]
 
 
def LDA_reduce_dimension(X,y,nComponents):
    '''
    输入：X为数据集(m*n)，y为label(m*1)，nComponents为目标维数
    输出：W 矩阵（n * nComponents）
    '''
    #y1= set(y) #set():剔除矩阵y里的重复元素,化为集合的形式
    labels=list(set(y)) #list():将其转化为列表
    """
    eg:
        >>> a=[3,2,1,2]
        >>> set(a)
        {1, 2, 3} 
        >>> list(set(a))
        [1, 2, 3]
        
        >>> e=set(a)
        >>> type(e)
        <class 'set'> #集合
        >>> f=list(e)
        >>> type(f)
        <class 'list'>#列表
    """
 
 
    xClasses={} #字典
    for label in labels:
       xClasses[label]=np.array([ X[i] for i in range(len(X)) if y[i]==label ])  #list解析
    """
    x=[1,2,3,4]
    y=[5,6,7,8]
    我想让着两个list中的偶数分别相加，应该结果是2+6,4+6,2+8,4+8
    下面用一句话来写:
    >>>[a + b for a in x for b in y if a%2 == 0 and b%2 ==0]  
    """
 
    #整体均值
    meanAll=np.mean(X,axis=0) # 按列求均值，结果为1*n(行向量)
    meanClasses={}
 
    #求各类均值
    for label in labels:
        meanClasses[label]=np.mean(xClasses[label],axis=0) #1*n
 
    #全局散度矩阵
    St=np.zeros((len(meanAll), len(meanAll) ))
    St=np.dot((X - meanAll).T, X - meanAll)
 
    #求类内散度矩阵
    # Sw=sum(np.dot((Xi-ui).T, Xi-ui))   i=1...m
    Sw=np.zeros((len(meanAll), len(meanAll) )) # n*n
    for i in labels:
        Sw+=np.dot( (xClasses[i]-meanClasses[i]).T, (xClasses[i]-meanClasses[i]) )
 
    # 求类间散度矩阵
    Sb = np.zeros((len(meanAll), len(meanAll)))  # n*n
    Sb=St-Sw
 
    #求类间散度矩阵
    # Sb=sum(len(Xj) * np.dot((uj-u).T,uj-u))  j=1...k
    # Sb=np.zeros((len(meanAll), len(meanAll) )) # n*n
    # for i in labels:
    #     Sb+= len(xClasses[i]) * np.dot( (meanClasses[i]-meanAll).T.reshape(len(meanAll),1),
    #                                     (meanClasses[i]-meanAll).reshape(1,len(meanAll))
    #                                )
 
    # 计算Sw-1*Sb的特征值和特征矩阵
    eigenValues,eigenVectors=np.linalg.eig(
        np.dot( np.linalg.inv(Sw), Sb)
    )
    #提取前nComponents个特征向量
    sortedIndices=np.argsort(eigenValues) #特征值排序
    W=eigenVectors[:,sortedIndices[:-nComponents-1:-1] ] # 提取前nComponents个特征向量
    return W
 
    """
    np.argsort()
    eg:
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])
    Two-dimensional array:
    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])
    >>> np.argsort(x, axis=0)
    array([[0, 1],
           [1, 0]])
    >>> np.argsort(x, axis=1)
    array([[0, 1],
           [0, 1]])
    """
 
 
 
 
#if '__main__'== __name__:
def lda_reduse_dimension(X):    
    #1.读取数据集
#    X = np.loadtxt("./feature_extract/fisherfeature_data.txt")
    y = np.loadtxt("./tsfuse/lrts10slabel12fea.txt")
    print(X.shape)
#    #2.LDA特征提取
#    W=LDA_reduce_dimension(X, y, 10) #得到投影矩阵
#    print(W.shape)
#    newX=np.dot(X,W)# (m*n) *(n*k)=m*k
#    print(newX.shape)
#    #3.绘图
#    # 指定默认字体
#    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
#    plt.figure(1)
#    plt.scatter(newX[:,0],newX[:, 1],c=y,marker='o') #c=y,
#    plt.title('Own LDA')
 
 
    #4.与sklearn自带库函数对比
    lda_Sklearn=LinearDiscriminantAnalysis(n_components=3)
    lda_Sklearn.fit(X,y)
    newX1=lda_Sklearn.transform(X)
    lda_score=lda_Sklearn.score(X,y)
    print(lda_score)
    np.savetxt('./feature_extract/ldafeature_data.txt', newX1, fmt='%.4f')
    plt.figure(2)
    plt.scatter(newX1[:, 0], newX1[:, 1], marker='o', c=y)
    plt.title('Dimension reduction result of LDA (2d)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    ax = plt.subplot(111,projection='3d')
    ax.scatter(newX1[:, 0], newX1[:, 1],newX1[:, 2], c=y) 
    ax.set_zlabel('Z',fontsize=15)
    ax.set_ylabel('Y',fontsize=15) 
    ax.set_xlabel('X',fontsize=15) 
    plt.title('Dimension reduction result of LDA (3d)',fontsize=20)
    plt.show()
    
    
#    #    PCA comparation
#    pca=PCA(n_components=3)
#    pcafeature=pca.fit_transform(X)#对样本进行降维
#    ax = plt.subplot(111,projection='3d')
#    ax.scatter(pcafeature[:, 0], pcafeature[:, 1],pcafeature[:, 2], c=y) 
#    ax.set_zlabel('Z',fontsize=15)
#    ax.set_ylabel('Y',fontsize=15) 
#    ax.set_xlabel('X',fontsize=15) 
#    plt.title('Dimension reduction result of PCA (3d)',fontsize=20)
#    plt.show()
    
    
 
