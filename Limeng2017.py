import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from numpy import *
import random
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics
from hausdorff import hausdorff_distance

#### 数据预处理
def readData():
    f = open('C:/Users/Administrator/Desktop/oldenburg_temp2.csv')
    df = pd.read_csv(f, usecols=['id', 'time', 'x', 'y'])
    df.sort_values(by=['id','time'],ascending= (True, True), inplace=True)
    #统计id列元素的值的个数
    #也就是每个id代表的轨迹有多少个点组成
    counts = dict(df['id'].value_counts())
    #小于20的删除
    todelete = [k for k,v in counts.items() if v<20 ]
    for key in todelete:
        df = df[~(df['id'].isin([key]))]
    for i in range(20,31):
        df = df[~(df['time'].isin([i]))]
    df = df.reset_index(drop = True)
    df = df.values
    arr = df.reshape(351, 20, 4)
    return arr

def list_sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]

def Limeng(n,b):
    a = readData()
    labels = np.zeros(shape=(351,20))
    labels.dtype = 'int64'
    cluster_centers = np.zeros(shape=(n,40))
    for i in range(20):
        #print(i)
        clf = KMeans(n_clusters=n, random_state=9)
        y_pred =  clf.fit_predict(a[:, i, 2:4])
        labels[:, i] = clf.labels_
        cluster_centers[:, 2*i:(2*i+2)] = clf.cluster_centers_
    newpaths = []
    for i in range(351):
        newpath = ""
        for j in range(20):
            string = "L"+ str(labels[i, j])
            newpath += string
        newpaths.append(newpath)
    result = Counter(newpaths)
    newpathsdict = dict(result)
    #计算u
    u = mean(list(newpathsdict.values()))
    #随机生成轨迹补足数量
    while(len(newpathsdict)<351):
        key = ""
        for i in range(20):
            string = "L" + str(random.randint(0,n-1))
            key += string
        newpathsdict.setdefault(key, 0)
    valueslist = sorted(list(newpathsdict.values()),reverse=True)
    newpathslist = list_sort_by_value(newpathsdict)
    #生成噪声
    lapnoise = []
    for i in range(351):
        ln = np.random.laplace(u, b)
        while ln > 2*u or ln < 0 :
            ln = np.random.laplace(u, b)
        lapnoise.append(ln)
    #添加噪声
    vcarr = np.array(valueslist)
    lcarr = np.array(lapnoise)
    noisecount = vcarr + lcarr
    #保序回归
    x = np.arange(351)
    y = noisecount
    y_ = IsotonicRegression(increasing=False).fit_transform(x,y)
    # # 作图
    # plt.plot(x,a,"b.-",markersize=8)
    # plt.plot(x,y,"r.",markersize=8)
    # plt.plot(x,y_,"g.-",markersize=8)
    # plt.show()
    NMI=metrics.normalized_mutual_info_score(vcarr, y_)
    # print(NMI)
    vcarr.resize(351,1)
    y_.resize(351,1)
    hau_dis = hausdorff_distance(vcarr, y_, distance="euclidean")
    return NMI, hau_dis
    # print(vcarr.shape,vcarr)
    # print("Hausdorfff distance test: {0}".format(hausdorff_distance(vcarr, y_, distance="euclidean")))

nrange = [10, 20, 30, 40]
brange = [2, 1.25]
nmi = []
h_d = []
for n in nrange:
    for b in brange:
        NMI, hau_dis = Limeng(n, b)
        nmi.append(NMI)
        h_d.append(hau_dis)
print(nmi, h_d)