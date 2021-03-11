import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from numpy import *
import random
from sklearn import metrics
from hausdorff import hausdorff_distance
import pywt
import math

#### 数据预处理
def readData():
    f = open('C:/Users/Administrator/Desktop/oldenburg_temp3.csv')
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
    arr = df.reshape(958, 20, 4)
    return arr

def list_sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]

def Limeng(n,b):
    a = readData()
    labels = np.zeros(shape=(958,20))
    labels.dtype = 'int64'
    cluster_centers = np.zeros(shape=(n,40))
    for i in range(20):
        #print(i)
        clf = KMeans(n_clusters=n, random_state=9)
        y_pred =  clf.fit_predict(a[:, i, 2:4])
        labels[:, i] = clf.labels_
        cluster_centers[:, 2*i:(2*i+2)] = clf.cluster_centers_
    newpaths = []
    for i in range(958):
        newpath = ""
        for j in range(20):
            string = "L"+ str(labels[i, j])
            newpath += string
        newpaths.append(newpath)
    result = Counter(newpaths)
    newpathsdict = dict(result)


    #随机生成轨迹补足数量
    while(len(newpathsdict)<958):
        key = ""
        for i in range(20):
            string = "L" + str(random.randint(0,n-1))
            key += string
        newpathsdict.setdefault(key, 0)


    # 取出truecount
    truecount = list(newpathsdict.values())
    cA, cD = pywt.dwt(truecount, 'haar')
    # 加噪重构
    lapnoiseca = []
    b = 2
    for i in range(479):
        ln = math.sqrt(2) * np.random.laplace(0, b)
        lapnoiseca.append(ln)
    lapnoisecd = []
    for i in range(479):
        ln = math.sqrt(2) * np.random.laplace(0, b)
        lapnoisecd.append(ln)
    cAnoise = np.array(lapnoiseca) + cA
    cDnoise = np.array(lapnoisecd) + cD
    noisecount = pywt.idwt(cAnoise, cDnoise, 'haar')
    # # 作图
    # plt.plot(x,a,"b.-",markersize=8)
    # plt.plot(x,y,"r.",markersize=8)
    # plt.plot(x,y_,"g.-",markersize=8)
    # plt.show()
    NMI=metrics.normalized_mutual_info_score(truecount, noisecount)
    # print(NMI)
    truecount.resize(1,958)
    noisecount.resize(1,958)
    hau_dis = hausdorff_distance(truecount,  noisecount, distance="euclidean")
    return NMI, hau_dis
    # print(vcarr.shape,vcarr)
    # print("Hausdorfff distance test: {0}".format(hausdorff_distance(vcarr, y_, distance="euclidean")))

nrange = [10, 20, 30, 40, 50, 60, 70, 80 ,90]
brange = [10, 5, 2, 1.25]
nmi = []
h_d = []
for b in brange:
    for n in nrange:
        NMI, hau_dis = Limeng(n, b)
        nmi.append(NMI)
        h_d.append(hau_dis)
print(nmi, h_d)