import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from numpy import *
import random
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics
from hausdorff import hausdorff_distance
import pywt
import math
import copy

import warnings
warnings.filterwarnings("ignore")

#### 数据预处理
def readData():
    f = open('D:/研究生/oldenburg_temp3.csv')
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

# 构建haar树
def haarWaveletTransform(data):
    coeffs = pywt.dwt(data, 'haar')
    # (cA, cD) : tuple   Approximation and detail coefficients.
    cA, cD = coeffs
    cA /= math.sqrt(2)
    cD /= math.sqrt(2)
    return cA,cD


def buildHaarTreeList(data):
    haarTreeList = []
    while(len(data)>=2):
        cA,cD = haarWaveletTransform(data)
        cD = cD.tolist()
        cD.reverse()
        # print(cA, cD)
        haarTreeList.extend(cD)
        data = cA
        if(len(data) == 1):
            cA = cA.tolist()
            cA.reverse()
            haarTreeList.extend(cA)
    haarTreeList.reverse()
    return haarTreeList

# 重建
def rebuildHaarTreeList(list):
    n = int(math.log(len(list), 2))
    # print(n)
    cA = list[0:1]
    for i in range(n):
        cD = list[int(math.pow(2,i)):int(math.pow(2,i+1))]
        data = math.sqrt(2) * pywt.idwt(cA, cD, 'haar')
        cA = data
    return data

# 添加噪声
def addNoise(n, eps):
    laplaceNoise = []
    h = int(math.log(n, 2))
    i=0
    while(i<n):
        if(i==0):
            layernumber = 0
        else:
            layernumber = int(math.log(i, 2))
        noise = np.random.laplace(0, (1 )/ (eps * int(math.pow(2, h - layernumber))))
        laplaceNoise.append(noise)
        i = i+1
    return laplaceNoise


def myMechanism(n, e):
    a = readData()
    labels = np.zeros(shape=(958, 20))
    labels.dtype = 'int64'
    cluster_centers = np.zeros(shape=(n, 40))
    for i in range(20):
        # print(i)
        clf = KMeans(n_clusters=n, random_state=9)
        y_pred = clf.fit_predict(a[:, i, 2:4])
        labels[:, i] = clf.labels_
        cluster_centers[:, 2 * i:(2 * i + 2)] = clf.cluster_centers_
    newpaths = []
    for i in range(958):
        newpath = ""
        for j in range(20):
            string = "L" + str(labels[i, j])
            newpath += string
        newpaths.append(newpath)
    result = Counter(newpaths)
    newpathsdict = dict(result)

    # 随机生成轨迹补足数量
    while (len(newpathsdict) < 958):
        key = ""
        for i in range(20):
            string = "L" + str(random.randint(0, n - 1))
            key += string
        newpathsdict.setdefault(key, 0)

    # 补齐2的n次方haar变换和重构
    values = list(newpathsdict.values())
    a = int(math.log(len(values), 2))
    for index in range(int(math.pow(2, a + 1)) - len(values)):
        values.append(0)

    temp = buildHaarTreeList(values)
    noise = addNoise(len(temp), 0.5)
    c = [temp[i] + noise[i] for i in range(len(temp))]
    noisecounts = rebuildHaarTreeList(c)

    i = 0
    newpathsdict2 = copy.deepcopy(newpathsdict);
    for key, value in newpathsdict.items():
        newpathsdict[key] = noisecounts[i]
        i = i + 1

    valueslist = sorted(list(newpathsdict.values()), reverse=True)
    newpathslist = list_sort_by_value(newpathsdict)

    truecounts = []
    for item in newpathslist:
        truecounts.append(newpathsdict2.get(item))

    # 根据一致性约束 trueconts 保序回归
    x = np.arange(958)
    y = np.array(truecounts)
    y_ = IsotonicRegression(increasing=False).fit_transform(x, y)

    NMI = metrics.normalized_mutual_info_score(y_, y)
    # hausdorff_distance
    y.resize(1,958)
    y_.resize(1,958)
    hau_dis = hausdorff_distance(y, y_, distance="euclidean")
    return NMI, hau_dis

nrange = [20,40,60,80]
erange = [0.1, 0.2, 0.5, 0.8]
nmi = []
h_d = []
for e in erange:
    for n in nrange:
        NMI, hau_dis = myMechanism(n, e)
        nmi.append(NMI)
        h_d.append(hau_dis)
    print("e %f finished" %(e))
print(nmi, h_d)