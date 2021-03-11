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
    f = open('E:/PyCharm/PycharmProjects/TTE/csvafterfilter2/20140830afterfilter2.csv')
    df_test = pd.read_csv(f, dtype = {'taxi_id' : np.uint16, 'week' :np.int8, 'y_n' : np.uint8, 'lat1' : np.float32, 'long1' : np.float32, 'distance_interval' : np.float64, 'data_time_sec' : np.uint32, 'time_id' : np.uint})
    del df_test['week']
    del df_test['y_n']
    del df_test['distance_interval']
    del df_test['time_id']
    df = df_test.values
    romas = np.zeros(shape=(1000, 20, 4))
    dfshape = 19513351 - 20
    for i in range(1000):
        k = np.random.randint(0, high=dfshape, size=None, dtype='l')
        for j in range(k, k + 20):
            romas[i][j - k] = df[j]
            romas[i][j - k][3] = df[j][2]
            romas[i][j - k][2] = df[j][1]
            romas[i][j - k][0] = i
            romas[i][j - k][1] = j - k
    return romas

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
    h = math.log(n, 2)
    i=0
    while(i<n):
        if(i==0):
            layernumber = 0
        else:
            layernumber = int(math.log(i, 2))
        noise = np.random.laplace(0, (1+h)/ (eps * int(math.pow(2, h - layernumber))))
        laplaceNoise.append(noise)
        i = i+1
    return laplaceNoise

def myMechanism(n, e):
    a = readData()
    labels = np.zeros(shape=(1000, 20))
    labels.dtype = 'int64'
    cluster_centers = np.zeros(shape=(n, 40))
    for i in range(20):
        # print(i)
        clf = KMeans(n_clusters=n, random_state=9)
        y_pred = clf.fit_predict(a[:, i, 2:4])
        labels[:, i] = clf.labels_
        cluster_centers[:, 2 * i:(2 * i + 2)] = clf.cluster_centers_
    newpaths = []
    for i in range(1000):
        newpath = ""
        for j in range(20):
            string = "L" + str(labels[i, j])
            newpath += string
        newpaths.append(newpath)
    result = Counter(newpaths)
    newpathsdict = dict(result)

    # 随机生成轨迹补足数量
    while (len(newpathsdict) < 1000):
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
    # print(len(temp))
    noise = addNoise(len(temp), e)
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
    x = np.arange(1000)
    y = np.array(truecounts)
    y_ = IsotonicRegression(increasing=False).fit_transform(x, y)