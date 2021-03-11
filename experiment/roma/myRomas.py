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
import time

import warnings
warnings.filterwarnings("ignore")

#### 数据预处理
def readData():
    f = open('E:/PyCharm/PycharmProjects/TTE/csvafterfilter/romaafterfilter.csv')
    df_test = pd.read_csv(f, dtype={'taxi_id': np.uint16, 'week': np.int8, 'mday': np.uint8, 'data_time_sec': np.uint32,
                                    'time_id': np.uint16, 'lat': np.float64, 'lon': np.float64})
    df_test = df_test[(df_test['mday'].isin([81]))]
    df_test['lat_interval'] = df_test['lat'] - df_test['lat'].shift(-1)
    df_test['lon_interval'] = df_test['lon'] - df_test['lon'].shift(-1)
    df_test = df_test[~(df_test['lat_interval'].isin([0]) & df_test['lon_interval'].isin([0]))]
    df_test = df_test.reset_index(drop=True)
    del df_test['lat_interval']
    del df_test['lon_interval']
    del df_test['week']
    del df_test['mday']
    del df_test['time_id']
    df = df_test.values
    romas = np.zeros(shape=(10000, 20, 4))
    dfshape = 638167 - 20
    for i in range(10000):
        k = np.random.randint(0, high=dfshape, size=None, dtype='l')
        for j in range(k, k + 20):
            romas[i][j - k] = df[j]
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

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

def myMechanism(n, e):
    a = readData()
    labels = np.zeros(shape=(10000, 20))
    labels.dtype = 'int64'
    cluster_centers = np.zeros(shape=(n, 40))
    starttime = time.time()
    for i in range(20):
        # print(i)
        clf = KMeans(n_clusters=n, random_state=9)
        y_pred = clf.fit_predict(a[:, i, 2:4])
        labels[:, i] = clf.labels_
        cluster_centers[:, 2 * i:(2 * i + 2)] = clf.cluster_centers_
    midtime = time.time()
    mergetime = midtime-starttime
    newpaths = []
    for i in range(10000):
        newpath = ""
        for j in range(20):
            string = "L" + str(labels[i, j])
            newpath += string
        newpaths.append(newpath)
    result = Counter(newpaths)
    newpathsdict = dict(result)

    # 随机生成轨迹补足数量
    while (len(newpathsdict) < 10000):
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
    x = np.arange(10000)
    y = np.array(truecounts)
    y_ = IsotonicRegression(increasing=False).fit_transform(x, y)
    endtime = time.time()
    dtime = endtime - starttime

    print(y)
    print(y_)
    NMI = metrics.normalized_mutual_info_score(y_, y)

    mapeyy = mape(y_, y)
    maeyy = metrics.mean_absolute_error(y, y_)
    # hausdorff_distance
    y.resize(1,10000)
    y_.resize(1,10000)
    # print(y.shape)
    hau_dis = hausdorff_distance(y, y_, distance="euclidean")


    return NMI, hau_dis,maeyy, mergetime, dtime

nrange = [20,40,60,80]
erange = [0.1, 0.2, 0.5, 0.8]
nmi = []
h_d = []
mergetime = []
dtime = []
mae1 = []
for e in erange:
    for n in nrange:
        NMI, hau_dis, Mae, Mergetime, Dtime = myMechanism(n, e)
        nmi.append(NMI)
        h_d.append(hau_dis)
        mergetime.append(Mergetime)
        dtime.append(Dtime)
        mae1.append(Mae)
    print("e %f finished" %(e))
print(nmi)
print(h_d)
# print(mape1)
print(mae1)
print(mergetime)
print(dtime)