import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from numpy import *
import random
import copy
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics
from hausdorff import hausdorff_distance

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
    romas = np.zeros(shape=(1000, 20, 4))
    dfshape = 638167 - 20
    for i in range(1000):
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

def Limeng(n,b):
    a = readData()
    labels = np.zeros(shape=(1000,20))
    labels.dtype = 'int64'
    cluster_centers = np.zeros(shape=(n,40))
    for i in range(20):
        #print(i)
        clf = KMeans(n_clusters=n, random_state=9)
        y_pred =  clf.fit_predict(a[:, i, 2:4])
        labels[:, i] = clf.labels_
        cluster_centers[:, 2*i:(2*i+2)] = clf.cluster_centers_
    newpaths = []
    for i in range(1000):
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
    while(len(newpathsdict)<1000):
        key = ""
        for i in range(20):
            string = "L" + str(random.randint(0,n-1))
            key += string
        newpathsdict.setdefault(key, 0)

    # 保存tc
    truecounts = [i for i in newpathsdict.values()]

    #生成噪声
    lapnoise = []
    for i in range(1000):
        ln = np.random.laplace(u, b)
        lapnoise.append(ln)

    #添加噪声
    i = 0
    for key, value in newpathsdict.items():
        newpathsdict[key] = value + lapnoise[i]
        i = i + 1

    # 保存tc
    noisecounts = [i for i in newpathsdict.values()]

    x = np.array(truecounts)
    y = np.array(noisecounts)

    print(x,y)
    maexy = metrics.mean_absolute_error(x, y)

    NMI = metrics.normalized_mutual_info_score(y, x)


    return NMI,maexy

nrange = [ 20,  40,  60,  80 ]
brange = [10, 5, 2, 1.25]
nmi = []
mae1 = []
for b in brange:
    for n in nrange:
        NMI, Mae = Limeng(n, b)
        nmi.append(NMI)
        mae1.append(Mae)
        print("n %f finished" % (n))
    print("e %f finished" % (b))
print(nmi)
print(mae1)