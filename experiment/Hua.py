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
    f = open('E:\PyCharm\PycharmProjects\differential-privacy-trajectory-data-publication\oldenburg_temp3.csv')
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

    #计算u
    u = mean(list(newpathsdict.values()))

    #随机生成轨迹补足数量
    while(len(newpathsdict)<958):
        key = ""
        for i in range(20):
            string = "L" + str(random.randint(0,n-1))
            key += string
        newpathsdict.setdefault(key, 0)

    # 保存tc
    truecounts = [i for i in newpathsdict.values()]

    #生成噪声
    lapnoise = []
    for i in range(958):
        ln = np.random.laplace(0, b)
        lapnoise.append(ln)

    #添加噪声 newpathsdict备份至newpathsdict2，之后newpathsdict包含噪声
    i = 0
    newpathsdict2 = copy.deepcopy(newpathsdict);
    for key, value in newpathsdict.items():
        newpathsdict[key] = value + lapnoise[i]
        i = i + 1
        # 保存tc
    noisecounts = [i for i in newpathsdict.values()]

    x = np.array(truecounts)
    y = np.array(noisecounts)

    print(x, y)
    maexy = metrics.mean_absolute_error(x, y)

    NMI = metrics.normalized_mutual_info_score(x, y)

    return NMI, maexy

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
    print("b %f finished" % (b))
print(nmi)
print(mae1)