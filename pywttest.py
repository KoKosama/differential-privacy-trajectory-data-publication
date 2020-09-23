import pywt
import math
import numpy as np
odata = [2,4,6,8,10,12,14,16,18,20]
a = int(math.log(958, 2))
# print(n)

for index in range(int(math.pow(2,a+1))-958):
    odata.append(0)


print(len(odata))
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

def rebuildHaarTreeList(list):
    n = int(math.log(len(list), 2))
    # print(n)
    cA = list[0:1]
    for i in range(n):
        cD = list[int(math.pow(2,i)):int(math.pow(2,i+1))]
        data = math.sqrt(2) * pywt.idwt(cA, cD, 'haar')
        cA = data
    return data

def addNoise(n, eps):
    laplaceNoise = []
    h = int(math.log(n, 2))
    i=0
    while(i<n):
        if(i==0):
            layernumber = 0
        else:
            layernumber = int(math.log(i, 2))
        print(layernumber)
        noise = np.random.laplace(0, (1 + h)/ (eps * int(math.pow(2, h - layernumber))))
        laplaceNoise.append(noise)
        i = i+1
    return laplaceNoise

temp = buildHaarTreeList(odata)
n = len(temp)
noise = addNoise(n, 0.5)
print(temp)
print(noise)
# print(rebuildHaarTreeList(temp))

c = [temp[i] + noise[i] for i in range(len(temp))]
print(c)
print(rebuildHaarTreeList(c))




print(cA,cD)
print(temp)
print(math.sqrt(2)* pywt.idwt(cA, cD, 'haar'))
print(pywt.wavelist('haar'))