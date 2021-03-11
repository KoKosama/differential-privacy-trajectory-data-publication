nmi2 = [0.409454782990604, 0.409454782990604, 0.35703361178861254, 0.35703361178861254]
nmi125 = [0.2726398254068354, 0.27263982540683535, 0.2174643835147551, 0.2174643835147551]
h_d2 = [1.277276291584331, 2.087922194225566, 1.6891842609462397, 1.170724842146086]
h_d125 = [1.0738243031142716, 1.6177490888398687, 0.7473442150801333, 0.4410332961870185]
nrange = [10, 20, 30, 40]

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# 作图
plt.figure(1)
plt.plot(nrange,h_d2,"b.-",markersize=8)
plt.plot(nrange,h_d125,"r.-",markersize=8)



plt.figure(2)
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

x = np.arange(4)

bar_width = 0.35
tick_label = ["10", "20", "30", "40"]

plt.bar(x, nmi2, bar_width, align="center", color="c", label="2", alpha=0.5)
plt.bar(x+bar_width, nmi125, bar_width, color="b", align="center", label="1.25", alpha=0.5)

plt.xlabel("k")
plt.ylabel("MI")

plt.xticks(x+bar_width/2, tick_label)

plt.legend()

plt.show()