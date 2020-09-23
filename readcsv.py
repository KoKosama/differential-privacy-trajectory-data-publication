import pandas as pd
import numpy as np

f = open('C:/Users/Administrator/Desktop/oldenburg_temp1.csv')
df_test = pd.read_csv(f, usecols=['id', 'time', 'x', 'y'])
print(f)
