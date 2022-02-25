import strategy as s
import pandas as pd
import numpy as np

data=pd.read_csv('large_cap.csv')
data.drop('Date',axis='columns',inplace=True)
trainingSize=.2
shortRange=np.linspace(3,35,5,dtype=int)
longRange=np.linspace(50,90,5,dtype=int)
upperPRange=np.linspace(.00000001,.025,5)
lowerPRange=np.linspace(.00000001,.01,5)

results=s.run(data,trainingSize,shortRange,longRange,upperPRange,lowerPRange)
results.to_csv('overnight.csv')