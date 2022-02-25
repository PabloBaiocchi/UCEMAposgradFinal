import strategy as s
import pandas as pd
import numpy as np

data=pd.read_csv('large_cap.csv')
data.drop('Date',axis='columns',inplace=True)
trainingSize=.2
shortRange=np.linspace(10,25,5,dtype=int)
longRange=np.linspace(50,75,5,dtype=int)
lowerPRange=np.linspace(.00000001,.0001,5)
upperPRange=np.linspace(.0001,.025,5)

results=s.run(data,trainingSize,shortRange,longRange,upperPRange,lowerPRange)
results.to_csv('largeCapTwoSidedResults.csv')