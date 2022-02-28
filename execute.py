import strategy as s
import pandas as pd
import numpy as np

data=pd.read_csv('large_cap.csv')
data.drop('Date',axis='columns',inplace=True)

targetFile='results.csv'
trainingSize=.2
longRange=np.linspace(50,100,5,dtype=int)
shortRange=[int(l/3) for l in longRange]
lowerPRange=[.000000001,.00000001,.0000001,.000001]
upperPRange=[.0000001,.000001,.00001]

results=s.run(data,trainingSize,shortRange,longRange,upperPRange,lowerPRange)
results.to_csv(targetFile)