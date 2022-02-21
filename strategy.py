import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import itertools

def getSmaTupples():
    short=np.arange(5,16,2)
    long=np.arange(20,61,5)
    return list(itertools.product(short,long))

def getPosition(signal):
    pos=['bought'] if signal[0]=='buy' else ['-']

    for i in range(1,len(signal)):
        if signal[i]=='buy':
            pos.append('bought')
        elif pos[i-1]=='bought' and signal[i]!='sell':
            pos.append('bought')
        else:
            pos.append('-')
            
    return pd.Series(pos)

def backtest(priceSeries,short,long,alpha):
    df=priceSeries.to_frame()
    df.reset_index(inplace=True,drop=True)
    df.columns=['price']
    df['returns']=df.price.pct_change()
    df['u_long']=df.returns.rolling(long).mean()
    df['s_long']=df.returns.rolling(long).std()
    t=stats.t.ppf(1-alpha/2,long-1)
    df['E_long']=t*df.s_long/long**.5
    df['floor']=df.u_long-df.E_long
    df['ceiling']=df.u_long+df.E_long
    df['u_short']=df.returns.rolling(short).mean()
    df['signal']=np.where(df.u_short<df.floor,'buy',np.where(df.u_short>df.ceiling,'sell','-'))
    df['position']=getPosition(df.signal)
    return df

def plotPosition(bt,figsize=(15,10)):
    posSeries=np.where(bt.position=='bought',bt.price,float('nan'))
    plt.figure(figsize=figsize)
    plt.plot(bt.index,bt.price,color='grey')
    plt.plot(bt.index,posSeries,color='green')
    
def statistics(bt,short,long,alpha):
    return {
        'short':short,
        'long':long,
        'alpha':alpha,
        'strategy_returns':bt[bt.position=='bought'].returns.mean(),
        'strategy_std':bt[bt.position=='bought'].returns.std(),
        'benchmark_returns':bt.returns.mean(),
        'benchmark_std':bt.returns.std()
    }

def train(priceSeries,alpha):
    smaTupples=getSmaTupples()
    rows=[]
    for short,long in smaTupples:
        bt=backtest(priceSeries,short,long,alpha)
        rows.append(statistics(bt,short,long,alpha))
    return pd.DataFrame(rows)

def strategy(priceSeries,trainingSize,alpha):
    pSeries=priceSeries.reset_index(drop=True)
    
    indexCutoff=int(len(pSeries)*trainingSize)
    
    trainingResults=train(pSeries[:indexCutoff],alpha)
    bestIteration=trainingResults.sort_values('strategy_returns',ascending=False).iloc[0]
    
    short=int(bestIteration.short)
    long=int(bestIteration.long)
    bt=backtest(pSeries[indexCutoff:],short,long,alpha)
    
    tradingStats=statistics(bt,short,long,alpha)
    tradingStats['training_strategy_returns']=bestIteration.strategy_returns
    tradingStats['training_strategy_std']=bestIteration.strategy_std
    tradingStats['training_benchmark_returns']=bestIteration.benchmark_returns
    tradingStats['training_strategy_std']=bestIteration.benchmark_std
    
    return bt,tradingStats