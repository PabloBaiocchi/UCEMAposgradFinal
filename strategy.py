import scipy.stats as sciStats
import itertools as it
import pandas as pd
import numpy as np
import datetime as dt

def error(sigma,n,p):
    return sciStats.t.ppf(1-p,n-1)*sigma/n**.5

def upperSignal(longMean,longStd,long,short,shortMean,p):
    upperBound=longMean+error(longStd,long,p)
    return {
        'p':p,
        'short':short,
        'long':long,
        'signal':shortMean>upperBound
    }

def lowerSignal(longMean,longStd,long,short,shortMean,p):
    lowerBound=longMean-error(longStd,long,p)
    return {
        'p':p,
        'short':short,
        'long':long,
        'signal':shortMean<lowerBound
    }

def buySellSignal(lowerSignal,upperSignal):
    sig=np.where(lowerSignal['signal'],'buy',np.where(upperSignal['signal'],'sell','-'))
    return {
        'signal':sig,
        'lower_p':lowerSignal['p'],
        'lower_short':lowerSignal['short'],
        'lower_long':lowerSignal['long'],
        'upper_p':upperSignal['p'],
        'upper_short':upperSignal['short'],
        'upper_long':upperSignal['long']
    }

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

def getSignals(returns,shortRange,longRange,upperPRange,lowerPRange):
    upperSignals=[]
    lowerSignals=[]
    for short in shortRange:
        shortMean=returns.rolling(short).mean()
        
        for long in longRange:
            window=returns.shift(short).rolling(long)
            longMean=window.mean()
            longStd=window.std()
            
            upperSignals=upperSignals+[upperSignal(longMean,longStd,long,short,shortMean,p) for p in upperPRange]
            lowerSignals=lowerSignals+[lowerSignal(longMean,longStd,long,short,shortMean,p) for p in lowerPRange]
            
    return lowerSignals, upperSignals

def train(trainingSet,buySellSignals):
    index=len(trainingSet)
    rows=[]
    for sig in buySellSignals:
        position=getPosition(sig['signal'][:index])
        rows.append({
            'returns':trainingSet[position=='bought'].mean(),
            'upper_short':sig['upper_short'],
            'upper_long':sig['upper_long'],
            'upper_p':sig['upper_p'],
            'lower_short':sig['lower_short'],
            'lower_long':sig['lower_long'],
            'lower_p':sig['lower_p'],
            'signal':sig['signal']
        })
    df=pd.DataFrame(rows)
    winner=df.sort_values('returns',ascending=False).iloc[0]
    return {
        'returns':trainingSet[position=='bought'].mean(),
        'benchmark_returns':trainingSet.mean(),
        'benchmark_std':trainingSet.std(),
        'upper_short':winner.upper_short,
        'upper_long':winner.upper_long,
        'upper_p':winner.upper_p,
        'lower_short':winner.lower_short,
        'lower_long':winner.lower_long,
        'lower_p':winner.lower_p,
        'signal':winner.signal
    }

def strategy(pSeries,trainingSize,shortRange,longRange,upperPRange,lowerPRange):
    returns=pSeries.pct_change()
    lowerSignals, upperSignals=getSignals(returns,shortRange,longRange,upperPRange,lowerPRange)
    buySellSignals=[buySellSignal(lower,upper) for lower,upper in it.product(lowerSignals,upperSignals)]
    
    trainingCutoff=int(len(pSeries)*trainingSize)
    trainingSet=returns[:trainingCutoff]
    testSet=returns[trainingCutoff:]
    testSet.reset_index(drop=True,inplace=True) #need to 0 index so index matches position index later
    
    trainingOutcome=train(trainingSet,buySellSignals)
    position=getPosition(trainingOutcome['signal'][trainingCutoff:])
    
    return {
        'upper_short':trainingOutcome['upper_short'],
        'lower_short':trainingOutcome['lower_short'],
        'upper_long':trainingOutcome['upper_long'],
        'lower_long':trainingOutcome['lower_long'],
        'upper_p':trainingOutcome['upper_p'],
        'lower_p':trainingOutcome['lower_p'],
        'training_returns':trainingOutcome['returns'],
        'training_benchmark_returns':trainingOutcome['benchmark_returns'],
        'training_benchmark_std':trainingOutcome['benchmark_std'],
        'returns':testSet[position=='bought'].mean(),
        'benchmark_returns':testSet.mean(),
        'benchmark_std':testSet.std()
    }

def run(data,trainingSize,shortRange,longRange,upperPRange,lowerPRange):
    rows=[]
    colAmount=len(data.columns)
    startTime=dt.datetime.now()
    lastStart=startTime
    for i,ticker in enumerate(data.columns):
        results=strategy(data[ticker],trainingSize,shortRange,longRange,upperPRange,lowerPRange)
        results['ticker']=ticker
        rows.append(results)
        lastStart=durationPrint(i,colAmount,ticker,startTime,lastStart)
    return pd.DataFrame(rows)

def durationPrint(i,colAmount,ticker,startTime,lastStart):
    now=dt.datetime.now()
    roundDuration=(now-lastStart).seconds
    print(f'{i+1}/{colAmount}: {ticker}')
    print(f'\tduration: {roundDuration} seconds')
    print(f'\ttotal duration: {(now-startTime).seconds/60} minutes')
    print(f'\testimated time remaining: {(colAmount-1-i)*roundDuration/60} minutes')
    return now