import itertools
import numpy as np
import pandas as pd

def rebalance(longAssetAmount,shortAssetAmount,longAssetPrice,shortAssetPrice,percentShort):
    cash=shortAssetAmount*shortAssetPrice*percentShort
    newShortAssetAmount=shortAssetAmount*(1-percentShort)
    newLongAssetAmount=longAssetAmount+cash/longAssetPrice
    return (newLongAssetAmount,newShortAssetAmount)

def getSmaTupples():
    short=np.arange(5,16,2)
    long=np.arange(20,61,5)
    return list(itertools.product(short,long))

def smaCross(short,long,short_before,long_before):
    if short_before<long_before and short>long:
        return 'cross-up'
    if short_before>long_before and short<long:
        return 'cross-down'
    return '-'

def getSignal(shortSma,longSma,priceSeries):
    df=priceSeries.copy().to_frame()
    rootCol='price'
    df.columns=[rootCol]
    shortSmaCol=f'sma_{shortSma}'
    longSmaCol=f'_sma_{longSma}'
    
    df[shortSmaCol]=df[rootCol].rolling(int(shortSma)).mean()
    df[longSmaCol]=df[rootCol].rolling(int(longSma)).mean()
    
    longSmaColBefore=f'{longSmaCol}_before'
    shortSmaColBefore=f'{shortSmaCol}_before'
    df[longSmaColBefore]=df[longSmaCol].shift(1)
    df[shortSmaColBefore]=df[shortSmaCol].shift(1)
    df.dropna(inplace=True)
    signal=df.apply(lambda row: smaCross(row[shortSmaCol],row[longSmaCol],row[shortSmaColBefore],row[longSmaColBefore]),axis=1)
    return signal

def trainingCutoffIndex(df,trainingSize):
    return int(len(df)*trainingSize)

def getInitialPosition(df,initialInvestment):
    firstRow=df.iloc[0]
    eth=initialInvestment/2/firstRow.price_eth
    btc=initialInvestment/2/firstRow.price_btc
    return eth,btc

def runIteration(trainDf,signal,percentShort,initialInvestment):
    resultRows=[]
    frame=trainDf.iloc[len(trainDf)-len(signal):].copy()
    eth,btc=getInitialPosition(frame,initialInvestment)
    frame['signal']=signal
    for index,row in frame.iterrows():
        if row['signal']=='cross-down':
            eth,btc=rebalance(eth,btc,row.price_eth,row.price_btc,percentShort)
        if row['signal']=='cross-up':
            btc,eth=rebalance(btc,eth,row.price_btc,row.price_eth,percentShort)
        resultRows.append({'date':row.date,'eth':eth,'btc':btc})
    return pd.DataFrame(resultRows)
    
def getSignals(trainDf):
    resultList=[]
    for shortSma,longSma in getSmaTupples():
        signal=getSignal(shortSma,longSma,trainDf['btc_eth'])
        resultList.append({
            'shortSma':shortSma,
            'longSma':longSma,
            'signalSeries':signal
        })
    return resultList

def getTrainDf(df,percentTrain):
    df.sort_values('date',inplace=True)
    return df[:trainingCutoffIndex(df,percentTrain)].copy()

def normalizeSignalLengths(signalPojos):
    lengths=[len(pojo['signalSeries'])for pojo in signalPojos]
    shortest=min(lengths)
    for pojo in signalPojos:
        signal=pojo['signalSeries']
        signal=signal[len(signal)-shortest:]
        pojo['signalSeries']=signal

def train(df,percentTrain,initialInvestment):
    trainDf=getTrainDf(df,percentTrain)
    signalPojos=getSignals(trainDf)
    normalizeSignalLengths(signalPojos)
    percentShorts=np.arange(.1,1,.1)
    results=[]
    for pojo in signalPojos:
        for ps in percentShorts:
            resultFrame=runIteration(trainDf,pojo['signalSeries'],ps,initialInvestment)
            results.append({
                'shortSma':pojo['shortSma'],
                'longSma':pojo['longSma'],
                'percentShort':ps,
                'signalSeries':pojo['signalSeries'],
                'resultFrame':resultFrame
            })
    return results

def summarizeIteration(iteration,priceDf):
    combined=iteration['resultFrame'].merge(priceDf,on='date')
    combined['portfolio_value']=combined.eth*combined.price_eth+combined.btc*combined.price_btc
    combined['perc_return']=combined.portfolio_value.pct_change()
    combined.dropna(inplace=True)
    iteration['portfolioFrame']=combined[['date','portfolio_value','perc_return']].copy()
    iteration['annualizedReturn']=(1+combined.perc_return.mean())**365
    iteration['annualizedVolatility']=365**.5*combined.perc_return.std()
    return {
        'short_sma':iteration['shortSma'],
        'long_sma':iteration['longSma'],
        'percent_short':iteration['percentShort'],
        'annualized_return':iteration['annualizedReturn'],
        'annualized_volatility':iteration['annualizedVolatility']
    }

def summarizeTraining(results,priceDf):
    summaries=[summarizeIteration(iteration,priceDf) for iteration in results]
    summary=pd.DataFrame(summaries)
    summary['sharpe']=summary.annualized_return/summary.annualized_volatility
    return summary

def getBenchmark(dates,priceDf):
    benchmarkDf=dates.to_frame()
    eth,btc=getInitialPosition(priceDf[priceDf.date==dates.min()],100000)
    benchmarkDf['btc']=np.ones(len(benchmarkDf))*btc
    benchmarkDf['eth']=np.ones(len(benchmarkDf))*eth
    benchmark={
    'shortSma':0,
    'longSma':0,
    'percentShort':0,
    'resultFrame':benchmarkDf
    }
    benchmarkSummary=summarizeIteration(benchmark,priceDf)
    return benchmark