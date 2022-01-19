import requests
import pandas as pd

def getRawData(symbol,alphavantageApiKey):
    baseUrl='https://www.alphavantage.co/query'
    params={
        'function':'DIGITAL_CURRENCY_DAILY',
        'symbol':symbol,
        'market':'USD',
        'apikey':alphavantageApiKey
    }
    response=requests.get(baseUrl,params=params)
    return response.json()

def rawToDf(raw):
    timeSeries=raw['Time Series (Digital Currency Daily)']
    rows=[{'date':key,'price':timeSeries[key]['4a. close (USD)']} for key in timeSeries.keys()]
    df=pd.DataFrame(rows)
    df['date']=pd.to_datetime(df.date)
    df['price']=df.price.astype(float)
    return df

def getData(alphavantageApiKey):
    btcRaw=getRawData('BTC',alphavantageApiKey)
    ethRaw=getRawData('ETH',alphavantageApiKey)

    ethDf=rawToDf(ethRaw)
    btcDf=rawToDf(btcRaw)

    df=ethDf.merge(btcDf,on='date',suffixes=['_eth','_btc'])
    df['btc_eth']=df.price_btc/df.price_eth
    df.sort_values('date',inplace=True)
    return df

def storeData(filePath,alphavantageApiKey):
    data=getData(alphavantageApiKey)
    data.to_csv(filePath)
