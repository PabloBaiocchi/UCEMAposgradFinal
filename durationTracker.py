import datetime as dt

class DurationTracker:

    def __init__(self,colAmount):
        self.colAmount = colAmount

    def start(self):
        now=dt.datetime.now()
        self.startTime=now
        self.lastTick=now

    def update(self,i,ticker):
        now=dt.datetime.now()
        roundDuration=(now-self.lastTick).seconds
        self.lastTick=now
        print(f'{i+1}/{self.colAmount}: {ticker}')
        print(f'\tduration: {roundDuration} seconds')
        print(f'\ttotal duration: {(now-self.startTime).seconds/60} minutes')
        print(f'\testimated time remaining: {(self.colAmount-1-i)*roundDuration/60} minutes')