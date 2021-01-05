import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from pandas_datareader import data

#Get stock data from Yahoo Finance

tickers = ['GME']

#Sepecify start and end date
start = datetime(2010, 1, 1)
todayDate = datetime.today().strftime('%Y-%m-%d')

#Create dataframe with asset data
df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, todayDate)['Adj Close'] for ticker in tickers]).T
df.columns = tickers

# plt.plot(df['GME'])
# plt.legend('GME')
# plt.title('stocks adjusted close')
# plt.show()

#Simple moving average with a 30 day window
sma30 = pd.DataFrame()
sma30['Adj. Close Price'] = df['GME'].rolling(window=30).mean()

# #Simple moving average with a 100 day window
sma100 = pd.DataFrame()
sma100['Adj. Close Price'] = df['GME'].rolling(window=100).mean()

 #Visualize new data

plt.figure(figsize = (12, 10))
plt.plot(df['GME'], label = 'GME')
plt.plot(sma30['Adj. Close Price'], label = 'SMA30')
plt.plot(sma100['Adj. Close Price'], label = 'SMA100')
plt.xlabel('GME adjusted close price')
plt.ylabel('Adj. Close price history')
plt.legend(loc='upper left')
#plt.show()

# #Create new dataframe to store new variables (sma30, sma100 and stock)
dataGME = pd.DataFrame()

dataGME['GME'] = df['GME']
dataGME['SMA30'] = sma30['Adj. Close Price']
dataGME['SMA100'] = sma100['Adj. Close Price']

 #Create logical function to determined BUY/SELL
def buy_sell(data):
    PriceBuy = []
    PriceSell = []
    flag = -1            #Flag tells me when the moving averages crossed each other

    for i in range(len(data)):
        if data['SMA30'][i] > data['SMA100'][i]:
            if flag != 1:
                PriceBuy.append(data['GME'][i])
                PriceSell.append(np.nan)
                flag = 1
            else:
                PriceBuy.append(np.nan)
                PriceSell.append(np.nan)

        elif data['SMA30'][i] < data['SMA100'][i]:
            if flag != 0:
                PriceBuy.append(np.nan)
                PriceSell.append(data['GME'][i])
                flag = 0
            else:
                PriceBuy.append(np.nan)
                PriceSell.append(np.nan)
        else:
            PriceBuy.append(np.nan)
            PriceSell.append(np.nan)

    return(PriceBuy, PriceSell)

 #Store buy and sell data into a variable
buy_sell = buy_sell(dataGME)

dataGME['Buy_Signal'] = buy_sell[0]
dataGME['Sell_Signal'] = buy_sell[1]


#Visualize data and trading strategy
plt.figure(figsize=(12.5, 4.5))

plt.plot(dataGME['GME'], label = 'GME')
plt.plot(dataGME['SMA30'], label = 'SMA30')
plt.plot(dataGME['SMA100'], label = 'SMA100')
plt.scatter(dataGME.index, dataGME['Buy_Signal'], label='Buy', marker='o', color='green')
plt.scatter(dataGME.index, dataGME['Sell_Signal'], label='Sell', marker='v', color='red')
plt.title('GME Adj. Close History with Buy/Sell Signals')
plt.ylabel('GME Adj. Close Price')
plt.legend(loc='upper left')
plt.show()