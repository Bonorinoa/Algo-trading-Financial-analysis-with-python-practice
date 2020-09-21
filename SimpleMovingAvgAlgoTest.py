import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

#Get stock data from Quandl
import quandl
apple = quandl.get("WIKI/AAPL")

#Visualize the data
#plt.figure(figsize = (8, 6))
#plt.plot(apple['Adj. Close'], label = 'AAPL')
#plt.xlabel('Apple adjusted close price')
#plt.ylabel('Adj. Close price history')
#plt.show()

#Simple moving average with a 30 day window
sma30 = pd.DataFrame()
sma30['Adj. Close Price'] = apple['Adj. Close'].rolling(window=30).mean()

#Simple moving average with a 100 day window
sma100 = pd.DataFrame()
sma100['Adj. Close Price'] = apple['Adj. Close'].rolling(window=100).mean()

#Visualize new data
#plt.figure(figsize = (12, 10))
#plt.plot(apple['Adj. Close'], label = 'AAPL')
#plt.plot(sma30['Adj. Close Price'], label = 'SMA30')
#plt.plot(sma100['Adj. Close Price'], label = 'SMA100')
#plt.xlabel('Apple adjusted close price')
#plt.ylabel('Adj. Close price history')
#plt.legend(loc='upper left')
#plt.show()

#Create new dataframe to store new variables (sma30, sma100 and apple)
data = pd.DataFrame()

data['APPLE'] = apple['Adj. Close']
data['SMA30'] = sma30['Adj. Close Price']
data['SMA100'] = sma100['Adj. Close Price']
print(data)

#Create logical function to determined BUY/SELL
def buy_sell(data):
    PriceBuy = []
    PriceSell = []
    flag = -1            #Flag tells me when the moving averages crossed each other

    for i in range(len(data)):
        if data['SMA30'][i] > data['SMA100'][i]:
            if flag != 1:
                PriceBuy.append(data['APPLE'][i])
                PriceSell.append(np.nan)
                flag = 1
            else:
                PriceBuy.append(np.nan)
                PriceSell.append(np.nan)
        elif data['SMA30'][i] < data['SMA100'][i]:
            if flag != 0:
                PriceBuy.append(np.nan)
                PriceSell.append(data['APPLE'][i])
                flag = 0
            else:
                PriceBuy.append(np.nan)
                PriceSell.append(np.nan)
        else:
            PriceBuy.append(np.nan)
            PriceSell.append(np.nan)

    return(PriceBuy, PriceSell)

#Store buy and sell data into a variable
buy_sell = buy_sell(data)

data['Buy_Signal'] = buy_sell[0]
data['Sell_Signal'] = buy_sell[1]

#Visualize data and trading strategy
plt.figure(figsize=(12.5, 4.5))

plt.plot(data['APPLE'], label = 'APPLE')
plt.plot(data['SMA30'], label = 'SMA30')
plt.plot(data['SMA100'], label = 'SMA100')
plt.scatter(data.index, data['Buy_Signal'], label='Buy', marker='^', color='green')
plt.scatter(data.index, data['Sell_Signal'], label='Sell', marker='v', color='red')
plt.title('Apple Adj. Close History with Buy/Sell Signals')
plt.ylabel('Apple Adj. Close Price')
plt.legend(loc='upper left')
plt.show()