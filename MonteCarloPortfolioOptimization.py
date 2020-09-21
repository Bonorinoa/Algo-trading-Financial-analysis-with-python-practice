#Monte Carlo simulation appreach to portfolio optimization


import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

import scipy.optimize as sco
from scipy import stats

#Select assets
tickers = ['TSLA', 'APHA', 'QQQ', 'AAPL', 'DAL', 'WFC', 'QCOM', 'BAM', 'GOLD']

#Sepecify start and end date
start = datetime(2010, 1, 1)
todayDate = datetime.today().strftime('%Y-%m-%d')

#Create dataframe with asset data
df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, todayDate)['Adj Close'] for ticker in tickers]).T
df.columns = tickers

#Function to calculate the following: 
#annualised return, annualised standard deviation and annualised Sharpe ratio of a portfolio
#Parameters provided are:
#weights, mean returns over historic data, covariance matrix and risk free interest rate

def calc_portfolio_perf(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio

#Creates multiple randomly weighted portfolios and passes them to the above function
#Basically apply monte carlo simulation brute force
def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf):
    results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random_sample(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
        results_matrix[0,i] = portfolio_return
        results_matrix[1,i] = portfolio_std
        results_matrix[2,i] = sharpe_ratio
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j+3,i] = weights[j]
            
    results_df = pd.DataFrame(results_matrix.T,columns=['ret','stdev','sharpe'] + [ticker for ticker in tickers])
        
    return results_df

#Set mean returns by calculating daily average percentage change
mean_returns = df.pct_change().mean()
#Set covariance matrix
cov = df.pct_change().cov()
#Set number of portfolios to simulate
num_portfolios = 100000
#Set risk free interest rate (Treasury bill interest rate is often used for this value)
rf = 0.11

#Store the results on a variable
results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf)

#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]

#locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

#create scatter plot coloured by Sharpe Ratio
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.colorbar()

#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500)

#plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=500)
#plt.show()

print(max_sharpe_port.to_frame().T)

print(min_vol_port.to_frame().T)
