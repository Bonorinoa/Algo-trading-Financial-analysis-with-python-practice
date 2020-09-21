#Sharpe ratio maximization approach + Value at Risk (VaR) minimization approach for portfolio optimization


import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

import scipy.optimize as sco
from scipy import stats

#Select assets
tickers = ['TSLA', 'APHA', 'AAPL', 'ROKU','MDB']

#Sepecify start and end date
start = datetime(2010, 1, 1)
todayDate = datetime.today().strftime('%Y-%m-%d')

#Create dataframe with asset data
df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, todayDate)['Adj Close'] for ticker in tickers]).T
df.columns = tickers

#Note that sharpe ratio is negative because we want to maximize it and optimization functions minimize
#And SciPy only offers a minimize function. Easy fix by just putting the minus sign in front
def calc_neg_sharpe(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return -sharpe_ratio

#We define the following constraints because we chose the SLSQP function
#Stand for Sequential Least-Square Programming
#it takes parameters in a dictionary format.
#Type can be 'eq' or 'ineq' which stands for equality and inequality
#'eq' means we are looking for our function to equate to zero
#Fun refers to the function defininf the constraint
#which in our case is that the sume of the weights must be equal to 1 because we want to eqaute to zero
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

#Bounds are just to specify the the stock weights must be between 0 and 1
def max_sharpe_ratio(mean_returns, cov, rf):
    num_assets = len(mean_returns)
    args = (mean_returns, cov, rf)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    #We want weights to be between 0 and 1
    bound = (0.0,1.0)
    #Apply bound to each stock/asset
    bounds = tuple(bound for asset in range(num_assets))
    #Recall sco is the SciPy optimizer function
    result = sco.minimize(calc_neg_sharpe, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

#Set mean returns by calculating daily average percentage change
mean_returns = df.pct_change().mean()
#Set covariance matrix
cov = df.pct_change().cov()
#Set number of portfolios to simulate
num_portfolios = 100000
#Set risk free interest rate (Treasury bill interest rate is often used for this value)
rf = 0.11

optimal_port_sharpe = max_sharpe_ratio(mean_returns, cov, rf)

print(pd.DataFrame([round(x,5) for x in optimal_port_sharpe['x']],index=tickers).T)

#Simple calculation of portfolio standard deviation
def calc_portfolio_std(weights, mean_returns, cov):
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    return portfolio_std

#Same function as our max_sharpe_ratio but without risk-free rate to identify minimum variance portfolio
#We just change the name of the function 
def min_variance(mean_returns, cov):
    num_assets = len(mean_returns)
    args = (mean_returns, cov)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_portfolio_std, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

min_port_variance = min_variance(mean_returns, cov)

#print(pd.DataFrame([round(x,4) for x in min_port_variance['x']],index=tickers).T)

