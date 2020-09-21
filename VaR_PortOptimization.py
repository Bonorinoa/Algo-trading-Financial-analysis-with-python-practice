#Identifying the portfolio weights that minimise the Value at Risk (VaR).
#The logic is very similar to the one followed with a Monte Carlo simulation approach
#returns the parametric portfolio VaR to a confidence level determined by the value of the “alpha” argument
#(confidence level will be 1 – alpha), and to a time scale determined by the “days” argument.

import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

import scipy.optimize as sco
from scipy import stats

#Select assets
tickers = ['TSLA', 'APHA', 'QQQ', 'AAPL', 'DAL', 'WFC', 'QCOM', 'BAM', 'ROKU']

#Sepecify start and end date
start = datetime(2010, 1, 1)
todayDate = datetime.today().strftime('%Y-%m-%d')

#Create dataframe with asset data
df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, todayDate)['Adj Close'] for ticker in tickers]).T
df.columns = tickers

def calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha, days):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_return, portfolio_std, portfolio_var
    

def simulate_random_portfolios_VaR(num_portfolios, mean_returns, cov, alpha, days):
    results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random_sample(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, portfolio_VaR = calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha, days)
        results_matrix[0,i] = portfolio_return
        results_matrix[1,i] = portfolio_std
        results_matrix[2,i] = portfolio_VaR
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j+3,i] = weights[j]
            
    results_df = pd.DataFrame(results_matrix.T,columns=['ret','stdev','VaR'] + [ticker for ticker in tickers])
        
    return results_df

mean_returns = df.pct_change().mean()
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.11
days = 252
alpha = 0.05
results_frame = simulate_random_portfolios_VaR(num_portfolios, mean_returns, cov, alpha, days)

#This time we plot the results of each portfolio with annualised return remaining on the y-axis
#but the x-axis this time representing the portfolio VaR (rather than standard deviation). 
#The plot colours the data points according to the value of VaR for that portfolio.

#locate positon of portfolio with minimum VaR
min_VaR_port = results_frame.iloc[results_frame['VaR'].idxmin()]
#create scatter plot coloured by VaR
plt.subplots(figsize=(10,8))
plt.scatter(results_frame.VaR,results_frame.ret,c=results_frame.VaR,cmap='RdYlBu')
plt.xlabel('Value at Risk')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of minimum VaR portfolio
plt.scatter(min_VaR_port[2],min_VaR_port[0],marker=(5,1,0),color='r',s=500)

#plt.show()

print(min_VaR_port.to_frame().T)

#Value at Risk (VaR) calculation goes as follos:
# VaR = Rp - (z * SDp)
# Rp = portfolio return
# z = critical t
# Sdp = portfolio standard deviation 

#Sharpe ratio calculation goes as follow:
#Sharpe ratio = (Rp - Rf) / SDp
# Rp = portfolio return
# Sdp = portfolio standard deviation
# Rf = Risk free rate (Treasury bill interest rate was used as reference)

#From this we can see that VaR decreases when portfolio returns increase and vice versa
#Whereas the Sharpe ratio increases as portfolio returns increase
#Therefore, what minimises VaR in terms of returns actually maximises the Sharpe ratio.

#Last but not least we introduce a second approach to calculating minimum VaR portfolio
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
def calc_portfolio_VaR(weights, mean_returns, cov, alpha, days):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_var
def min_VaR(mean_returns, cov, alpha, days):
    num_assets = len(mean_returns)
    args = (mean_returns, cov, alpha, days)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_portfolio_VaR, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

min_port_VaR = min_VaR(mean_returns, cov, alpha, days)

print(pd.DataFrame([round(x,2) for x in min_port_VaR['x']],index=tickers).T)