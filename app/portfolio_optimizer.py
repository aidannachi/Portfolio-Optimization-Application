
# Import libraries
import numpy as np
import pandas as pd
import scipy.optimize as sc
from data_processing import portfolioPerformance, fetchData

def negativeSharpe(tickers, riskFreeRate=0):
    """
    Calculate the negative of the Sharpe ratio for a given portfolio of assets.

    The Sharpe ratio measures the risk-adjusted return of a portfolio, calculated as the 
    portfolio's excess return over the risk-free rate, divided by its standard deviation.

    Parameters:
    - tickers (list): A list of stock ticker symbols (strings) representing the assets in 
      the portfolio. These tickers are used to fetch the stock data and compute returns and 
      standard deviation.
    - riskFreeRate (float, optional): The risk-free rate (annualized), which represents 
      the return on a theoretically "risk-free" asset, such as government bonds. Default 
      is 0 (no risk-free rate is considered).

    Returns:
    - float: The negative of the Sharpe ratio, calculated as:
        - negativeSharpe = - [(Portfolio Return - Risk-Free Rate) / Portfolio Std Dev]
    """

    portfolioReturns, portfolioStd = portfolioPerformance(tickers)

    return - (portfolioReturns- riskFreeRate) / portfolioStd


def maxSharpe(tickers, riskFreeRate=0, constraintSet=(0,1)):
    """ 
    Minimize the negative Sharpe Ratio (which is actually maximizing the Sharpe Ratio),
    by altering the weights of the portfolio.
    """

    meanReturns, covMatrix = fetchData(tickers)

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    result = sc.minimize(negativeSharpe, tickers, numAssets*[1./numAssets], args=args,
                                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

def portfolioStd(tickers):
    """ Return Portfolio Standard Deviation. """

    return portfolioPerformance(tickers)[1]

def minimizePortfolioStd(tickers, riskFreeRate=0, constraintSet=(0,1)):
    """ 
    Minimize the portfolio variance by altering the weights// allocation
    of assets in the portfolio.
    """

    meanReturns, covMatrix = fetchData(tickers)

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    result = sc.minimize(portfolioStd, tickers, numAssets*[1./numAssets], args=args,
                                         method='SLSQP', bounds=bounds, constraints=constraints)


