# Portfolio Optimizer
# 2024.12.20
# Aidan Nachi


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sc
from data_processing import expectedPortfolioPerformance


def negativeSharpe(weights, meanReturns, covMatrix, riskFreeRate=0):
    """
    Calculate the negative of the Sharpe ratio for a given portfolio of assets.

    The Sharpe ratio measures the risk-adjusted return of a portfolio, calculated as the 
    portfolio's excess return over the risk-free rate, divided by its standard deviation.

    Parameters:
    - weights (numpy.ndarray): Array of portfolio weights, representing the proportion of 
      each asset in the portfolio.
    - meanReturns (numpy.ndarray): Array of expected returns for each asset.
    - covMatrix (numpy.ndarray): Covariance matrix of asset returns.
    - riskFreeRate (float, optional): Annualized risk-free rate, representing the return on 
      a theoretically "risk-free" asset (e.g., government bonds). Default is 0.

    Returns:
    - float: The negative of the Sharpe ratio, calculated as:
        - negativeSharpe = - [(Portfolio Return - Risk-Free Rate) / Portfolio Std Dev]
    """

    portfolioReturns, portfolioStd = expectedPortfolioPerformance(weights, meanReturns, covMatrix)

    return - (portfolioReturns - riskFreeRate) / portfolioStd


def maxSharpe(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    """
    Maximize the Sharpe ratio of a portfolio by optimizing its asset allocation.

    This function minimizes the negative Sharpe ratio, effectively maximizing the 
    risk-adjusted return of the portfolio, subject to constraints on weights.

    Parameters:
    - meanReturns (numpy.ndarray): Array of expected returns for each asset.
    - covMatrix (numpy.ndarray): Covariance matrix of asset returns.
    - riskFreeRate (float, optional): Annualized risk-free rate, representing the return on 
      a theoretically "risk-free" asset (e.g., government bonds). Default is 0.
    - constraintSet (tuple, optional): Bounds for the portfolio weights. Default is (0, 1), 
      which means weights must be between 0 and 1 (no short selling allowed).

    Returns:
    - scipy.optimize.OptimizeResult: The optimization result containing:
        - `x` (numpy.ndarray): Optimal portfolio weights that maximize the Sharpe ratio.
        - Additional information about the optimization process, such as success status.
    """

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    result = sc.minimize(negativeSharpe, numAssets*[1./numAssets], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result


def portfolioStd(weights, meanReturns, covMatrix):
    """ Return Portfolio Standard Deviation. """

    return expectedPortfolioPerformance(weights, meanReturns, covMatrix)[1]


def minimizePortfolioStd(meanReturns, covMatrix, constraintSet=(0,1)):
    """
    Minimize the portfolio's standard deviation (risk) by optimizing the allocation of assets.

    This function identifies the portfolio weights that achieve the lowest possible 
    standard deviation while ensuring the weights sum to 1 and remain within specified bounds.

    Parameters:
    - meanReturns (numpy.ndarray): Array of expected returns for each asset.
    - covMatrix (numpy.ndarray): Covariance matrix of asset returns.
    - constraintSet (tuple, optional): Bounds for the portfolio weights. Default is (0, 1), 
      which means weights must be between 0 and 1 (no short selling allowed).

    Returns:
    - scipy.optimize.OptimizeResult: The optimization result containing:
        - `x` (numpy.ndarray): Optimal portfolio weights that minimize the standard deviation.
        - Additional information about the optimization process, such as success status.
    """

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    result = sc.minimize(portfolioStd, numAssets*[1./numAssets], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result


def portfolioReturns(weights, meanReturns, covMatrix):
    """ Return Portfolio returns. """

    return expectedPortfolioPerformance(weights, meanReturns, covMatrix)[0]


def negativeReturns(weights, meanReturns):
    """
    Calculate the negative return of a portfolio for a given set of weights and each
    asset's expected annualized return.
    """

    return - np.dot(weights, meanReturns)


def maxReturn(meanReturns, constraintSet=(0,1)):
    """
    Optimize the portfolio for a maximum return subject to constraints.
    """

    numAssets = len(meanReturns)
    args = (meanReturns)

    # Constraints: weights sum up to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds for the weights
    bounds = tuple(constraintSet for _ in range(numAssets))

    # Maximize return (minimize negative return).
    result = sc.minimize(negativeReturns, numAssets*[1./numAssets], args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result


def efficientOptimization(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    """
    Optimize the portfolio to achieve the minimum variance for a given target return.

    This function finds the portfolio weights that minimize portfolio risk (standard deviation)
    while ensuring the portfolio achieves a specified target return and adheres to the constraints.

    Parameters:
    - meanReturns (numpy.ndarray): Array of expected returns for each asset.
    - covMatrix (numpy.ndarray): Covariance matrix of asset returns.
    - returnTarget (float): Desired portfolio return to be achieved.
    - constraintSet (tuple, optional): Bounds for the portfolio weights. Default is (0, 1),
      which means weights must be between 0 and 1 (no short selling allowed).

    Returns:
    - scipy.optimize.OptimizeResult: The optimization result containing:
        - `x` (numpy.ndarray): Optimal portfolio weights that minimize the variance for the given return.
        - Additional information about the optimization process, such as success status.
    """

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)

    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturns(x, meanReturns, covMatrix) - returnTarget},
                   {'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    
    bounds = tuple(bound for assets in range(numAssets))
    effOptimized = sc.minimize(portfolioStd, numAssets*[1./numAssets], args=args,
                               method = 'SLSQP', bounds=bounds, constraints=constraints)
    
    return effOptimized



def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0, 1)):
    """
    Compute key portfolio metrics and the efficient frontier for a given set of assets.

    This function determines:
    - The portfolio with the maximum Sharpe Ratio (highest risk-adjusted return).
    - The portfolio with the minimum volatility (lowest risk).
    - The efficient frontier, representing the portfolio with the minimum variance 
      for a series of target returns.

    Parameters:
    - meanReturns (pandas.Series): Expected annualized returns for each asset in the portfolio.
    - covMatrix (numpy.ndarray): Covariance matrix of the asset returns.
    - riskFreeRate (float, optional): Annualized risk-free rate for Sharpe Ratio calculations. Default is 0.
    - constraintSet (tuple, optional): Bounds for portfolio weights. Default is (0, 1), 
      meaning weights are constrained to a range of 0 to 1 (no short selling).

    Returns:
    - tuple: Contains the following results:
        - maxSR_returns (float): Annualized return (%) of the maximum Sharpe Ratio portfolio.
        - maxSR_std (float): Annualized standard deviation (%) of the maximum Sharpe Ratio portfolio.
        - maxSR_allocation (pandas.DataFrame): Asset allocations (%) for the maximum Sharpe Ratio portfolio.
        - minVol_returns (float): Annualized return (%) of the minimum volatility portfolio.
        - minVol_std (float): Annualized standard deviation (%) of the minimum volatility portfolio.
        - minVol_allocation (pandas.DataFrame): Asset allocations (%) for the minimum volatility portfolio.
        - efficientList (list): Portfolio volatilities (standard deviations) along the efficient frontier.
        - targetReturns (numpy.ndarray): Target returns used to compute the efficient frontier.

    Notes:
    - The efficient frontier is calculated using 20 evenly spaced target returns 
      between the returns of the minimum volatility portfolio and the maximum Sharpe Ratio portfolio.
    - Asset allocations for the maximum Sharpe Ratio and minimum volatility portfolios are expressed as percentages and rounded for clarity.
    - The function assumes that mean returns and the covariance matrix are derived from historical data or simulated values.
    """

    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSharpe(meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxSR_returns, maxSR_std = expectedPortfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['Allocation'])
    maxSR_allocation['Allocation'] = [round(i*100,0) for i in maxSR_allocation['Allocation']]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizePortfolioStd(meanReturns, covMatrix, constraintSet)
    minVol_returns, minVol_std = expectedPortfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['Allocation'])
    minVol_allocation['Allocation'] = [round(i*100,0) for i in minVol_allocation['Allocation']]

    # Max Return Portfolio
    maxReturn_Portfolio = maxReturn(meanReturns, constraintSet)
    maxReturn_returns, maxReturn_std = expectedPortfolioPerformance(maxReturn_Portfolio['x'], meanReturns, covMatrix)
    maxReturn_allocation = pd.DataFrame(maxReturn_Portfolio['x'], index=meanReturns.index, columns=['Allocation'])
    maxReturn_allocation['Allocation']  = [round(i * 100, 0) for i in maxReturn_allocation['Allocation']]

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxReturn_returns, 200)
    for target in targetReturns:
        efficientList.append(efficientOptimization(meanReturns, covMatrix, target)['fun'])

    # Scale to real sizes
    efficientList = [round(ef_std * 100, 2) for ef_std in efficientList]
    targetReturns = [round(target * 100, 2) for target in targetReturns] 

    # Convert to percentages
    maxSR_returns, maxSR_std = round(maxSR_returns * 100, 2), round(maxSR_std * 100, 2)
    minVol_returns, minVol_std = round(minVol_returns * 100, 2), round(minVol_std * 100, 2)
    maxReturn_returns, maxReturn_std = round(maxReturn_returns * 100, 2), round(maxReturn_std * 100, 2)

    return (
        maxSR_returns, maxSR_std, maxSR_allocation,
        minVol_returns, minVol_std, minVol_allocation,
        maxReturn_returns, maxReturn_std, maxReturn_allocation,
        efficientList, targetReturns
    )


def get_optimized_data(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0, 1)):
    """ Get the minimum volitility and maximum sharpe ratio asset allocations and details. """

    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSharpe(meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxSR_returns, maxSR_std = expectedPortfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['Allocation'])
    maxSR_allocation.index.name = 'Tickers'
    maxSR_allocation['Allocation'] = [f"{round(i*100, 2)}%" for i in maxSR_allocation['Allocation']]
                                   
    # Min Volatility Portfolio
    minVol_Portfolio = minimizePortfolioStd(meanReturns, covMatrix, constraintSet)
    minVol_returns, minVol_std = expectedPortfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['Allocation'])
    minVol_allocation.index.name = 'Tickers'
    minVol_allocation['Allocation'] = [f"{round(i*100, 2)}%" for i in minVol_allocation['Allocation']]

    # Max Return Portfolio
    maxReturn_Portfolio = maxReturn(meanReturns, constraintSet)
    maxReturn_returns, maxReturn_std = expectedPortfolioPerformance(maxReturn_Portfolio['x'], meanReturns, covMatrix)
    maxReturn_allocation = pd.DataFrame(maxReturn_Portfolio['x'], index=meanReturns.index, columns=['Allocation'])
    maxReturn_allocation['Allocation'] = [f"{round(i*100, 2)}%" for i in maxReturn_allocation['Allocation']]

    # Turn numbers to percentages
    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    maxReturn_returns, maxReturn_std = round(maxReturn_returns*100,2), round(maxReturn_std*100,2)

    maxSharpeRatio = round(maxSR_returns / maxSR_std, 2)
    minVolSharpe = round(minVol_returns / minVol_std, 2)
    maxReturnSharpe = round(maxReturn_returns / maxReturn_std, 2)
    
    results =  (maxSharpeRatio, maxSR_returns, maxSR_std, 
                maxSR_allocation, minVolSharpe, minVol_returns, minVol_std, 
                minVol_allocation, maxReturnSharpe, maxReturn_returns, 
                maxReturn_std, maxReturn_allocation)

    return results


def plotEfficientFrontier(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """ Return a graph plotting the min vol, max sr and efficient frontier. """

    """
    Plot the efficient frontier from Min Vol to Max Return portfolio.
    """
    maxSR_returns, maxSR_std, maxSR_allocation, \
    minVol_returns, minVol_std, minVol_allocation, \
    maxReturn_returns, maxReturn_std, maxReturn_allocation, \
    efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(efficientList, targetReturns, linestyle='--', color='blue', label='Efficient Frontier')

    # Plot Min Vol, Max SR, and Max Return portfolios
    ax.scatter(minVol_std, minVol_returns, color='red', label='Minimum Volatility Portfolio', s=100, edgecolors='black')
    ax.text(minVol_std + 0.005, minVol_returns, 'Min Vol', fontsize=10)

    ax.scatter(maxSR_std, maxSR_returns, color='green', label='Maximum Sharpe Ratio Portfolio', s=100, edgecolors='black')
    ax.text(maxSR_std + 0.005, maxSR_returns, 'Max SR', fontsize=10)

    ax.scatter(maxReturn_std, maxReturn_returns, color='orange', label='Maximum Return Portfolio', s=100, edgecolors='black')
    ax.text(maxReturn_std + 0.005, maxReturn_returns, 'Max Return', fontsize=10)

    # Add labels, legend, and title
    ax.set_xlabel('Volatility (Standard Deviation)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig