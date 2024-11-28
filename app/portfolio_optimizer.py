
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sc
from data_processing import portfolioPerformance, fetchData

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

    portfolioReturns, portfolioStd = portfolioPerformance(weights, meanReturns, covMatrix)

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

    # ...
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

    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

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


def portfolioReturns(weights, meanReturns, covMatrix):
    """ Return Portfolio returns. """

    return portfolioPerformance(weights, meanReturns, covMatrix)[0]


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
    Calculate portfolio metrics and the efficient frontier for a given set of assets.

    This function computes:
    - The portfolio with the maximum Sharpe Ratio (highest risk-adjusted return).
    - The portfolio with the minimum volatility (lowest risk).
    - The efficient frontier, which shows the minimum variance portfolio for a range of target returns.

    Parameters:
    - meanReturns (pandas.Series): Expected returns for each asset in the portfolio.
    - covMatrix (numpy.ndarray): Covariance matrix of asset returns.
    - riskFreeRate (float, optional): The risk-free rate (annualized) used in Sharpe Ratio calculations. Default is 0.
    - constraintSet (tuple, optional): Bounds for the portfolio weights. Default is (0, 1), 
      meaning weights are constrained between 0 and 1 (no short selling).

    Returns:
    - tuple: Contains the following outputs:
        - maxSR_returns (float): Annualized return (%) of the maximum Sharpe Ratio portfolio.
        - maxSR_std (float): Annualized standard deviation (%) of the maximum Sharpe Ratio portfolio.
        - maxSR_allocation (pandas.DataFrame): Asset allocations (%) in the maximum Sharpe Ratio portfolio.
        - minVol_returns (float): Annualized return (%) of the minimum volatility portfolio.
        - minVol_std (float): Annualized standard deviation (%) of the minimum volatility portfolio.
        - minVol_allocation (pandas.DataFrame): Asset allocations (%) in the minimum volatility portfolio.
        - efficientList (list): List of portfolio standard deviations (volatilities) along the efficient frontier.

    Notes:
    - The efficient frontier is calculated using 20 evenly spaced target returns between the minimum 
      volatility portfolio return and the maximum Sharpe Ratio portfolio return.
    - Allocations for the maximum Sharpe Ratio and minimum volatility portfolios are rounded and expressed as percentages.
    - This function assumes the covariance matrix and mean returns are pre-calculated based on historical data or simulations.
    """

    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSharpe(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizePortfolioStd(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOptimization(meanReturns, covMatrix, target)['fun'])
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList


def plotEfficientFrontier(meanReturns, covMatrix, riskFreeRate=0):
    """
    Plot the Efficient Frontier for a given portfolio of assets.

    Parameters:
    - meanReturns (numpy.ndarray): Expected returns of assets.
    - covMatrix (numpy.ndarray): Covariance matrix of asset returns.
    - riskFreeRate (float, optional): Risk-free rate (annualized). Default is 0.

    Returns:
    - None: Displays a graph of the Efficient Frontier.
    """

    # Calculate max Sharpe and min volatility portfolios
    maxSR_Portfolio = maxSharpe(meanReturns, covMatrix, riskFreeRate)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)

    minVol_Portfolio = minimizePortfolioStd(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)

    # Generate Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 100)
    for target in targetReturns:
        efficientList.append(efficientOptimization(meanReturns, covMatrix, target)['fun'])

    # Set up plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot Efficient Frontier
    plt.plot(efficientList, targetReturns * 100, label='Efficient Frontier', color='darkblue', linewidth=2)

    # Highlight Maximum Sharpe Ratio Portfolio
    plt.scatter(maxSR_std * 100, maxSR_returns * 100, color='red', marker='*', s=200, label='Max Sharpe Ratio')

    # Highlight Minimum Volatility Portfolio
    plt.scatter(minVol_std * 100, minVol_returns * 100, color='green', marker='o', s=150, label='Min Volatility')

    # Add labels, title, and legend
    plt.title('Efficient Frontier', fontsize=16)
    plt.xlabel('Risk (Standard Deviation) %', fontsize=12)
    plt.ylabel('Return %', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    plt.show()