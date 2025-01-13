# Risk
# 2025.1.12
# Aidan Nachi

# Import Libraries
import numpy as np
import pandas as pd


class Risk:
    def __init__(self, dailyReturns, weights):
        self.dailyReturns = dailyReturns
        self.weights = weights


    def getPortfolioReturns(self):
        """ Get daily portfolio returns. """

        # Prepare allocation (weights) and daily returns dataframes.
        weightsCleaned = self.weights.loc[self.dailyReturns.columns, 'Allocation']
        weightsCleaned = weightsCleaned.str.rstrip('%').astype(float) / 100
        returns = self.dailyReturns.copy()
        returns["portfolio"] = returns.dot(weightsCleaned)

        return returns["portfolio"]
    

    def historicalVaR(self, returns=None, alpha=5):
        """
        Read in a pandas dataframe or series of returns and output the
        percentile of the distribution at the given confidence level.
        """

        returns = self.getPortfolioReturns()

        if isinstance(returns, pd.Series):
            return np.percentile(returns, alpha)
        
        # A passed user-defined function will be passed a Series for evaluation.
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(self.historicalVaR, alpha)
        

    def historicalCVaR(self, returns=None, alpha=5):
        """
        Read in a pandas dataframe or series of returns and output the
        CVaR for the CVaR for the returns.
        """

        returns = self.getPortfolioReturns()

        if isinstance(returns, pd.Series):
            belowVaR = returns <= self.historicalVaR(returns, alpha)
            return returns[belowVaR].mean() 
        
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(self.historicalCVaR, alpha)
        
    def getVarCvar(self):
        """ 
        Return a pandas dataframe of portfolio VaR and CVaR for confidence levels
        90%, 95%, and 99%
        """

        # Alphas corresponding to 90%, 95%, and 99% confidence levels.
        alphas = [10, 5, 1]
        results = {'Confidence Level': [], 'VaR': [], 'CVaR': []}

        for a in alphas:
            var = round(-self.historicalVaR(alpha=a) * 100, 2)
            cvar = round(-self.historicalCVaR(alpha=a) * 100, 2)
            results['Confidence Level'].append(f"{100 - a}%")
            results['VaR'].append(var)
            results['CVaR'].append(cvar)

        # Convert the dictionary into a DataFrame
        results_df = pd.DataFrame(results)

        return results_df
