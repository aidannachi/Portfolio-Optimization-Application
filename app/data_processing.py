# Data Processing
# 2024.11.23
# Aidan Nachi

# Import libraries
import numpy as  np
import pandas as pd
import datetime as dt
import yfinance as yf

class dataProcessing:
    def __init__():
        """
        Initialize the dataProcessing class. No specific initialization needed.
        """
        pass


    @staticmethod
    def tickerSymbols():
        """ Return a list of all the ticker symbols listed on Yahoo Finance. """

        # URL for the list of S&P 500 companies on Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

        # Read the table directly from the Wikipedia page
        table = pd.read_html(url)

        # The first table contains the tickers
        tickers_df = table[0]
        tickers = tickers_df['Symbol'].tolist()  # Get the tickers

        return tickers

    
    @staticmethod
    def fetchData(tickers, startDate, endDate):
        """
        Fetch closing prices for a list of stock tickers within a specified date range,
        and compute the mean returns and covariance matrix of the returns.

        Parameters:
        - tickers (list): List of tickers symbols.
        - startDate (str or None): Start date for data fetching (YYYY-MM-DD format or None for default).
        - endDate (str or None): End date for data fetching (YYYY-MM-DD format or None for default).

        Returns:
        - meanReturns (pd.Series): Mean daily returns for each stock.
        - covMatrix (pd.DataFrame): Covariance matrix of the returns.
        """
   
        # Download data for all stocks
        stockData = yf.download(tickers, start=startDate, end=endDate)['Adj Close']

        # Calculate daily percentage returns
        daily_returns = stockData.pct_change().dropna()

        # Get annual volitility and returns.
        annual_returns = daily_returns.mean() * 252 
        annual_volatility = daily_returns.std() * np.sqrt(252)

        annual_returns = round(annual_returns * 100, 2).map("{:.2f}%".format)
        annual_volatility = round(annual_volatility * 100, 2).map("{:.2f}%".format)

        # Compute mean returns for each asset and construct a covariance matrix
        meanReturns = daily_returns.mean()
        covMatrix = daily_returns.cov()

        # Create a DataFrame
        assetResults = pd.DataFrame({
            'Ticker': tickers,
            'Expected Return': annual_returns,
            'Volitility (Standard Deviation)': annual_volatility
        }, index=tickers)

        return assetResults, meanReturns, covMatrix


    @staticmethod
    def expectedPortfolioPerformance(weights, meanReturns, covMatrix):
        """
        Computes the annualized return and standard deviation (risk) of a portfolio.

        Parameters:
        - weights (numpy.ndarray): Array of portfolio weights for each asset. The sum of weights should be 1.
        - meanReturns (numpy.ndarray): Array of expected daily returns for each asset.
        - covMatrix (numpy.ndarray): Covariance matrix of daily returns for the assets.

        Returns:
        - tuple:
            - returns (float): Annualized portfolio return as a percentage, rounded to two decimal places.
            - std (float): Annualized portfolio standard deviation (risk) as a percentage, rounded to two decimal places.
        """

        # Calculate the portfolio return and std as a percentage rounded to 2 decimal place.
        expectedReturns = (np.sum(meanReturns * weights) * 252) 
        std =  (np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)) 

        return expectedReturns, std


    @staticmethod 
    def assetCorrelations(tickers, startDate=None, endDate=None):
        """
        Return a correlation matrix for the assets.

        Parameters:
        - tickers (list): List of ticker symbols.
        - startDate (str or None): Start date for data fetching (YYYY-MM-DD format or None for default).
        - endDate (str or None): End date for data fetching (YYYY-MM-DD format or None for default).

        Returns:
        - correlation_matrix (pd.DataFrame): Correlation matrix of the assets.
        """

        # Set default date range if not provided
        if not endDate:
            endDate = dt.datetime.now()
        else:
            endDate = pd.to_datetime(endDate)

        if not startDate:
            startDate = endDate - dt.timedelta(days=365)
        else:
            startDate = pd.to_datetime(startDate)

        # Get the adjusted closing prices for each ticker.
        asset_data = yf.download(tickers, start=startDate, end=endDate)
        adj_close = asset_data['Adj Close']

        # Calculate the daily returns for each asset and use them to build a correlation matrix.
        daily_returns = adj_close.pct_change().dropna()
        correlation_matrix = daily_returns.corr()

        return correlation_matrix



