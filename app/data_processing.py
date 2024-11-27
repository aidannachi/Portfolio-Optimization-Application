# Data Processing
# 2024.11.23
# Aidan Nachi

# Import libraries
import numpy as  np
import pandas as pd
import datetime as dt
import yfinance as yf


def tickerSymbols():
    """ Return a list of all the ticker symbols listed on Yahoo Finance. """

    # URL for the list of S&P 500 companies on Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Read the table directly from the Wikipedia page
    tables = pd.read_html(url)

    # The first table contains the tickers
    tickers_df = tables[0]
    tickers = tickers_df['Symbol'].tolist()  # Get the tickers

    return tickers


def fetchData(tickers):
    """
    Fetch closing prices for a list of stock tickers within a specified date range,
    and compute the mean returns and covariance matrix of the returns.

    Parameters:
    - tickers (list): List of tickers symbols.

    Returns:
    - meanReturns (pd.Series): Mean daily returns for each stock.
    - covMatrix (pd.DataFrame): Covariance matrix of the returns.
    """

    # Get end and start dates that go from today to a year back.
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365)    
    
    # Download data for all stocks
    stockData = yf.download(tickers, start=startDate, end=endDate)['Close']

    # Calculate daily percentage returns
    returns = stockData.pct_change().dropna()

    # Compute mean returns for each asset and construct a covariance matrix
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    return meanReturns, covMatrix


def portfolioPerformance(tickers):
    """
    Calculate the annualized return and standard deviation (risk) of a portfolio.

    Parameters:
    - tickers (list): List of stock ticker symbols representing the portfolio's assets.

    Returns:
    - returns (float): Annualized portfolio return as a percentage.
    - std (float): Annualized portfolio standard deviation (risk) as a percentage.
    """

    # Set all weights equally (1/n for n assets).
    weights = np.array([(1 / len(tickers))] * len(tickers))

    # Get the mean and covariance matrix for the portfolio
    meanReturns, covMatrix = fetchData(tickers)

    # Calculate the portfolio return and std as a percentage rounded to 2 decimal place.
    returns = round( (np.sum(meanReturns * weights) * 252) * 100, 2)
    std = round( (np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)) * 100, 2)

    return returns, std
    
