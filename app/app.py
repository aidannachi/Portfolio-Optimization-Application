import streamlit as st
from data_processing import fetchData, tickerSymbols, portfolioPerformance
from portfolio_optimizer import calculatedResults, plotEfficientFrontier
import datetime as dt 
import numpy as np


# Application Title
st.markdown("<h1 style='text-align: center;'>Portfolio Optimization</h1>", unsafe_allow_html=True)

# Allow users to select tickers.
tickers = st.multiselect('Select assets for your portfolio', tickerSymbols(), default=['AAPL', 'MSFT', 'TSLA'])

# Display portfolio data 
st.subheader("Portfolio Data")

if len(tickers) != 0:
    # Grab data for selected assets
    meanReturns, covMatrix = fetchData(tickers)

    # Get data for equally weighted portfolio.
    num_tickers = len(tickers)
    weights = np.array([1/num_tickers] * num_tickers)
    returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

    st.write(f"Portfolio Return: {round(returns * 100, 2)}%")
    st.write(f"Porfolio Standard Deviation: {round(std * 100, 2)}%")

    # Optimize the portfolio.
    graph = plotEfficientFrontier(meanReturns, covMatrix)
    st.pyplot(graph)

else:
    st.write("Awaiting asset selection...")