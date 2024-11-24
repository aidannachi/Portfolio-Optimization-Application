import streamlit as st
from data_processing import fetchData, portfolioPerformance, tickerSymbols
import datetime as dt 
import numpy as np


st.markdown("<h1 style='text-align: center;'>Portfolio Optimization</h1>", unsafe_allow_html=True)

tickers = st.multiselect('Select assets for your portfolio', tickerSymbols(), default=['AAPL', 'MSFT', 'TSLA'])

st.subheader("Portfolio Data")
if len(tickers) != 0:
    returns, std = portfolioPerformance(tickers)
    st.write(f"Portfolio Return: {returns}%")
    st.write(f"Porfolio Standard Deviation: {std}%")
else:
    st.write("Awaiting asset selection...")