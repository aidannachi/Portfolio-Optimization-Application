# App
# 2024.11.23
# Aidan Nachi


import streamlit as st
from data_processing import fetchData, tickerSymbols, portfolioPerformance
from portfolio_optimizer import calculatedResults, plotEfficientFrontier, get_optimized_data
import datetime as dt 
import numpy as np
import plotly.express as px
import streamlit_shadcn_ui as ui


# Application Title
st.markdown("<h1 style='text-align: center;'>Portfolio Optimization</h1>", unsafe_allow_html=True)

# User input section.
paramsCont = st.container(border=True)
with paramsCont:
    
    # Allow users to select tickers.
    tickers = st.multiselect('Select assets for your portfolio', tickerSymbols(), default=['AAPL', 'MSFT', 'TSLA'])

    # Allow user to select the timeframe.
    dateCol1, dateCol2 = st.columns(2)
    with dateCol1:
        startDate = st.date_input("Start Date", dt.date.today() - dt.timedelta(days=365))
    with dateCol2:
        endDate = st.date_input("End Date", dt.date.today())

    # Allow users to select constraints.
    assetConstraintsOption = st.selectbox("Asset Constraints", ["Yes", "No"], index=1)
    if assetConstraintsOption == "Yes":
        minAssetAllocationCol, maxAssetAllocationCol = st.columns(2)
        with minAssetAllocationCol:
            minAllocation = st.number_input("Min Weight", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")

        with maxAssetAllocationCol:
            maxAllocation = st.number_input("Max Weight", min_value=0.0, max_value=1.0, value=1.0, format="%.2f")
    elif assetConstraintsOption=="No":
        minAllocation = 0
        maxAllocation = 1

    # Allow users to select the optimization objective.
    opObjCol, riskFreeRateCol = st.columns(2)
    with opObjCol:
        optimizationObjective = st.selectbox("Optimization Objective", ["Maximize Sharpe Ratio", "Minimize Volatility", "Maximize Return"])
    with riskFreeRateCol:
        riskFreeRate = st.number_input("Risk-Free-Rate (%)", min_value=0.0, max_value=100.0, value=0.0, format="%.2f")
        riskFreeRate /= 100

    # Have user calculate based on their inputs.
    calc = st.button("Calculate")
    

# Display portfolio data 
if calc:
    if len(tickers) == 0:
        st.error("Select one or more asset for your portfolio")
    else:
        st.markdown("### Portfolio Results")

        # Grab data for selected assets
        meanReturns, covMatrix = fetchData(tickers, startDate, endDate)

        # Get data for equally weighted portfolio.
        num_tickers = len(tickers)
        weights = np.array([1/num_tickers] * num_tickers)
        returns, std = portfolioPerformance(weights, meanReturns, covMatrix)
        equalSharpe = round(returns / std, 2)

        # Display data for equally weighted portfolio
        equalWeightedCont = st.container(border=True)
        with equalWeightedCont:
            st.markdown("##### Equally Weighted Portfolio")
            st.markdown(f"**Expected Return:** {round(returns*100, 2)}%")
            st.markdown(f"**Volatility:** {round(std*100, 2)}%")
            st.markdown(f"**Sharpe Ratio:** {equalSharpe}")

        # Get data for maxSR and minVol portfolios.
        maxSharpeRatio, maxSR_returns, maxSR_std, \
        maxSR_allocation, minVolSharpe, minVol_returns, minVol_std, \
        minVol_allocation, maxReturnSharpe, maxReturn_returns, \
        maxReturn_std, maxReturn_allocation = get_optimized_data(meanReturns, covMatrix, riskFreeRate, constraintSet=(minAllocation, maxAllocation))

        # Clean up tables
        maxSR_allocation["Assets"] = maxSR_allocation.index
        maxSR_allocation = maxSR_allocation[["Assets", "Allocation"]]

        minVol_allocation["Assets"] = minVol_allocation.index
        minVol_allocation = minVol_allocation[["Assets", "Allocation"]]

        maxReturn_allocation["Assets"] = maxReturn_allocation.index
        maxReturn_allocation = maxReturn_allocation[["Assets", "Allocation"]]

        st.markdown("### Optimized Portfolio Results")

        # Display data for max sharpe portfolio
        if optimizationObjective == "Maximize Sharpe Ratio":
            maxSharpeCont = st.container(border=True)
            with maxSharpeCont:
                st.markdown("##### Max Sharpe Ratio Portfolio")
                st.markdown(f"**Expected Return:** {maxSR_returns}%")
                st.markdown(f"**Volatility:** {maxSR_std}%")
                st.markdown(f"**Sharpe Ratio:** {maxSharpeRatio}")

                st.markdown("##### Asset Allocation")
                maxSharpeTable, maxSharpePie = st.columns(2)
                with maxSharpeTable:
                    ui.table(maxSR_allocation)

                with maxSharpePie:
                    # Ensure Allocation values are numeric and non zero.
                    maxSR_allocation['Numeric Allocation'] = maxSR_allocation['Allocation'].str.rstrip('%').astype(float)
                    maxSR_allocation_pie = maxSR_allocation[maxSR_allocation['Numeric Allocation'] != 0]

                    fig = px.pie(
                        maxSR_allocation_pie, 
                        names='Assets', 
                        values='Numeric Allocation', 
                    )
                    fig.update_layout(width=180, height=200, margin=dict(t=20, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)

        # Display data for min vol portfolio
        elif optimizationObjective == "Minimize Volatility":
            minVolCont = st.container(border=True)
            with minVolCont:
                st.markdown("##### Min Volitility Portfolio")
                st.markdown(f"**Expected Return:** {minVol_returns}%")
                st.markdown(f"**Volatility:** {minVol_std}%")
                st.markdown(f"**Sharpe Ratio:** {minVolSharpe}")

                st.markdown("##### Asset Allocation")
                minVolTable, minVolPie = st.columns(2)
                with minVolTable:
                    ui.table(minVol_allocation)

                with minVolPie:
                    # Ensure Allocation values are numeric and non zero.
                    minVol_allocation['Numeric Allocation'] = minVol_allocation['Allocation'].str.rstrip('%').astype(float)
                    minVol_allocation_pie = minVol_allocation[minVol_allocation['Numeric Allocation'] != 0]

                    fig = px.pie(
                        minVol_allocation_pie, 
                        names='Assets', 
                        values='Numeric Allocation', 
                    )
                    fig.update_layout(width=180, height=200, margin=dict(t=20, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)


        # Display data for max return portfolio.
        elif optimizationObjective == "Maximize Return":
            maxReturnsCont = st.container(border=True)
            with maxReturnsCont:
                st.markdown("##### Max Returns Portfolio")
                st.markdown(f"**Expected Return:** {maxReturn_returns}%")
                st.markdown(f"**Volatility:** {maxReturn_std}%")
                st.markdown(f"**Sharpe Ratio:** {maxReturnSharpe}")

                st.markdown("##### Asset Allocation")
                maxReturnTable, maxReturnPie = st.columns(2)
                with maxReturnTable:
                    ui.table(maxReturn_allocation)

                with maxReturnPie:
                    # Ensure Allocation values are numeric and non zero.
                    maxReturn_allocation['Numeric Allocation'] = maxReturn_allocation['Allocation'].str.rstrip('%').astype(float)
                    maxReturn_allocation_pie = maxReturn_allocation[maxReturn_allocation['Numeric Allocation'] != 0]

                    fig = px.pie(
                        maxReturn_allocation_pie, 
                        names='Assets', 
                        values='Numeric Allocation', 
                    )
                    fig.update_layout(width=180, height=200, margin=dict(t=20, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)

                
        # Draw Efficient Frontier with minVol and maxSharpe
        graph = plotEfficientFrontier(meanReturns, covMatrix)

        graphCont = st.container(border=True)
        with graphCont:
            st.markdown("#### Efficient Frontier")
            st.pyplot(graph)

        

else:
    st.markdown(
        """
        <style>
        .bouncing-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%; /* Adjust as necessary for your layout */
        }

        .bouncing-dots {
            font-size: 20px;
            color: #555;
            display: inline-block;
            text-align: center;
        }

        .bouncing-dots span {
            display: inline-block;
            animation: bounce 1.5s infinite;
        }

        .bouncing-dots span:nth-child(2) {
            animation-delay: 0.3s;
        }

        .bouncing-dots span:nth-child(3) {
            animation-delay: 0.6s;
        }

        @keyframes bounce {
            0%, 80%, 100% {
                transform: scale(0);
            } 
            40% {
                transform: scale(1);
            }
        }
        </style>

        <div class="bouncing-container">
            <div class="bouncing-dots">
                Awaiting Input<span>.</span><span>.</span><span>.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
