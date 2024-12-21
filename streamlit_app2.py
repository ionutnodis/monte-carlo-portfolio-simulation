
"""
Monte Carlo Portfolio Optimization Web Application
This Streamlit application performs portfolio optimization using Monte Carlo simulation
to find the optimal asset allocation based on historical data. The app allows users
to input multiple stock tickers and customize various parameters to simulate and
visualize the optimal portfolio allocation.

Key Features:

- Interactive date range and ticker symbol selection
- Customizable risk-free rate and number of portfolio simulations  
- Individual asset weight constraints
- Portfolio optimization using Modern Portfolio Theory
- Risk metrics calculation (Max Drawdown, VaR)
- Visual analysis through:
    - Historical performance chart
    - Correlation heatmap
    - Efficient frontier plot
-  Downloadable portfolio report

The optimization process:
1. Fetches historical price data using yfinance
2. Calculates returns and covariance matrix
3. Runs Monte Carlo simulation with user-defined constraints
4. Finds optimal portfolio weights using the Sharpe ratio
5. Generates interactive visualizations and risk metrics

Dependencies:
- yfinance: For fetching historical stock data
- pandas: For data manipulation and analysis
- numpy: For numerical computations
- matplotlib: For plotting
- seaborn: For correlation heatmap
- streamlit: For web application interface

Usage:
Run the script using: streamlit run streamlit_app2.py
"""

import yfinance as yf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit title
st.title("Monte Carlo Portfolio Simulation with Enhanced Features")

# Inputs for Streamlit
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
tickers = st.sidebar.text_input("Enter Tickers (comma-separated)", value="SPY,VTI,QQQ")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.1) / 100
num_portfolios = st.sidebar.slider("Number of Portfolios to Simulate", 1000, 100000, 50000, 1000)

# Split tickers and fetch historical data dynamically
ticker_list = [ticker.strip() for ticker in tickers.split(",")]
st.write(f"Fetching historical data for: {', '.join(ticker_list)}")
data = {}
for ticker in ticker_list:
    data[ticker] = yf.download(ticker, start=start_date, end=end_date)

# Combine adjusted close prices into a single DataFrame
prices = pd.concat([data[ticker]["Adj Close"] for ticker in ticker_list], axis=1)
prices.columns = ticker_list

# Calculate daily returns
returns = prices.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Portfolio constraints
st.sidebar.header("Portfolio Constraints")
constraints = {
    ticker: {
        "min": st.sidebar.slider(f"Min weight for {ticker}", 0.0, 1.0, 0.0, 0.01),
        "max": st.sidebar.slider(f"Max weight for {ticker}", 0.0, 1.0, 1.0, 0.01)
    }
    for ticker in ticker_list
}
st.sidebar.write("Risk-Free Asset Constraints")
rf_min = st.sidebar.slider("Min weight for Risk-Free Asset", 0.0, 1.0, 0.0, 0.01)
rf_max = st.sidebar.slider("Max weight for Risk-Free Asset", 0.0, 1.0, 1.0, 0.01)

# Simulate portfolios (including risk-free asset)
st.write("Running Monte Carlo Simulation...")
weights = np.random.dirichlet(np.ones(len(ticker_list) + 1), size=num_portfolios)  # Include risk-free asset
weights_risky = weights[:, :-1]
weights_rf = weights[:, -1]  # Weight for risk-free asset

# Apply constraints
valid_indices = np.ones(len(weights), dtype=bool)
for i, ticker in enumerate(ticker_list):
    valid_indices &= (weights_risky[:, i] >= constraints[ticker]["min"]) & (weights_risky[:, i] <= constraints[ticker]["max"])
valid_indices &= (weights_rf >= rf_min) & (weights_rf <= rf_max)

weights_risky = weights_risky[valid_indices]
weights_rf = weights_rf[valid_indices]
weights = weights[valid_indices]

portfolio_returns = np.dot(weights_risky, mean_returns) + weights_rf * risk_free_rate
portfolio_vol = np.sqrt(np.einsum('ij,jk,ik->i', weights_risky, cov_matrix.values, weights_risky))
sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_vol

# Find the portfolio with the highest Sharpe ratio
max_sharpe_idx = np.argmax(sharpe_ratios)
optimal_weights = weights[max_sharpe_idx]

# Historical performance of the optimal portfolio
optimal_portfolio_returns = (returns @ optimal_weights[:-1]) + optimal_weights[-1] * risk_free_rate
cumulative_returns = (1 + optimal_portfolio_returns).cumprod()
st.subheader("Historical Performance of Optimal Portfolio")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cumulative_returns, label="Optimal Portfolio")
ax.set_title("Cumulative Returns")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend()
plt.grid(alpha=0.5)
st.pyplot(fig)

# Risk metrics
st.subheader("Risk Metrics")
max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
var_95 = np.percentile(optimal_portfolio_returns, 5)
st.write(f"Maximum Drawdown: {max_drawdown:.2%}")
st.write(f"Value at Risk (95% confidence): {var_95:.2%}")

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix of Asset Returns")
st.pyplot(fig)

# Efficient Frontier
st.subheader("Efficient Frontier")
efficient_vol = []
efficient_ret = []
for w in weights_risky:
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    efficient_vol.append(vol)
    efficient_ret.append(ret)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(efficient_vol, efficient_ret, c="blue", s=1, label="Efficient Frontier")
ax.scatter(portfolio_vol[max_sharpe_idx], portfolio_returns[max_sharpe_idx], color='red', s=100, label='Optimal Portfolio')
ax.set_title("Efficient Frontier")
ax.set_xlabel("Volatility (Risk)")
ax.set_ylabel("Return")
ax.legend()
st.pyplot(fig)

st.subheader("Downloadable Report")
def generate_report():
    # Create a DataFrame for weights
    weights_data = {
        "Ticker": ticker_list + ["Risk-Free Asset"],
        "Optimal Weight": list(optimal_weights)
    }
    weights_df = pd.DataFrame(weights_data)
    
    # Add single-value metrics as a separate DataFrame
    metrics_data = {
        "Metric": ["Expected Return", "Expected Volatility", "Sharpe Ratio", "Maximum Drawdown", "VaR (95%)"],
        "Value": [
            portfolio_returns[max_sharpe_idx],
            portfolio_vol[max_sharpe_idx],
            sharpe_ratios[max_sharpe_idx],
            max_drawdown,
            var_95
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Combine weights and metrics into one CSV string
    report_csv = weights_df.to_csv(index=False) + "\n" + metrics_df.to_csv(index=False)
    return report_csv

st.download_button("Download Portfolio Report", generate_report(), "portfolio_report.csv", "text/csv")

