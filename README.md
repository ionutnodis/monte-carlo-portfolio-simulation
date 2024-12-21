# Optimal Portfolio Monte Carlo Simulations

This project introduces a robust Streamlit web application designed to optimize portfolio allocation using Monte Carlo simulations. Users can dynamically retrieve financial data from Yahoo Finance, analyze portfolio performance, and export actionable insights through detailed visualizations and downloadable reports.

## Key Features

- **Dynamic Asset Selection**: Input multiple stock or ETF tickers for in-depth analysis.
- **Monte Carlo Portfolio Simulation**: Simulates thousands of portfolio combinations to identify optimal allocation strategies.
- **Customizable Constraints**: Define minimum and maximum weights for individual assets, including a risk-free asset.
- **Advanced Risk Metrics**: Analyze Maximum Drawdown and Value at Risk (VaR) at a 95% confidence level.
- **Efficient Frontier Analysis**: Visualize the trade-off between portfolio risk and return.
- **Correlation Heatmap**: Explore interdependencies among selected assets with a detailed heatmap.
- **Historical Performance Visualization**: Track cumulative returns for the optimal portfolio over the analysis period.
- **Comprehensive Report Generation**: Export optimal weights and key portfolio metrics in a CSV format.

## How to Set Up and Run

### System Requirements

- Python 3.8 or later
- Required libraries:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `streamlit`

Install the dependencies using:

```bash
pip install yfinance pandas numpy matplotlib seaborn streamlit
```

### Steps to Execute

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/monte-carlo-portfolio
   cd monte-carlo-portfolio
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Access the application via the local URL provided (e.g., `http://localhost:8501`).

## Usage Guide

1. **Configure Inputs**: Use the sidebar to define the start and end dates, asset tickers, and portfolio constraints.
2. **Analyze Outputs**:
   - **Optimal Portfolio Weights**: View allocation for each asset.
   - **Performance Metrics**: Access detailed risk and return insights.
   - **Heatmap**: Examine correlations between selected assets.
   - **Efficient Frontier**: Explore the risk-return spectrum of the simulated portfolios.
   - **Historical Returns**: Assess the cumulative performance of the optimal portfolio.
3. **Export Results**: Download the analysis report as a CSV file.

## File Structure

- `streamlit_app.py`: The main application script encompassing data retrieval, Monte Carlo simulations, and interactive visualizations.

## Visualization Highlights

### Interactive Sidebar Configuration

Easily customize analysis parameters, including date ranges, tickers, and portfolio constraints, using the interactive sidebar interface.

### Efficient Frontier

The efficient frontier graph showcases the risk-return trade-offs for simulated portfolios, with the optimal portfolio prominently highlighted.

### Risk and Correlation Analysis

- **Risk Metrics**: Provides a breakdown of key risk measures, including Maximum Drawdown and Value at Risk.
- **Correlation Heatmap**: Offers an intuitive visualization of asset correlations, aiding diversification decisions.

## Example Outputs

- **Optimal Portfolio Weights**:
  - Asset A: 40%
  - Asset B: 50%
  - Risk-Free Asset: 10%
- **Risk Metrics**:
  - Maximum Drawdown: -12%
  - VaR (95% Confidence): -8%

## About the Author

- **Ionut Catalin Nodis**: Creator and developer of this application.

## License

This project is distributed under the MIT License. See the `LICENSE` file for full details.

