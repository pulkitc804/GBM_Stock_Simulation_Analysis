# data_fetcher.py
"""
Fetches historical stock price data using yfinance.
"""
import pandas as pd
import yfinance as yf
import numpy as np


def fetch_stock_data(ticker, start_date, end_date=None):
    """
    Fetch historical stock price data.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date as string 'YYYY-MM-DD'
        end_date: End date as string 'YYYY-MM-DD' (optional, defaults to today)

    Returns:
        pandas Series of adjusted close prices
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date or 'today'}...")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {ticker}. Check the ticker symbol and date range.")

    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.get_level_values(0):
            prices = df['Adj Close'].squeeze()
        elif 'Close' in df.columns.get_level_values(0):
            prices = df['Close'].squeeze()
        else:
            raise ValueError("Could not find price column in data")
    else:
        if 'Adj Close' in df.columns:
            prices = df['Adj Close']
        elif 'Close' in df.columns:
            prices = df['Close']
        else:
            raise ValueError("Could not find price column in data")

    # Clean and sort
    prices = prices.dropna().sort_index()

    print(f"✓ Fetched {len(prices)} data points from {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices


def calculate_log_returns(prices):
    """
    Calculate log returns from price series.

    Args:
        prices: pandas Series of prices

    Returns:
        pandas Series of log returns
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def estimate_parameters(log_returns, trading_days=252):
    """
    Estimate GBM parameters (drift and volatility) from historical log returns.

    Args:
        log_returns: pandas Series of log returns
        trading_days: Number of trading days per year (default 252)

    Returns:
        tuple: (mu, sigma) - annualized drift and volatility
    """
    # Daily statistics
    daily_mean = log_returns.mean()
    daily_std = log_returns.std()

    # Annualize
    mu = daily_mean * trading_days + 0.5 * (daily_std ** 2) * trading_days
    sigma = daily_std * np.sqrt(trading_days)

    return mu, sigma