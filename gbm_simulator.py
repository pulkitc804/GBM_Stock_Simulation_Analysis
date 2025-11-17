# gbm_simulator.py
"""
Geometric Brownian Motion simulation for stock prices.
"""
import numpy as np


def simulate_gbm(S0, mu, sigma, days=252, num_simulations=1000, seed=None):
    """
    Simulate stock price paths using Geometric Brownian Motion.

    GBM formula: dS = μ * S * dt + σ * S * dW
    Discrete form: S(t+1) = S(t) * exp((μ - 0.5*σ²)*dt + σ*√dt*Z)

    Args:
        S0: Initial stock price (float)
        mu: Annual drift (expected return)
        sigma: Annual volatility (standard deviation)
        days: Number of days to simulate (default 252 = 1 year)
        num_simulations: Number of paths to simulate
        seed: Random seed for reproducibility

    Returns:
        tuple: (time_array, price_paths)
            - time_array: Days from 0 to days
            - price_paths: numpy array of shape (days+1, num_simulations)
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure S0 is a scalar
    if hasattr(S0, 'iloc'):
        S0 = float(S0.iloc[-1] if len(S0) > 1 else S0.iloc[0])
    else:
        S0 = float(S0)

    # Time parameters
    dt = 1.0 / 252  # Daily time step (assuming 252 trading days per year)
    time_array = np.arange(0, days + 1)

    # Initialize price array
    prices = np.zeros((days + 1, num_simulations))
    prices[0, :] = S0

    # Generate random shocks
    random_shocks = np.random.standard_normal((days, num_simulations))

    # Simulate paths
    for t in range(1, days + 1):
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks[t - 1, :]
        prices[t, :] = prices[t - 1, :] * np.exp(drift + diffusion)

    return time_array, prices


def calculate_statistics(price_paths):
    """
    Calculate statistics from simulated price paths.

    Args:
        price_paths: numpy array of shape (days+1, num_simulations)

    Returns:
        dict: Statistics including mean, median, percentiles
    """
    final_prices = price_paths[-1, :]

    stats = {
        'mean_path': price_paths.mean(axis=1),
        'median_path': np.median(price_paths, axis=1),
        'percentile_5': np.percentile(price_paths, 5, axis=1),
        'percentile_25': np.percentile(price_paths, 25, axis=1),
        'percentile_75': np.percentile(price_paths, 75, axis=1),
        'percentile_95': np.percentile(price_paths, 95, axis=1),
        'final_mean': final_prices.mean(),
        'final_median': np.median(final_prices),
        'final_std': final_prices.std(),
        'final_min': final_prices.min(),
        'final_max': final_prices.max()
    }

    return stats


def calculate_var(price_paths, S0, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) from simulated paths.

    Args:
        price_paths: numpy array of price paths
        S0: Initial price
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        float: VaR as a percentage loss
    """
    final_prices = price_paths[-1, :]
    returns = (final_prices - S0) / S0
    var = -np.percentile(returns, (1 - confidence_level) * 100)
    return var