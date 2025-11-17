# advanced_models.py
"""
Advanced stochastic models for stock price prediction.
Includes Jump Diffusion (Merton Model) and GARCH volatility.
"""
import numpy as np
from arch import arch_model


def simulate_jump_diffusion(S0, mu, sigma, jump_lambda, jump_mean, jump_std,
                            days=252, num_simulations=1000, seed=None):
    """
    Merton Jump Diffusion Model - adds sudden jumps to GBM.

    dS = μ*S*dt + σ*S*dW + J*S*dN
    where J = jump size, N = Poisson process

    Args:
        S0: Initial price
        mu: Drift
        sigma: Volatility
        jump_lambda: Average number of jumps per year
        jump_mean: Mean jump size (as %)
        jump_std: Std dev of jump size
        days: Forecast horizon
        num_simulations: Number of paths
        seed: Random seed

    Returns:
        tuple: (time_array, price_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    S0 = float(S0) if not isinstance(S0, float) else S0
    dt = 1.0 / 252
    time_array = np.arange(0, days + 1)

    prices = np.zeros((days + 1, num_simulations))
    prices[0, :] = S0

    for t in range(1, days + 1):
        # Standard GBM component
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.standard_normal(num_simulations)

        # Jump component (Poisson process)
        num_jumps = np.random.poisson(jump_lambda * dt, num_simulations)
        jump_sizes = np.zeros(num_simulations)

        for i in range(num_simulations):
            if num_jumps[i] > 0:
                jumps = np.random.normal(jump_mean, jump_std, num_jumps[i])
                jump_sizes[i] = np.sum(jumps)

        # Combine: GBM + Jumps
        prices[t, :] = prices[t - 1, :] * np.exp(drift + diffusion + jump_sizes)

    return time_array, prices


def fit_garch_model(log_returns):
    """
    Fit GARCH(1,1) model to capture volatility clustering.

    Args:
        log_returns: pandas Series of log returns

    Returns:
        fitted GARCH model
    """
    # Convert to percentage returns for ARCH package
    returns_pct = log_returns * 100

    # Fit GARCH(1,1)
    model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal')
    result = model.fit(disp='off', show_warning=False)

    return result


def forecast_garch_volatility(garch_result, horizon=252):
    """
    Forecast future volatility using fitted GARCH model.

    Args:
        garch_result: Fitted GARCH model
        horizon: Forecast horizon in days

    Returns:
        numpy array of forecasted volatilities
    """
    forecast = garch_result.forecast(horizon=horizon)
    variance_forecast = forecast.variance.values[-1, :]
    volatility_forecast = np.sqrt(variance_forecast) / 100  # Convert back from %

    return volatility_forecast


def simulate_gbm_with_garch(S0, mu, garch_volatility, num_simulations=1000, seed=None):
    """
    Simulate GBM with time-varying GARCH volatility.

    Args:
        S0: Initial price
        mu: Drift
        garch_volatility: Array of forecasted volatilities
        num_simulations: Number of paths
        seed: Random seed

    Returns:
        tuple: (time_array, price_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    S0 = float(S0) if not isinstance(S0, float) else S0
    days = len(garch_volatility)
    dt = 1.0 / 252
    time_array = np.arange(0, days + 1)

    prices = np.zeros((days + 1, num_simulations))
    prices[0, :] = S0

    for t in range(1, days + 1):
        sigma_t = garch_volatility[t - 1]  # Time-varying volatility
        drift = (mu - 0.5 * sigma_t ** 2) * dt
        diffusion = sigma_t * np.sqrt(dt) * np.random.standard_normal(num_simulations)
        prices[t, :] = prices[t - 1, :] * np.exp(drift + diffusion)

    return time_array, prices


def detect_regime_changes(log_returns, window=60):
    """
    Detect regime changes (bull/bear markets) using rolling volatility.

    Args:
        log_returns: pandas Series of log returns
        window: Rolling window size

    Returns:
        pandas Series indicating regimes (1=high vol, 0=low vol)
    """
    rolling_vol = log_returns.rolling(window=window).std()
    median_vol = rolling_vol.median()

    # High volatility regime if above median
    regimes = (rolling_vol > median_vol).astype(int)

    return regimes


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate

    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe


def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown from price series.

    Args:
        prices: Array or Series of prices

    Returns:
        float: Maximum drawdown as percentage
    """
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    max_dd = drawdown.min()
    return abs(max_dd)