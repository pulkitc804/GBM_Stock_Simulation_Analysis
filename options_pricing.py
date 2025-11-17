# options_pricing.py
"""
Options pricing using Black-Scholes and Monte Carlo simulation.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar


def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for European call option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        float: Call option price
    """
    if T <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """
    Black-Scholes formula for European put option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        float: Put option price
    """
    if T <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def monte_carlo_option_price(S0, K, T, r, sigma, option_type='call',
                             num_simulations=10000, seed=None):
    """
    Price European options using Monte Carlo simulation.

    Args:
        S0: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        num_simulations: Number of simulations
        seed: Random seed

    Returns:
        tuple: (option_price, standard_error)
    """
    if seed is not None:
        np.random.seed(seed)

    # Simulate final stock prices
    Z = np.random.standard_normal(num_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:  # put
        payoffs = np.maximum(K - ST, 0)

    # Discount back to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(num_simulations)

    return option_price, standard_error


def calculate_greeks(S, K, T, r, sigma):
    """
    Calculate option Greeks for a call option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        dict: Greeks (delta, gamma, vega, theta, rho)
    """
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    delta = norm.cdf(d1)

    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (divided by 100 to get per 1% change in volatility)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Theta (per day)
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365

    # Rho (per 1% change in interest rate)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def implied_volatility(option_price, S, K, T, r, option_type='call',
                       initial_guess=0.3):
    """
    Calculate implied volatility from option price using Newton-Raphson.

    Args:
        option_price: Observed option price
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        option_type: 'call' or 'put'
        initial_guess: Initial volatility guess

    Returns:
        float: Implied volatility
    """
    def objective(sigma):
        if option_type == 'call':
            theoretical_price = black_scholes_call(S, K, T, r, sigma)
        else:
            theoretical_price = black_scholes_put(S, K, T, r, sigma)
        return abs(theoretical_price - option_price)

    result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
    return result.x


def asian_option_price(S0, K, T, r, sigma, option_type='call',
                       num_simulations=10000, num_steps=252, seed=None):
    """
    Price Asian (average price) options using Monte Carlo.

    Args:
        S0: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        num_simulations: Number of paths
        num_steps: Number of time steps
        seed: Random seed

    Returns:
        float: Asian option price
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / num_steps
    prices = np.zeros((num_steps + 1, num_simulations))
    prices[0, :] = S0

    # Simulate paths
    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        prices[t, :] = prices[t-1, :] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        )

    # Calculate average price for each path
    avg_prices = prices.mean(axis=0)

    # Calculate payoffs based on average price
    if option_type == 'call':
        payoffs = np.maximum(avg_prices - K, 0)
    else:
        payoffs = np.maximum(K - avg_prices, 0)

    # Discount to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price