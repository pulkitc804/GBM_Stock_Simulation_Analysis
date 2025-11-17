# portfolio_optimization.py
"""
Portfolio optimization using Modern Portfolio Theory (Markowitz).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def calculate_portfolio_metrics(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio expected return and volatility.

    Args:
        weights: Array of portfolio weights
        mean_returns: Array of expected returns
        cov_matrix: Covariance matrix of returns

    Returns:
        tuple: (expected_return, volatility)
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Calculate negative Sharpe ratio (for minimization).

    Args:
        weights: Portfolio weights
        mean_returns: Expected returns
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate

    Returns:
        float: Negative Sharpe ratio
    """
    p_return, p_volatility = calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
    sharpe = (p_return - risk_free_rate) / p_volatility
    return -sharpe


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02,
                       target_return=None, allow_short=False):
    """
    Find optimal portfolio weights (maximum Sharpe ratio or minimum variance).

    Args:
        mean_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        target_return: Target return (for minimum variance portfolio)
        allow_short: Allow short selling

    Returns:
        dict: Optimal weights and portfolio metrics
    """
    num_assets = len(mean_returns)

    # Initial guess (equal weights)
    init_weights = np.array([1.0 / num_assets] * num_assets)

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Add target return constraint if specified
    if target_return is not None:
        constraints = [
            constraints,
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return}
        ]

    # Bounds: 0 <= weight <= 1 (or allow negative for short selling)
    bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(num_assets))

    # Optimize
    if target_return is None:
        # Maximize Sharpe ratio
        result = minimize(
            negative_sharpe_ratio,
            init_weights,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    else:
        # Minimize variance for target return
        result = minimize(
            lambda w: calculate_portfolio_metrics(w, mean_returns, cov_matrix)[1],
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

    optimal_weights = result.x
    p_return, p_volatility = calculate_portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
    sharpe = (p_return - risk_free_rate) / p_volatility

    return {
        'weights': optimal_weights,
        'return': p_return,
        'volatility': p_volatility,
        'sharpe_ratio': sharpe
    }


def efficient_frontier(mean_returns, cov_matrix, num_portfolios=100,
                       risk_free_rate=0.02, allow_short=False):
    """
    Generate efficient frontier.

    Args:
        mean_returns: Expected returns
        cov_matrix: Covariance matrix
        num_portfolios: Number of portfolios to generate
        risk_free_rate: Risk-free rate
        allow_short: Allow short selling

    Returns:
        DataFrame: Portfolio returns and volatilities on efficient frontier
    """
    min_return = mean_returns.min()
    max_return = mean_returns.max()
    target_returns = np.linspace(min_return, max_return, num_portfolios)

    frontier_volatilities = []
    frontier_returns = []
    frontier_weights = []

    for target in target_returns:
        try:
            result = optimize_portfolio(
                mean_returns,
                cov_matrix,
                risk_free_rate,
                target_return=target,
                allow_short=allow_short
            )
            frontier_returns.append(result['return'])
            frontier_volatilities.append(result['volatility'])
            frontier_weights.append(result['weights'])
        except:
            continue

    return pd.DataFrame({
        'return': frontier_returns,
        'volatility': frontier_volatilities,
        'weights': frontier_weights
    })


def plot_efficient_frontier(frontier_df, optimal_portfolio=None,
                            individual_assets=None, asset_names=None):
    """
    Plot the efficient frontier.

    Args:
        frontier_df: DataFrame from efficient_frontier()
        optimal_portfolio: Dict with optimal portfolio metrics
        individual_assets: Tuple of (returns, volatilities) for individual assets
        asset_names: List of asset names
    """
    plt.figure(figsize=(12, 7))

    # Plot efficient frontier
    plt.plot(frontier_df['volatility'], frontier_df['return'],
             'b-', linewidth=2, label='Efficient Frontier')

    # Plot optimal portfolio (max Sharpe)
    if optimal_portfolio:
        plt.scatter(optimal_portfolio['volatility'],
                    optimal_portfolio['return'],
                    marker='*', color='red', s=500,
                    label=f"Max Sharpe ({optimal_portfolio['sharpe_ratio']:.2f})",
                    zorder=5)

    # Plot individual assets
    if individual_assets:
        returns, volatilities = individual_assets
        plt.scatter(volatilities, returns, marker='o', s=100,
                    alpha=0.6, label='Individual Assets')

        if asset_names:
            for i, name in enumerate(asset_names):
                plt.annotate(name,
                             (volatilities[i], returns[i]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=9)

    plt.xlabel('Volatility (Risk)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier - Portfolio Optimization', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_var_cvar(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95)

    Returns:
        tuple: (VaR, CVaR)
    """
    var = -np.percentile(returns, (1 - confidence_level) * 100)
    cvar = -returns[returns <= -var].mean()
    return var, cvar


def monte_carlo_portfolio_simulation(weights, mean_returns, cov_matrix,
                                     initial_investment=10000, days=252,
                                     num_simulations=1000, seed=None):
    """
    Monte Carlo simulation for portfolio value.

    Args:
        weights: Portfolio weights
        mean_returns: Expected returns (annualized)
        cov_matrix: Covariance matrix (annualized)
        initial_investment: Starting portfolio value
        days: Simulation horizon
        num_simulations: Number of paths
        seed: Random seed

    Returns:
        numpy array: Simulated portfolio values (days+1, num_simulations)
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert annual to daily
    daily_returns = mean_returns / 252
    daily_cov = cov_matrix / 252

    portfolio_values = np.zeros((days + 1, num_simulations))
    portfolio_values[0, :] = initial_investment

    for t in range(1, days + 1):
        # Generate correlated random returns for each asset
        random_returns = np.random.multivariate_normal(daily_returns, daily_cov, num_simulations)

        # Calculate portfolio return
        portfolio_returns = np.dot(random_returns, weights)

        # Update portfolio value
        portfolio_values[t, :] = portfolio_values[t-1, :] * (1 + portfolio_returns)

    return portfolio_values