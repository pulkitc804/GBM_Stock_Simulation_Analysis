# enhanced_visualizations.py
"""
Enhanced visualizations including interactive plots and advanced analytics.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


def plot_correlation_heatmap(returns_df, title="Asset Correlation Matrix"):
    """
    Plot correlation heatmap for multiple assets.

    Args:
        returns_df: DataFrame with returns for multiple assets
        title: Plot title
    """
    corr_matrix = returns_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, fmt='.2f',
                cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_rolling_statistics(prices, ticker, window=30):
    """
    Plot rolling mean and standard deviation.

    Args:
        prices: pandas Series of prices
        ticker: Stock ticker
        window: Rolling window size
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price with rolling mean
    ax1.plot(prices.index, prices.values, label='Price', alpha=0.7, color='blue')
    ax1.plot(rolling_mean.index, rolling_mean.values,
             label=f'{window}-Day Moving Average', color='red', linewidth=2)
    ax1.fill_between(prices.index,
                     prices.values - 2*rolling_std.values,
                     prices.values + 2*rolling_std.values,
                     alpha=0.2, color='gray', label='±2σ Band')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title(f'{ticker} - Price with Rolling Statistics', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Rolling volatility
    ax2.plot(rolling_std.index, rolling_std.values, color='orange', linewidth=2)
    ax2.fill_between(rolling_std.index, 0, rolling_std.values, alpha=0.3, color='orange')
    ax2.set_ylabel('Volatility ($)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_title(f'{window}-Day Rolling Volatility', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_monte_carlo_cone(historical_prices, time_array, price_paths,
                          stats_dict, ticker, percentiles=[5, 25, 75, 95]):
    """
    Plot Monte Carlo simulation results as a cone of uncertainty.

    Args:
        historical_prices: Historical price data
        time_array: Simulation time array
        price_paths: Simulated price paths
        stats_dict: Statistics dictionary
        ticker: Stock ticker
        percentiles: List of percentiles to plot
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    # Historical prices
    lookback = min(252, len(historical_prices))
    hist_dates = range(-lookback, 0)
    ax.plot(hist_dates, historical_prices.values[-lookback:],
            color='black', linewidth=3, label='Historical', zorder=5)

    # Median forecast
    ax.plot(time_array, stats_dict['median_path'],
            color='#E63946', linewidth=3, label='Median Forecast', zorder=4)

    # Percentile bands (creating uncertainty cone)
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(percentiles)//2))

    for i, (low, high) in enumerate(zip(percentiles[:len(percentiles)//2],
                                        reversed(percentiles[len(percentiles)//2:]))):
        low_pct = np.percentile(price_paths, low, axis=1)
        high_pct = np.percentile(price_paths, high, axis=1)

        ax.fill_between(time_array, low_pct, high_pct,
                        alpha=0.3, color=colors[i],
                        label=f'{low}th-{high}th Percentile')

    # Vertical line at present
    ax.axvline(0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(0, ax.get_ylim()[1] * 0.98, 'TODAY', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_title(f'{ticker} - Monte Carlo Price Forecast Cone',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Days from Today', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_return_distribution_analysis(log_returns, ticker):
    """
    Comprehensive return distribution analysis with multiple plots.

    Args:
        log_returns: Series of log returns
        ticker: Stock ticker
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Histogram with normal overlay
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(log_returns, bins=50, density=True, alpha=0.7,
             color='skyblue', edgecolor='black', label='Returns')
    mu, sigma = log_returns.mean(), log_returns.std()
    x = np.linspace(log_returns.min(), log_returns.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
             label='Normal Distribution')
    ax1.axvline(mu, color='green', linestyle='--', linewidth=2, label=f'Mean: {mu:.4f}')
    ax1.set_title('Return Distribution vs Normal', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Log Return')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 2])
    stats.probplot(log_returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Time series of returns
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(log_returns.index, log_returns.values, linewidth=0.8, alpha=0.7)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax3.fill_between(log_returns.index, 0, log_returns.values,
                     where=(log_returns.values > 0), alpha=0.3, color='green')
    ax3.fill_between(log_returns.index, 0, log_returns.values,
                     where=(log_returns.values < 0), alpha=0.3, color='red')
    ax3.set_title('Returns Over Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Log Return')
    ax3.grid(True, alpha=0.3)

    # 4. ACF plot
    ax4 = fig.add_subplot(gs[2, 0])
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(log_returns, lags=40, ax=ax4, alpha=0.05)
    ax4.set_title('Autocorrelation', fontsize=11, fontweight='bold')

    # 5. Volatility clustering (ACF of squared returns)
    ax5 = fig.add_subplot(gs[2, 1])
    plot_acf(log_returns**2, lags=40, ax=ax5, alpha=0.05)
    ax5.set_title('ACF of Squared Returns', fontsize=11, fontweight='bold')

    # 6. Statistics box
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    skewness = stats.skew(log_returns)
    kurtosis = stats.kurtosis(log_returns)
    jb_stat, jb_pvalue = stats.jarque_bera(log_returns)

    stats_text = f"""
    STATISTICS
    ──────────────
    Mean: {mu:.6f}
    Std Dev: {sigma:.6f}
    Skewness: {skewness:.4f}
    Kurtosis: {kurtosis:.4f}
    
    Jarque-Bera Test:
    Statistic: {jb_stat:.2f}
    P-value: {jb_pvalue:.4f}
    
    Min: {log_returns.min():.6f}
    Max: {log_returns.max():.6f}
    """

    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

    fig.suptitle(f'{ticker} - Comprehensive Return Analysis',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.show()


def plot_risk_metrics_dashboard(price_paths, S0, ticker, forecast_days):
    """
    Dashboard showing various risk metrics.

    Args:
        price_paths: Simulated price paths
        S0: Initial price
        ticker: Stock ticker
        forecast_days: Forecast horizon
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    final_prices = price_paths[-1, :]
    returns = (final_prices - S0) / S0

    # 1. Distribution of final prices
    ax1 = axes[0, 0]
    ax1.hist(final_prices, bins=50, color='teal', alpha=0.7, edgecolor='black')
    ax1.axvline(S0, color='red', linestyle='--', linewidth=2, label=f'Current: ${S0:.2f}')
    ax1.axvline(np.mean(final_prices), color='yellow', linestyle='--',
                linewidth=2, label=f'Mean: ${np.mean(final_prices):.2f}')
    ax1.set_title('Distribution of Final Prices', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Price ($)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribution of returns
    ax2 = axes[0, 1]
    ax2.hist(returns * 100, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    var_95 = -np.percentile(returns, 5) * 100
    ax2.axvline(-var_95, color='darkred', linestyle='--', linewidth=2,
                label=f'VaR (95%): {var_95:.1f}%')
    ax2.set_title('Distribution of Returns', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Probability of profit/loss
    ax3 = axes[1, 0]
    prob_profit = (returns > 0).sum() / len(returns) * 100
    prob_loss = 100 - prob_profit
    ax3.bar(['Profit', 'Loss'], [prob_profit, prob_loss],
            color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax3.set_title('Probability of Profit/Loss', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probability (%)')
    ax3.set_ylim(0, 100)
    for i, (label, value) in enumerate([('Profit', prob_profit), ('Loss', prob_loss)]):
        ax3.text(i, value + 2, f'{value:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Risk metrics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    var_95 = -np.percentile(returns, 5)
    var_99 = -np.percentile(returns, 1)
    cvar_95 = -returns[returns <= -var_95].mean()
    expected_return = returns.mean()
    volatility = returns.std()

    risk_text = f"""
    RISK METRICS ({forecast_days} days)
    {'='*35}
    
    Expected Return:    {expected_return*100:>8.2f}%
    Volatility:         {volatility*100:>8.2f}%
    
    Value at Risk (VaR):
      95% Confidence:   {var_95*100:>8.2f}%
      99% Confidence:   {var_99*100:>8.2f}%
    
    CVaR (95%):         {cvar_95*100:>8.2f}%
    
    Best Case (95%):    {np.percentile(returns, 95)*100:>8.2f}%
    Worst Case (5%):    {np.percentile(returns, 5)*100:>8.2f}%
    
    Probability > 0:    {prob_profit:>8.1f}%
    """

    ax4.text(0.1, 0.9, risk_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    fig.suptitle(f'{ticker} - Risk Analysis Dashboard',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()