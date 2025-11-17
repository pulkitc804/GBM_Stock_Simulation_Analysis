# plotter.py
"""
Visualization functions for stock price analysis and GBM simulations.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")


def plot_historical_prices(prices, ticker):
    """
    Plot historical stock prices.

    Args:
        prices: pandas Series of prices
        ticker: Stock ticker symbol
    """
    plt.figure(figsize=(12, 5))
    plt.plot(prices.index, prices.values, color='#2E86AB', linewidth=1.5)
    plt.title(f'{ticker} Historical Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Price ($)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_returns_analysis(log_returns, ticker):
    """
    Plot returns distribution and QQ plot.

    Args:
        log_returns: pandas Series of log returns
        ticker: Stock ticker symbol
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of returns
    axes[0].hist(log_returns, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[0].axvline(log_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {log_returns.mean():.4f}')
    axes[0].set_title(f'{ticker} Daily Log Returns Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Log Return', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # QQ plot
    stats.probplot(log_returns, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_simulation_results(historical_prices, time_array, price_paths, stats_dict, ticker):
    """
    Plot GBM simulation results with historical data.

    Args:
        historical_prices: pandas Series of historical prices
        time_array: Array of time steps
        price_paths: numpy array of simulated price paths
        stats_dict: Dictionary of statistics from simulations
        ticker: Stock ticker symbol
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot historical prices (last year)
    lookback = min(252, len(historical_prices))
    hist_dates = range(-lookback, 0)
    ax.plot(hist_dates, historical_prices.values[-lookback:],
            color='black', linewidth=2.5, label='Historical', zorder=5)

    # Plot a sample of simulated paths
    sample_paths = min(50, price_paths.shape[1])
    for i in range(sample_paths):
        ax.plot(time_array, price_paths[:, i],
                color='lightblue', alpha=0.3, linewidth=0.5)

    # Plot mean and percentiles
    ax.plot(time_array, stats_dict['mean_path'],
            color='#E63946', linewidth=2.5, label='Mean Forecast', zorder=4)
    ax.fill_between(time_array,
                    stats_dict['percentile_5'],
                    stats_dict['percentile_95'],
                    color='#F4A261', alpha=0.3, label='90% Confidence Interval')

    # Add vertical line at present
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0, ax.get_ylim()[1] * 0.95, 'Today', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Formatting
    ax.set_title(f'{ticker} - GBM Price Forecast ({price_paths.shape[1]} simulations)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Days', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_final_distribution(price_paths, S0, ticker, forecast_days):
    """
    Plot distribution of final prices.

    Args:
        price_paths: numpy array of simulated price paths
        S0: Initial price
        ticker: Stock ticker symbol
        forecast_days: Number of days forecasted
    """
    final_prices = price_paths[-1, :]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(final_prices, bins=50, color='#2A9D8F', alpha=0.7, edgecolor='black')

    # Add vertical lines for statistics
    mean_price = final_prices.mean()
    median_price = np.median(final_prices)
    ax.axvline(S0, color='black', linestyle='--', linewidth=2, label=f'Current: ${S0:.2f}')
    ax.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.2f}')
    ax.axvline(median_price, color='orange', linestyle='--', linewidth=2, label=f'Median: ${median_price:.2f}')

    # Add percentiles
    p5 = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)
    ax.axvline(p5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'5th %ile: ${p5:.2f}')
    ax.axvline(p95, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'95th %ile: ${p95:.2f}')

    ax.set_title(f'{ticker} - Price Distribution After {forecast_days} Days',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Price ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_summary_statistics(mu, sigma, stats_dict, var, S0, ticker):
    """
    Print summary statistics to console.

    Args:
        mu: Estimated drift
        sigma: Estimated volatility
        stats_dict: Dictionary of simulation statistics
        var: Value at Risk
        S0: Initial price
        ticker: Stock ticker symbol
    """
    print("\n" + "="*60)
    print(f"📊 SUMMARY STATISTICS FOR {ticker}")
    print("="*60)
    print(f"\n📈 Model Parameters:")
    print(f"   Annual Drift (μ):      {mu:.4f} ({mu*100:.2f}%)")
    print(f"   Annual Volatility (σ): {sigma:.4f} ({sigma*100:.2f}%)")

    print(f"\n💰 Current Price: ${S0:.2f}")

    print(f"\n🎯 Forecast Statistics:")
    print(f"   Mean Final Price:   ${stats_dict['final_mean']:.2f}")
    print(f"   Median Final Price: ${stats_dict['final_median']:.2f}")
    print(f"   Std Deviation:      ${stats_dict['final_std']:.2f}")
    print(f"   Min Price:          ${stats_dict['final_min']:.2f}")
    print(f"   Max Price:          ${stats_dict['final_max']:.2f}")

    print(f"\n⚠️  Risk Metrics:")
    print(f"   95% Value at Risk:  {var*100:.2f}%")
    print(f"   5th Percentile:     ${np.percentile(price_paths[-1,:], 5):.2f}")
    print(f"   95th Percentile:    ${np.percentile(price_paths[-1,:], 95):.2f}")

    print("\n" + "="*60 + "\n")