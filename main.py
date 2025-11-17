# main.py
"""
Main script for Stock Price Prediction using Geometric Brownian Motion.

This script:
1. Fetches historical stock data
2. Estimates GBM parameters (drift and volatility)
3. Simulates future price paths
4. Visualizes results and statistics
"""

from data_fetcher import fetch_stock_data, calculate_log_returns, estimate_parameters
from gbm_simulator import simulate_gbm, calculate_statistics, calculate_var
from plotter import (plot_historical_prices, plot_returns_analysis,
                     plot_simulation_results, plot_final_distribution,
                     print_summary_statistics)


def main():
    """
    Main function to run the GBM stock price simulation.
    """
    print("="*60)
    print("🚀 STOCK PRICE PREDICTION WITH GBM")
    print("="*60)

    # ============= USER INPUTS =============
    ticker = input("\n📊 Enter stock ticker (e.g., AAPL, TSLA, MSFT): ").strip().upper()
    start_date = input("📅 Enter start date (YYYY-MM-DD, e.g., 2020-01-01): ").strip()

    forecast_input = input("🔮 Enter forecast period in days (default 252 = 1 year): ").strip()
    forecast_days = int(forecast_input) if forecast_input else 252

    sim_input = input("🎲 Enter number of simulations (default 1000): ").strip()
    num_simulations = int(sim_input) if sim_input else 1000

    print("\n" + "="*60)
    print("⏳ Processing...")
    print("="*60 + "\n")

    # ============= STEP 1: FETCH DATA =============
    try:
        prices = fetch_stock_data(ticker, start_date)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    # Plot historical prices
    plot_historical_prices(prices, ticker)

    # ============= STEP 2: CALCULATE RETURNS & PARAMETERS =============
    log_returns = calculate_log_returns(prices)
    mu, sigma = estimate_parameters(log_returns)

    print(f"\n✅ Estimated Parameters:")
    print(f"   Drift (μ): {mu:.4f} ({mu*100:.2f}% annually)")
    print(f"   Volatility (σ): {sigma:.4f} ({sigma*100:.2f}% annually)")

    # Plot returns analysis
    plot_returns_analysis(log_returns, ticker)

    # ============= STEP 3: RUN GBM SIMULATION =============
    print(f"\n🔄 Running {num_simulations} simulations for {forecast_days} days...")

    S0 = prices.iloc[-1]  # Current price
    time_array, price_paths = simulate_gbm(
        S0=S0,
        mu=mu,
        sigma=sigma,
        days=forecast_days,
        num_simulations=num_simulations,
        seed=42  # For reproducibility
    )

    print("✅ Simulation complete!")

    # ============= STEP 4: CALCULATE STATISTICS =============
    stats_dict = calculate_statistics(price_paths)
    var = calculate_var(price_paths, S0, confidence_level=0.95)

    # ============= STEP 5: VISUALIZE RESULTS =============
    print("\n📈 Generating plots...\n")

    # Plot simulation results
    plot_simulation_results(prices, time_array, price_paths, stats_dict, ticker)

    # Plot final price distribution
    plot_final_distribution(price_paths, S0, ticker, forecast_days)

    # ============= STEP 6: PRINT SUMMARY =============
    # Need to pass price_paths globally or fix the print function
    import numpy as np

    print("\n" + "="*60)
    print(f"📊 SUMMARY STATISTICS FOR {ticker}")
    print("="*60)
    print(f"\n📈 Model Parameters:")
    print(f"   Annual Drift (μ):      {mu:.4f} ({mu*100:.2f}%)")
    print(f"   Annual Volatility (σ): {sigma:.4f} ({sigma*100:.2f}%)")

    print(f"\n💰 Current Price: ${S0:.2f}")

    print(f"\n🎯 Forecast Statistics (after {forecast_days} days):")
    print(f"   Mean Final Price:   ${stats_dict['final_mean']:.2f}")
    print(f"   Median Final Price: ${stats_dict['final_median']:.2f}")
    print(f"   Std Deviation:      ${stats_dict['final_std']:.2f}")
    print(f"   Min Price:          ${stats_dict['final_min']:.2f}")
    print(f"   Max Price:          ${stats_dict['final_max']:.2f}")

    print(f"\n⚠️  Risk Metrics:")
    print(f"   95% Value at Risk:  {var*100:.2f}%")
    print(f"   5th Percentile:     ${np.percentile(price_paths[-1,:], 5):.2f}")
    print(f"   95th Percentile:    ${np.percentile(price_paths[-1,:], 95):.2f}")

    expected_return = ((stats_dict['final_mean'] - S0) / S0) * 100
    print(f"\n📊 Expected Return: {expected_return:.2f}%")

    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()