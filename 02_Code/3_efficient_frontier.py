"""
German Market Asset Pricing - Efficient Frontier Analysis
Mean-Variance Portfolio Optimization using Monte Carlo Simulation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def efficient_frontier():
    print("=" * 60)
    print("EFFICIENT FRONTIER ANALYSIS")
    print("=" * 60)
    
    # ============================================================================
    # 1. DATA SETUP
    # ============================================================================
    print("\n1. Loading and preparing data...")
    
    # Load data
    data = pd.read_csv('german_market_data.csv', index_col=0, parse_dates=True)
    
    # Remove duplicate columns
    data = data.loc[:, ~data.columns.duplicated()]
    
    # Ensure index is Datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Calculate Daily Returns
    returns = data.pct_change()
    
    # Drop Market and RF columns (we only want stocks)
    stock_returns = returns.drop(columns=['Market', 'RF'], errors='ignore')
    
    print(f"   Initial stocks: {stock_returns.shape[1]}")
    
    # Filter: Select only the Top 20 stocks with most complete data (fewest NaNs)
    nan_counts = stock_returns.isna().sum()
    top_20_stocks = nan_counts.nsmallest(20).index.tolist()
    
    # Select only these 20 stocks
    stock_returns = stock_returns[top_20_stocks]
    
    print(f"   Selected top 20 stocks with most complete data")
    print(f"   Selected stocks: {', '.join(top_20_stocks)}")
    
    # Drop any remaining rows with NaNs for these 20 stocks
    stock_returns = stock_returns.dropna()
    
    print(f"   Final data shape: {stock_returns.shape[0]} rows, {stock_returns.shape[1]} columns")
    print(f"   Date range: {stock_returns.index.min().date()} to {stock_returns.index.max().date()}")
    
    # ============================================================================
    # 2. SIMULATION (MONTE CARLO LOOP)
    # ============================================================================
    print("\n2. Running Monte Carlo simulation (10,000 portfolios)...")
    
    # Parameters
    num_portfolios = 10000
    risk_free_rate = 0.025  # 2.5% annualized
    trading_days = 252  # Annualization factor
    
    # Calculate mean returns and covariance matrix
    mean_returns = stock_returns.mean() * trading_days  # Annualized
    cov_matrix = stock_returns.cov() * trading_days  # Annualized
    
    num_stocks = len(stock_returns.columns)
    
    # Storage for results
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpe_ratios = []
    portfolio_weights_list = []
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_portfolios):
        # Generate random weights (sum = 1)
        weights = np.random.random(num_stocks)
        weights = weights / np.sum(weights)
        
        # Calculate Portfolio Return (annualized)
        portfolio_return = np.sum(mean_returns * weights)
        
        # Calculate Portfolio Volatility (annualized standard deviation)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe Ratio = (Return - RiskFree) / Volatility
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Store results
        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_volatility)
        portfolio_sharpe_ratios.append(sharpe_ratio)
        portfolio_weights_list.append(weights)
    
    print(f"   Simulation complete!")
    
    # Convert to numpy arrays for easier manipulation
    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)
    portfolio_sharpe_ratios = np.array(portfolio_sharpe_ratios)
    portfolio_weights_list = np.array(portfolio_weights_list)
    
    # ============================================================================
    # 3. OPTIMIZATION RESULTS
    # ============================================================================
    print("\n3. Finding optimal portfolios...")
    
    # Find portfolio with Maximum Sharpe Ratio
    max_sharpe_idx = np.argmax(portfolio_sharpe_ratios)
    max_sharpe_return = portfolio_returns[max_sharpe_idx]
    max_sharpe_vol = portfolio_volatilities[max_sharpe_idx]
    max_sharpe_weights = portfolio_weights_list[max_sharpe_idx]
    
    # Find portfolio with Minimum Volatility
    min_vol_idx = np.argmin(portfolio_volatilities)
    min_vol_return = portfolio_returns[min_vol_idx]
    min_vol_vol = portfolio_volatilities[min_vol_idx]
    min_vol_weights = portfolio_weights_list[min_vol_idx]
    
    print(f"   Max Sharpe Portfolio: Return={max_sharpe_return:.2%}, Vol={max_sharpe_vol:.2%}, Sharpe={portfolio_sharpe_ratios[max_sharpe_idx]:.2f}")
    print(f"   Min Volatility Portfolio: Return={min_vol_return:.2%}, Vol={min_vol_vol:.2%}, Sharpe={portfolio_sharpe_ratios[min_vol_idx]:.2f}")
    
    # ============================================================================
    # 4. VISUALIZATION
    # ============================================================================
    print("\n4. Creating visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot (Vol vs Return)
    scatter = plt.scatter(
        portfolio_volatilities,
        portfolio_returns,
        c=portfolio_sharpe_ratios,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    
    # Plot Red Star for Max Sharpe portfolio
    plt.scatter(
        max_sharpe_vol,
        max_sharpe_return,
        marker='*',
        color='red',
        s=500,
        label='Max Sharpe Ratio',
        edgecolors='black',
        linewidths=1.5
    )
    
    # Plot Blue Diamond for Min Volatility portfolio
    plt.scatter(
        min_vol_vol,
        min_vol_return,
        marker='D',
        color='blue',
        s=500,
        label='Min Volatility',
        edgecolors='black',
        linewidths=1.5
    )
    
    # Add colorbar
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Labels and title
    plt.xlabel('Volatility (Annualized)', fontsize=12)
    plt.ylabel('Return (Annualized)', fontsize=12)
    plt.title('Efficient Frontier - German Market Portfolio Optimization', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    print(f"   Visualization saved to 'efficient_frontier.png'")
    
    # ============================================================================
    # 5. OUTPUT
    # ============================================================================
    print("\n" + "=" * 60)
    print("OPTIMAL PORTFOLIO WEIGHTS")
    print("=" * 60)
    
    # Create weight dictionaries
    max_sharpe_dict = dict(zip(stock_returns.columns, max_sharpe_weights))
    min_vol_dict = dict(zip(stock_returns.columns, min_vol_weights))
    
    # Print Optimal Portfolio Weights (Max Sharpe) - only weights > 1%
    print("\nOptimal Portfolio Weights (Max Sharpe Ratio):")
    print("-" * 60)
    max_sharpe_sorted = sorted(max_sharpe_dict.items(), key=lambda x: x[1], reverse=True)
    printed_any = False
    for stock, weight in max_sharpe_sorted:
        if weight > 0.01:  # Only print weights > 1%
            print(f"  {stock:15s}: {weight:6.2%}")
            printed_any = True
    if not printed_any:
        print("  (No weights > 1%)")
    
    # Print Safe Portfolio Weights (Min Volatility) - only weights > 1%
    print("\nSafe Portfolio Weights (Min Volatility):")
    print("-" * 60)
    min_vol_sorted = sorted(min_vol_dict.items(), key=lambda x: x[1], reverse=True)
    printed_any = False
    for stock, weight in min_vol_sorted:
        if weight > 0.01:  # Only print weights > 1%
            print(f"  {stock:15s}: {weight:6.2%}")
            printed_any = True
    if not printed_any:
        print("  (No weights > 1%)")
    
    print("=" * 60)
    
    return {
        'max_sharpe': {
            'weights': max_sharpe_dict,
            'return': max_sharpe_return,
            'volatility': max_sharpe_vol,
            'sharpe': portfolio_sharpe_ratios[max_sharpe_idx]
        },
        'min_vol': {
            'weights': min_vol_dict,
            'return': min_vol_return,
            'volatility': min_vol_vol,
            'sharpe': portfolio_sharpe_ratios[min_vol_idx]
        }
    }

if __name__ == "__main__":
    efficient_frontier()

