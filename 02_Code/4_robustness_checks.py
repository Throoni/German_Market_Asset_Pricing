"""
German Market Asset Pricing - Robustness Checks
Creates rolling beta analysis and performance backtest charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def robustness_checks():
    print("=" * 60)
    print("ROBUSTNESS CHECKS")
    print("=" * 60)
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("\n1. Loading data...")
    data = pd.read_csv('german_market_data.csv', index_col=0, parse_dates=True)
    
    # Remove duplicate columns
    data = data.loc[:, ~data.columns.duplicated()]
    
    # Calculate Returns
    returns = data.pct_change()
    
    # Separate Market returns
    market_returns = returns['Market'].copy()
    
    print(f"   Data loaded: {returns.shape[0]} rows, {returns.shape[1]} columns")
    
    # ============================================================================
    # 2. ANALYSIS A: ROLLING BETA (TIME-VARYING RISK)
    # ============================================================================
    print("\n2. Analysis A: Calculating Rolling Beta for SAP.DE...")
    
    # Select SAP.DE (Largest German Stock)
    stock_ticker = 'SAP.DE'
    
    if stock_ticker not in returns.columns:
        print(f"   ERROR: {stock_ticker} not found in data")
        return
    
    stock_returns = returns[stock_ticker].copy()
    
    # Calculate Rolling 252-Day Beta (1 Year window)
    # Formula: Rolling_Covariance(Stock, Market) / Rolling_Variance(Market)
    window = 252
    
    # Align stock and market returns
    aligned_data = pd.DataFrame({
        'Stock': stock_returns,
        'Market': market_returns
    }).dropna()
    
    # Calculate rolling covariance and variance
    # For rolling covariance, we need to use a custom function
    def rolling_covariance(x, y, window):
        """Calculate rolling covariance between two series"""
        result = []
        for i in range(len(x)):
            if i < window - 1:
                result.append(np.nan)
            else:
                x_window = x.iloc[i-window+1:i+1]
                y_window = y.iloc[i-window+1:i+1]
                cov = x_window.cov(y_window)
                result.append(cov)
        return pd.Series(result, index=x.index)
    
    rolling_cov = rolling_covariance(aligned_data['Stock'], aligned_data['Market'], window)
    rolling_var = aligned_data['Market'].rolling(window=window).var()
    
    # Calculate rolling beta
    rolling_beta = rolling_cov / rolling_var
    
    # Calculate static beta (average of rolling betas, excluding NaN)
    static_beta = rolling_beta.dropna().mean()
    
    print(f"   Static Beta (Average): {static_beta:.4f}")
    
    # Plot Rolling Beta
    plt.figure(figsize=(14, 6))
    plt.plot(rolling_beta.index, rolling_beta.values, linewidth=1.5, label='Rolling Beta (252-Day)', color='steelblue')
    plt.axhline(y=static_beta, color='red', linestyle='--', linewidth=2, label=f'Static Beta (Avg: {static_beta:.2f})')
    plt.title('Time-Varying Risk: SAP.DE Rolling Beta (12-Month)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Beta', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rolling_beta.png', dpi=300, bbox_inches='tight')
    print("   Saved: 'rolling_beta.png'")
    
    # ============================================================================
    # 3. ANALYSIS B: STRATEGY BACKTEST (PERFORMANCE)
    # ============================================================================
    print("\n3. Analysis B: Running Performance Backtest...")
    
    # Define the "Optimal Portfolio" using equal weights of Top 5 Sharpe Ratio stocks
    # Based on user specification: RHM.DE, HAG.DE, ENR.DE, NEM.DE, TLX.DE
    optimal_stocks = ['RHM.DE', 'HAG.DE', 'ENR.DE', 'NEM.DE', 'TLX.DE']
    
    # Check which stocks are available
    available_stocks = [s for s in optimal_stocks if s in returns.columns]
    
    if len(available_stocks) < 3:
        print(f"   WARNING: Only {len(available_stocks)} stocks available. Using available stocks.")
    
    if len(available_stocks) == 0:
        print("   ERROR: None of the specified stocks are available")
        return
    
    print(f"   Using {len(available_stocks)} stocks: {', '.join(available_stocks)}")
    
    # Calculate daily portfolio return: (Ret_Stock1 + ... + Ret_StockN) / N
    portfolio_returns = returns[available_stocks].mean(axis=1)
    
    # Calculate Cumulative Returns: (1 + Returns).cumprod() * 100
    portfolio_cumulative = (1 + portfolio_returns).cumprod() * 100
    market_cumulative = (1 + market_returns).cumprod() * 100
    
    # Plot Performance Backtest
    plt.figure(figsize=(14, 6))
    plt.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
             linewidth=2, label='Optimal Portfolio', color='steelblue')
    plt.plot(market_cumulative.index, market_cumulative.values, 
             linewidth=2, label='DAX (Market)', color='grey', linestyle='--')
    plt.title('Performance Backtest: Optimal Portfolio vs DAX', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Index = 100)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backtest_plot.png', dpi=300, bbox_inches='tight')
    print("   Saved: 'backtest_plot.png'")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Portfolio Final Value: {portfolio_cumulative.iloc[-1]:.2f}")
    print(f"Market Final Value: {market_cumulative.iloc[-1]:.2f}")
    print(f"Outperformance: {portfolio_cumulative.iloc[-1] - market_cumulative.iloc[-1]:.2f} points")
    print("=" * 60)
    
    print("\nRobustness charts created.")

if __name__ == "__main__":
    robustness_checks()

