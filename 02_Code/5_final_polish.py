"""
German Market Asset Pricing - Final Polish
Performs final robustness checks and organizes project files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from pathlib import Path

def final_polish():
    print("=" * 60)
    print("FINAL POLISH: ROBUSTNESS CHECKS & FILE ORGANIZATION")
    print("=" * 60)
    
    # ============================================================================
    # PART 1: FINAL ANALYTICS
    # ============================================================================
    print("\n" + "=" * 60)
    print("PART 1: FINAL ANALYTICS")
    print("=" * 60)
    
    # 1. Load Data
    print("\n1. Loading data...")
    data = pd.read_csv('german_market_data.csv', index_col=0, parse_dates=True)
    data = data.loc[:, ~data.columns.duplicated()]
    
    betas = pd.read_csv('beta_results.csv', index_col=0)
    
    print(f"   Market data: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"   Beta results: {len(betas)} stocks")
    
    # Calculate Returns
    returns = data.pct_change()
    market_returns = returns['Market'].copy()
    
    # ============================================================================
    # 2. Robustness Check (Expanded): Rolling Beta for 3 Stocks
    # ============================================================================
    print("\n2. Calculating Rolling Beta for 3 stocks...")
    
    # Define stocks
    stocks_to_analyze = ['SAP.DE', 'ALV.DE']
    
    # Find one random SDAX stock (small cap - not in DAX)
    # DAX stocks are typically the largest, so we'll pick from stocks not in our main list
    all_stocks = [col for col in returns.columns if col not in ['Market', 'RF']]
    # Remove the two we already have
    remaining_stocks = [s for s in all_stocks if s not in stocks_to_analyze]
    
    # Pick a random SDAX stock (we'll use one that exists in our data)
    if remaining_stocks:
        import random
        random.seed(42)  # For reproducibility
        sdax_stock = random.choice(remaining_stocks)
        stocks_to_analyze.append(sdax_stock)
        print(f"   Selected SDAX stock: {sdax_stock}")
    
    # Function to calculate rolling beta
    def rolling_beta_calc(stock_ret, market_ret, window=252):
        """Calculate rolling beta using rolling covariance/variance"""
        aligned_data = pd.DataFrame({
            'Stock': stock_ret,
            'Market': market_ret
        }).dropna()
        
        def rolling_covariance(x, y, window):
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
        rolling_beta = rolling_cov / rolling_var
        return rolling_beta
    
    # Calculate rolling betas for all 3 stocks
    rolling_betas = {}
    for stock in stocks_to_analyze:
        if stock in returns.columns:
            stock_returns = returns[stock].copy()
            rolling_beta = rolling_beta_calc(stock_returns, market_returns, window=252)
            rolling_betas[stock] = rolling_beta
            print(f"   Calculated rolling beta for {stock}")
        else:
            print(f"   WARNING: {stock} not found in data")
    
    # Plot all 3 lines
    plt.figure(figsize=(14, 6))
    colors = ['steelblue', 'darkgreen', 'coral']
    for i, (stock, rolling_beta) in enumerate(rolling_betas.items()):
        plt.plot(rolling_beta.index, rolling_beta.values, 
                linewidth=1.5, label=stock, color=colors[i % len(colors)])
    
    plt.title('Time-Varying Risk Across Sectors', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rolling Beta (252-Day)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rolling_beta_sector_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: 'rolling_beta_sector_comparison.png'")
    
    # ============================================================================
    # 3. Size Effect Analysis: DAX vs SDAX Beta Comparison
    # ============================================================================
    print("\n3. Analyzing Size Effect (DAX vs SDAX)...")
    
    # Define DAX tickers (Top 40 largest German stocks)
    # We'll use the stocks that have the most complete data as a proxy for DAX
    # In reality, DAX has 40 stocks, but we'll use a heuristic based on data completeness
    nan_counts = returns.drop(columns=['Market', 'RF'], errors='ignore').isna().sum()
    top_40_stocks = nan_counts.nsmallest(40).index.tolist()
    
    # Get betas for DAX and SDAX
    dax_betas = betas[betas.index.isin(top_40_stocks)]['Beta'].dropna()
    sdax_betas = betas[~betas.index.isin(top_40_stocks)]['Beta'].dropna()
    
    print(f"   DAX stocks: {len(dax_betas)}")
    print(f"   SDAX stocks: {len(sdax_betas)}")
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    box_data = [dax_betas.values, sdax_betas.values]
    box_labels = ['DAX (Large Cap)', 'SDAX (Small Cap)']
    
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    
    plt.title('Size Effect: Beta Comparison (DAX vs SDAX)', fontsize=14, fontweight='bold')
    plt.ylabel('Beta', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('size_effect_boxplot.png', dpi=300, bbox_inches='tight')
    print("   Saved: 'size_effect_boxplot.png'")
    
    # ============================================================================
    # 4. Correlation Heatmap
    # ============================================================================
    print("\n4. Generating Correlation Heatmap...")
    
    # Select top stocks for correlation (to keep heatmap readable)
    # Use stocks with most complete data
    stock_returns_clean = returns.drop(columns=['Market', 'RF'], errors='ignore')
    nan_counts = stock_returns_clean.isna().sum()
    top_stocks = nan_counts.nsmallest(20).index.tolist()  # Top 20 for readability
    
    # Calculate correlation matrix
    correlation_matrix = stock_returns_clean[top_stocks].corr()
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Stock Return Correlation Matrix (Top 20 Stocks)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("   Saved: 'correlation_matrix.png'")
    
    print("\n" + "=" * 60)
    print("PART 1 COMPLETE: All analytics finished")
    print("=" * 60)
    
    # ============================================================================
    # PART 2: FILE MANAGEMENT
    # ============================================================================
    print("\n" + "=" * 60)
    print("PART 2: FILE MANAGEMENT")
    print("=" * 60)
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    dirs = ['01_Data', '02_Code', '03_Results_and_Plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   Created: {dir_name}/")
    
    # Move files
    print("\n2. Moving files...")
    
    # Move CSV data files
    if os.path.exists('german_market_data.csv'):
        shutil.move('german_market_data.csv', '01_Data/german_market_data.csv')
        print("   Moved: german_market_data.csv -> 01_Data/")
    
    # Move results CSV
    if os.path.exists('beta_results.csv'):
        shutil.move('beta_results.csv', '03_Results_and_Plots/beta_results.csv')
        print("   Moved: beta_results.csv -> 03_Results_and_Plots/")
    
    # Move Python files (except this one, which we'll move at the end)
    py_files = [f for f in os.listdir('.') if f.endswith('.py') and f != '5_final_polish.py']
    for py_file in py_files:
        if os.path.exists(py_file):
            shutil.move(py_file, f'02_Code/{py_file}')
            print(f"   Moved: {py_file} -> 02_Code/")
    
    # Move PNG images
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    for png_file in png_files:
        if os.path.exists(png_file):
            shutil.move(png_file, f'03_Results_and_Plots/{png_file}')
            print(f"   Moved: {png_file} -> 03_Results_and_Plots/")
    
    # Move requirements.txt
    if os.path.exists('requirements.txt'):
        shutil.move('requirements.txt', '02_Code/requirements.txt')
        print("   Moved: requirements.txt -> 02_Code/")
    
    # Keep README.md in root (or move to 02_Code if preferred)
    # We'll keep it in root as that's standard practice
    if os.path.exists('README.md'):
        print("   Kept: README.md in root directory")
    
    # Move this script last
    if os.path.exists('5_final_polish.py'):
        shutil.move('5_final_polish.py', '02_Code/5_final_polish.py')
        print("   Moved: 5_final_polish.py -> 02_Code/")
    
    # ============================================================================
    # PART 3: FINAL MESSAGE
    # ============================================================================
    print("\n" + "=" * 60)
    print("PROJECT ORGANIZED SUCCESSFULLY")
    print("=" * 60)
    print("\nDirectory Structure:")
    print("  /01_Data/")
    print("    - german_market_data.csv")
    print("\n  /02_Code/")
    print("    - 0_data_download.py")
    print("    - 1_beta_estimation.py")
    print("    - 2_capm_test.py")
    print("    - 3_efficient_frontier.py")
    print("    - 4_robustness_checks.py")
    print("    - 5_final_polish.py")
    print("    - requirements.txt")
    print("\n  /03_Results_and_Plots/")
    print("    - beta_results.csv")
    print("    - All .png visualization files")
    print("\n  / (Root)")
    print("    - README.md")
    print("    - main.ipynb (if exists)")
    print("=" * 60)

if __name__ == "__main__":
    final_polish()

