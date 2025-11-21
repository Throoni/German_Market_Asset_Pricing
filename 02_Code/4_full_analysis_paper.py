"""
German Market Asset Pricing - Full Analysis Pipeline
Paper-Ready Statistical Validation for Master's Thesis

This script performs:
1. Data handling with daily/monthly frequency conversion
2. CAPM time-series and cross-sectional validation
3. Portfolio optimization (monthly)
4. Publication-quality visualizations
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality plotting style
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

def load_and_prepare_data():
    """Load data and create daily/monthly datasets with proper RF handling"""
    print("=" * 80)
    print("STEP 1: DATA LOADING & PREPARATION")
    print("=" * 80)
    
    # Load data
    data_path = Path('01_Data/german_market_data.csv')
    if not data_path.exists():
        data_path = Path('german_market_data.csv')
    
    print(f"\n1.1 Loading data from {data_path}...")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data = data.loc[:, ~data.columns.duplicated()]
    
    print(f"   Loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
    
    # Check for ADS.DE (Adidas)
    if 'ADS.DE' in data.columns:
        print("   ✓ ADS.DE found in dataset")
    else:
        print("   ⚠ ADS.DE not found in dataset (may need to add to fetch list)")
    
    # Verify RF is daily (should be very small values)
    rf_sample = data['RF'].dropna().iloc[0:5]
    print(f"\n1.2 Verifying Risk-Free Rate format...")
    print(f"   Sample RF values: {rf_sample.values}")
    print(f"   RF range: [{data['RF'].min():.6f}, {data['RF'].max():.6f}]")
    if data['RF'].max() > 0.01:
        print("   ⚠ WARNING: RF values seem too large for daily rates!")
    else:
        print("   ✓ RF confirmed as daily rates")
    
    # Create Daily dataset (use as is)
    print("\n1.3 Creating Daily dataset...")
    data_daily = data.copy()
    print(f"   Daily dataset: {data_daily.shape[0]} observations")
    
    # Create Monthly dataset
    print("\n1.4 Creating Monthly dataset...")
    # Resample prices to month-end
    data_monthly = data.resample('ME').last()
    
    # Compound daily RF to monthly: (1+r).prod() - 1 for each month
    print("   Compounding daily RF to monthly rates...")
    rf_monthly = data['RF'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    data_monthly['RF'] = rf_monthly
    
    print(f"   Monthly dataset: {data_monthly.shape[0]} observations")
    print(f"   Monthly RF range: [{data_monthly['RF'].min():.6f}, {data_monthly['RF'].max():.6f}]")
    
    return data_daily, data_monthly

def capm_time_series_regression(returns, rf, market_returns, frequency='Daily'):
    """
    Run CAPM time-series regression for each stock:
    (R_stock - Rf) = alpha + beta * (R_market - Rf)
    """
    print(f"\n{'='*80}")
    print(f"STEP 2: CAPM TIME-SERIES REGRESSION ({frequency.upper()})")
    print("=" * 80)
    
    # Get stock columns (exclude Market and RF)
    stock_columns = [col for col in returns.columns if col not in ['Market', 'RF']]
    
    results = []
    
    for stock in stock_columns:
        try:
            # Create excess returns
            stock_excess = returns[stock] - rf
            market_excess = market_returns - rf
            
            # Align data
            aligned = pd.DataFrame({
                'Stock_Excess': stock_excess,
                'Market_Excess': market_excess
            }).dropna()
            
            # Minimum observations: 60 for daily, 12 for monthly
            min_obs = 60 if frequency == 'Daily' else 12
            if len(aligned) < min_obs:
                continue
            
            # Run regression
            Y = aligned['Stock_Excess']
            X = aligned['Market_Excess']
            X_with_const = sm.add_constant(X)
            
            model = sm.OLS(Y, X_with_const).fit()
            
            # Extract results
            alpha = model.params['const']
            beta = model.params['Market_Excess']
            alpha_tstat = model.tvalues['const']
            beta_tstat = model.tvalues['Market_Excess']
            
            # Test if beta != 1
            beta_test = (beta - 1) / model.bse['Market_Excess']
            beta_pvalue = 2 * (1 - abs(beta_test))  # Two-tailed test approximation
            
            # More precise: use t-test for beta != 1
            from scipy import stats
            beta_pvalue_precise = 2 * (1 - stats.t.cdf(abs(beta_test), model.df_resid))
            
            results.append({
                'Stock': stock,
                'Alpha': alpha,
                'Beta': beta,
                'Alpha_tstat': alpha_tstat,
                'Beta_tstat': beta_tstat,
                'Beta_test_tstat': beta_test,
                'Beta_pvalue_not_one': beta_pvalue_precise,
                'Alpha_pvalue': model.pvalues['const'],
                'Beta_pvalue': model.pvalues['Market_Excess'],
                'R_squared': model.rsquared,
                'N_observations': len(aligned)
            })
            
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    print(f"\n   Completed regressions for {len(results_df)} stocks")
    
    return results_df

def capm_cross_sectional_regression(returns, rf, market_returns, capm_results, frequency='Daily'):
    """
    Cross-sectional regression: Mean_Excess_Return = gamma_0 + gamma_1 * Beta
    Tests if slope (gamma_1) equals realized Market Risk Premium
    """
    print(f"\n{'='*80}")
    print(f"STEP 3: CAPM CROSS-SECTIONAL REGRESSION ({frequency.upper()})")
    print("=" * 80)
    
    # Check if we have any CAPM results
    if len(capm_results) == 0:
        print(f"   ⚠ WARNING: No CAPM results available for {frequency} frequency")
        print("   Skipping cross-sectional regression")
        return {
            'model': None,
            'capm_data': pd.DataFrame(),
            'realized_mrp': np.nan,
            'slope': np.nan,
            'slope_tstat': np.nan,
            'slope_pvalue': np.nan,
            'mrp_test_pvalue': np.nan
        }
    
    # Calculate mean excess returns for each stock
    stock_columns = [col for col in returns.columns if col not in ['Market', 'RF']]
    
    mean_excess_returns = []
    for stock in stock_columns:
        stock_excess = returns[stock] - rf
        mean_excess = stock_excess.mean()
        mean_excess_returns.append({
            'Stock': stock,
            'Mean_Excess_Return': mean_excess
        })
    
    mean_returns_df = pd.DataFrame(mean_excess_returns)
    
    # Merge with betas
    capm_data = mean_returns_df.merge(capm_results[['Stock', 'Beta']], on='Stock', how='inner')
    
    if len(capm_data) == 0:
        print(f"   ⚠ WARNING: No matching stocks between returns and CAPM results")
        return {
            'model': None,
            'capm_data': pd.DataFrame(),
            'realized_mrp': np.nan,
            'slope': np.nan,
            'slope_tstat': np.nan,
            'slope_pvalue': np.nan,
            'mrp_test_pvalue': np.nan
        }
    
    # Annualize if monthly
    if frequency == 'Monthly':
        capm_data['Mean_Excess_Return'] = capm_data['Mean_Excess_Return'] * 12
    
    # Calculate realized Market Risk Premium
    market_excess = market_returns - rf
    realized_mrp = market_excess.mean()
    if frequency == 'Monthly':
        realized_mrp = realized_mrp * 12
    
    print(f"\n   Realized Market Risk Premium: {realized_mrp:.4f} ({'annualized' if frequency == 'Monthly' else 'daily'})")
    
    # Run cross-sectional regression
    Y = capm_data['Mean_Excess_Return']
    X = capm_data['Beta']
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(Y, X_with_const).fit()
    
    intercept = model.params['const']
    slope = model.params['Beta']
    slope_tstat = model.tvalues['Beta']
    slope_pvalue = model.pvalues['Beta']
    
    print(f"\n   Cross-Sectional Regression Results:")
    print(f"   Intercept (gamma_0): {intercept:.6f} (t-stat: {model.tvalues['const']:.3f})")
    print(f"   Slope (gamma_1): {slope:.6f} (t-stat: {slope_tstat:.3f}, p-value: {slope_pvalue:.4f})")
    print(f"   R-squared: {model.rsquared:.4f}")
    
    # Test if slope equals realized MRP
    slope_se = model.bse['Beta']
    mrp_test = (slope - realized_mrp) / slope_se
    from scipy import stats
    mrp_pvalue = 2 * (1 - stats.t.cdf(abs(mrp_test), model.df_resid))
    
    print(f"\n   Test: Slope = Realized MRP?")
    print(f"   Difference: {slope - realized_mrp:.6f}")
    print(f"   t-statistic: {mrp_test:.3f}")
    print(f"   p-value: {mrp_pvalue:.4f}")
    
    return {
        'model': model,
        'capm_data': capm_data,
        'realized_mrp': realized_mrp,
        'slope': slope,
        'slope_tstat': slope_tstat,
        'slope_pvalue': slope_pvalue,
        'mrp_test_pvalue': mrp_pvalue
    }

def portfolio_optimization(returns_monthly, rf_monthly):
    """
    Portfolio optimization using monthly data
    Find: 1) Tangency Portfolio (Max Sharpe), 2) Global Minimum Variance Portfolio
    """
    print(f"\n{'='*80}")
    print("STEP 4: PORTFOLIO OPTIMIZATION (MONTHLY)")
    print("=" * 80)
    
    # Get stock columns
    stock_columns = [col for col in returns_monthly.columns if col not in ['Market', 'RF']]
    stock_returns = returns_monthly[stock_columns].dropna()
    
    # Select stocks with sufficient data
    min_obs = len(stock_returns) * 0.8  # At least 80% of observations
    valid_stocks = stock_returns.columns[stock_returns.isna().sum() < min_obs].tolist()
    stock_returns = stock_returns[valid_stocks].dropna()
    
    print(f"\n   Using {len(valid_stocks)} stocks with sufficient data")
    print(f"   Observations: {len(stock_returns)} months")
    
    # Annualize: Mean * 12, Cov * 12
    mean_returns_annual = stock_returns.mean() * 12
    cov_matrix_annual = stock_returns.cov() * 12
    
    # Get annualized RF (use average of recent months to avoid negative values)
    # Annualize monthly RF: (1 + r_monthly)^12 - 1
    recent_rf = rf_monthly.iloc[-12:].mean()  # Average of last 12 months
    if recent_rf < 0:
        # If negative, use the mean of all positive monthly RF values
        positive_rf = rf_monthly[rf_monthly > 0]
        if len(positive_rf) > 0:
            recent_rf = positive_rf.mean()
        else:
            # Fallback to a reasonable assumption (2.5% annual)
            recent_rf = (1.025**(1/12)) - 1
            print("   ⚠ WARNING: All RF values negative, using 2.5% annual assumption")
    
    latest_rf_annual = (1 + recent_rf)**12 - 1
    print(f"\n   Annualized Risk-Free Rate: {latest_rf_annual:.4f} ({latest_rf_annual*100:.2f}%)")
    
    n_stocks = len(valid_stocks)
    
    # Helper functions for optimization
    def portfolio_return(weights):
        return np.sum(mean_returns_annual * weights)
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
    
    def negative_sharpe(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -(ret - latest_rf_annual) / vol
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: weights between 0 and 1 (long-only)
    bounds = tuple((0, 1) for _ in range(n_stocks))
    
    # Initial guess: equal weights
    initial_weights = np.array([1/n_stocks] * n_stocks)
    
    # 1. Tangency Portfolio (Max Sharpe)
    print("\n   4.1 Optimizing Tangency Portfolio (Max Sharpe)...")
    result_tangency = minimize(negative_sharpe, initial_weights, method='SLSQP',
                               bounds=bounds, constraints=constraints)
    
    tangency_weights = result_tangency.x
    tangency_return = portfolio_return(tangency_weights)
    tangency_vol = portfolio_volatility(tangency_weights)
    tangency_sharpe = (tangency_return - latest_rf_annual) / tangency_vol
    
    print(f"   ✓ Tangency Portfolio:")
    print(f"     Annualized Return: {tangency_return:.4f} ({tangency_return*100:.2f}%)")
    print(f"     Annualized Volatility: {tangency_vol:.4f} ({tangency_vol*100:.2f}%)")
    print(f"     Sharpe Ratio: {tangency_sharpe:.4f}")
    
    # 2. Global Minimum Variance Portfolio
    print("\n   4.2 Optimizing Global Minimum Variance Portfolio...")
    result_gmv = minimize(portfolio_variance, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    
    gmv_weights = result_gmv.x
    gmv_return = portfolio_return(gmv_weights)
    gmv_vol = portfolio_volatility(gmv_weights)
    gmv_sharpe = (gmv_return - latest_rf_annual) / gmv_vol
    
    print(f"   ✓ Global Minimum Variance Portfolio:")
    print(f"     Annualized Return: {gmv_return:.4f} ({gmv_return*100:.2f}%)")
    print(f"     Annualized Volatility: {gmv_vol:.4f} ({gmv_vol*100:.2f}%)")
    print(f"     Sharpe Ratio: {gmv_sharpe:.4f}")
    
    # Create weight dictionaries
    tangency_dict = dict(zip(valid_stocks, tangency_weights))
    gmv_dict = dict(zip(valid_stocks, gmv_weights))
    
    # Print top weights
    print("\n   Top 10 Tangency Portfolio Weights:")
    tangency_sorted = sorted(tangency_dict.items(), key=lambda x: x[1], reverse=True)
    for stock, weight in tangency_sorted[:10]:
        if weight > 0.001:
            print(f"     {stock:15s}: {weight:6.2%}")
    
    print("\n   Top 10 GMV Portfolio Weights:")
    gmv_sorted = sorted(gmv_dict.items(), key=lambda x: x[1], reverse=True)
    for stock, weight in gmv_sorted[:10]:
        if weight > 0.001:
            print(f"     {stock:15s}: {weight:6.2%}")
    
    return {
        'tangency': {
            'weights': tangency_dict,
            'return': tangency_return,
            'volatility': tangency_vol,
            'sharpe': tangency_sharpe
        },
        'gmv': {
            'weights': gmv_dict,
            'return': gmv_return,
            'volatility': gmv_vol,
            'sharpe': gmv_sharpe
        },
        'stocks': valid_stocks,
        'mean_returns': mean_returns_annual,
        'cov_matrix': cov_matrix_annual,
        'rf_annual': latest_rf_annual
    }

def create_visualizations(capm_daily, capm_monthly, csr_daily, csr_monthly, portfolio_results):
    """Create publication-quality visualizations"""
    print(f"\n{'='*80}")
    print("STEP 5: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = Path('03_Results_and_Plots')
    output_dir.mkdir(exist_ok=True)
    
    # 1. SML Plot (Use Monthly if available, otherwise Daily)
    print("\n   5.1 Creating Security Market Line (SML) Plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use monthly if available, otherwise daily
    if csr_monthly['model'] is not None and len(csr_monthly['capm_data']) > 0:
        capm_data = csr_monthly['capm_data']
        model = csr_monthly['model']
        realized_mrp = csr_monthly['realized_mrp']
        freq_label = "Monthly"
    elif csr_daily['model'] is not None and len(csr_daily['capm_data']) > 0:
        capm_data = csr_daily['capm_data']
        model = csr_daily['model']
        realized_mrp = csr_daily['realized_mrp']
        freq_label = "Daily"
    else:
        print("   ⚠ WARNING: No cross-sectional regression data available for plotting")
        plt.close()
        return
    
    # Scatter plot
    ax.scatter(capm_data['Beta'], capm_data['Mean_Excess_Return'], 
              alpha=0.6, s=50, label='Stocks', color='steelblue')
    
    # Regression line (fitted SML)
    beta_range = np.linspace(capm_data['Beta'].min(), capm_data['Beta'].max(), 100)
    fitted_sml = model.params['const'] + model.params['Beta'] * beta_range
    ax.plot(beta_range, fitted_sml, 'r-', linewidth=2, label='Fitted SML', alpha=0.8)
    
    # Theoretical SML (from realized MRP)
    theoretical_sml = realized_mrp * beta_range
    ax.plot(beta_range, theoretical_sml, 'g--', linewidth=2, 
            label=f'Theoretical SML (MRP={realized_mrp:.4f})', alpha=0.8)
    
    ax.set_xlabel(r'$\beta$ (Market Beta)', fontsize=12)
    ylabel = 'Annualized Excess Return' if freq_label == 'Monthly' else 'Daily Excess Return'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Security Market Line (SML): German Market ({freq_label})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sml_plot_paper.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'sml_plot_paper.png'}")
    plt.close()
    
    # 2. Efficient Frontier with CML
    print("\n   5.2 Creating Efficient Frontier with Capital Market Line...")
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Generate efficient frontier
    mean_returns = portfolio_results['mean_returns']
    cov_matrix = portfolio_results['cov_matrix']
    rf = portfolio_results['rf_annual']
    stocks = portfolio_results['stocks']
    
    # Monte Carlo simulation for frontier
    n_portfolios = 10000
    portfolio_returns = []
    portfolio_vols = []
    portfolio_sharpes = []
    
    np.random.seed(42)
    for _ in range(n_portfolios):
        weights = np.random.random(len(stocks))
        weights = weights / weights.sum()
        ret = np.sum(mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - rf) / vol
        portfolio_returns.append(ret)
        portfolio_vols.append(vol)
        portfolio_sharpes.append(sharpe)
    
    # Plot efficient frontier
    scatter = ax.scatter(portfolio_vols, portfolio_returns, c=portfolio_sharpes,
                        cmap='viridis', alpha=0.4, s=10, label='Random Portfolios')
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    
    # Plot optimal portfolios
    tangency = portfolio_results['tangency']
    gmv = portfolio_results['gmv']
    
    ax.scatter(tangency['volatility'], tangency['return'], 
              marker='*', s=500, color='red', edgecolors='black', 
              linewidths=1.5, label='Tangency Portfolio (Max Sharpe)', zorder=5)
    ax.scatter(gmv['volatility'], gmv['return'], 
              marker='D', s=500, color='blue', edgecolors='black', 
              linewidths=1.5, label='Global Min Variance', zorder=5)
    
    # Capital Market Line (CML)
    # CML: E(Rp) = Rf + (E(Rt) - Rf) / σt * σp
    vol_range = np.linspace(0, max(portfolio_vols), 100)
    cml = rf + (tangency['return'] - rf) / tangency['volatility'] * vol_range
    ax.plot(vol_range, cml, 'orange', linewidth=2.5, linestyle='--', 
            label='Capital Market Line (CML)', alpha=0.9)
    
    ax.set_xlabel('Annualized Volatility', fontsize=12)
    ax.set_ylabel('Annualized Return', fontsize=12)
    ax.set_title('Efficient Frontier with Capital Market Line', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficient_frontier_paper.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'efficient_frontier_paper.png'}")
    plt.close()
    
    print("\n   ✓ All visualizations created")

def generate_statistical_summary(capm_daily, capm_monthly, csr_daily, csr_monthly, portfolio_results):
    """Generate statistical summary text file"""
    print(f"\n{'='*80}")
    print("STEP 6: GENERATING STATISTICAL SUMMARY")
    print("=" * 80)
    
    output_dir = Path('03_Results_and_Plots')
    output_file = output_dir / 'statistical_summary.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SUMMARY: GERMAN MARKET ASSET PRICING ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. Alpha significance
        f.write("1. ALPHA SIGNIFICANCE\n")
        f.write("-" * 80 + "\n")
        for freq, capm in [('Daily', capm_daily), ('Monthly', capm_monthly)]:
            if len(capm) > 0:
                significant_alphas = (capm['Alpha_pvalue'] < 0.05).sum()
                total = len(capm)
                pct = (significant_alphas / total * 100) if total > 0 else 0
                f.write(f"{freq:8s}: {significant_alphas:3d} / {total:3d} stocks with significant Alpha (p < 0.05) = {pct:5.2f}%\n")
            else:
                f.write(f"{freq:8s}: No data available\n")
        f.write("\n")
        
        # 2. Cross-sectional regression results
        f.write("2. CROSS-SECTIONAL REGRESSION (SECURITY MARKET LINE)\n")
        f.write("-" * 80 + "\n")
        for freq, csr in [('Daily', csr_daily), ('Monthly', csr_monthly)]:
            f.write(f"\n{freq} Frequency:\n")
            if csr['model'] is not None:
                f.write(f"  Slope (gamma_1): {csr['slope']:.6f}\n")
                f.write(f"  t-statistic: {csr['slope_tstat']:.3f}\n")
                f.write(f"  p-value: {csr['slope_pvalue']:.6f}\n")
                f.write(f"  Realized Market Risk Premium: {csr['realized_mrp']:.6f}\n")
                f.write(f"  Test (Slope = MRP) p-value: {csr['mrp_test_pvalue']:.6f}\n")
                if csr['slope_pvalue'] < 0.05:
                    f.write(f"  ✓ Beta factor is statistically significant (p < 0.05)\n")
                else:
                    f.write(f"  ✗ Beta factor is NOT statistically significant (p >= 0.05)\n")
            else:
                f.write(f"  No data available\n")
        f.write("\n")
        
        # 3. Portfolio optimization results
        f.write("3. PORTFOLIO OPTIMIZATION RESULTS\n")
        f.write("-" * 80 + "\n")
        tangency = portfolio_results['tangency']
        gmv = portfolio_results['gmv']
        
        f.write("\nTangency Portfolio (Max Sharpe Ratio):\n")
        f.write(f"  Annualized Return: {tangency['return']:.6f} ({tangency['return']*100:.2f}%)\n")
        f.write(f"  Annualized Volatility: {tangency['volatility']:.6f} ({tangency['volatility']*100:.2f}%)\n")
        f.write(f"  Sharpe Ratio: {tangency['sharpe']:.6f}\n")
        
        f.write("\nGlobal Minimum Variance Portfolio:\n")
        f.write(f"  Annualized Return: {gmv['return']:.6f} ({gmv['return']*100:.2f}%)\n")
        f.write(f"  Annualized Volatility: {gmv['volatility']:.6f} ({gmv['volatility']*100:.2f}%)\n")
        f.write(f"  Sharpe Ratio: {gmv['sharpe']:.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Statistical Summary\n")
        f.write("=" * 80 + "\n")
    
    print(f"   ✓ Saved: {output_file}")

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("GERMAN MARKET ASSET PRICING - FULL ANALYSIS PIPELINE")
    print("Paper-Ready Statistical Validation")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    data_daily, data_monthly = load_and_prepare_data()
    
    # Calculate returns
    returns_daily = data_daily.pct_change().dropna()
    returns_monthly = data_monthly.pct_change().dropna()
    
    rf_daily = returns_daily['RF']
    rf_monthly = returns_monthly['RF']
    market_daily = returns_daily['Market']
    market_monthly = returns_monthly['Market']
    
    # Step 2: CAPM Time-Series Regression
    capm_daily = capm_time_series_regression(returns_daily, rf_daily, market_daily, 'Daily')
    capm_monthly = capm_time_series_regression(returns_monthly, rf_monthly, market_monthly, 'Monthly')
    
    # Save results
    output_dir = Path('03_Results_and_Plots')
    output_dir.mkdir(exist_ok=True)
    capm_daily.to_csv(output_dir / 'capm_results_daily.csv', index=False)
    capm_monthly.to_csv(output_dir / 'capm_results_monthly.csv', index=False)
    print(f"\n   ✓ Saved: {output_dir / 'capm_results_daily.csv'}")
    print(f"   ✓ Saved: {output_dir / 'capm_results_monthly.csv'}")
    
    # Step 3: Cross-Sectional Regression
    csr_daily = capm_cross_sectional_regression(returns_daily, rf_daily, market_daily, capm_daily, 'Daily')
    csr_monthly = capm_cross_sectional_regression(returns_monthly, rf_monthly, market_monthly, capm_monthly, 'Monthly')
    
    # Step 4: Portfolio Optimization (Monthly only)
    portfolio_results = portfolio_optimization(returns_monthly, rf_monthly)
    
    # Step 5: Visualizations
    create_visualizations(capm_daily, capm_monthly, csr_daily, csr_monthly, portfolio_results)
    
    # Step 6: Statistical Summary
    generate_statistical_summary(capm_daily, capm_monthly, csr_daily, csr_monthly, portfolio_results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - ALL RESULTS SAVED")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  - {output_dir / 'capm_results_daily.csv'}")
    print(f"  - {output_dir / 'capm_results_monthly.csv'}")
    print(f"  - {output_dir / 'sml_plot_paper.png'}")
    print(f"  - {output_dir / 'efficient_frontier_paper.png'}")
    print(f"  - {output_dir / 'statistical_summary.txt'}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

