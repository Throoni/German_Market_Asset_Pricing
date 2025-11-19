import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def run_capm_test():
    print("============================================================")
    print("CAPM CROSS-SECTIONAL TEST")
    print("============================================================")

    # 1. Load Data
    print("1. Loading data...")
    try:
        prices = pd.read_csv('german_market_data.csv', index_col=0, parse_dates=True)
        # Remove duplicate columns
        prices = prices.loc[:, ~prices.columns.duplicated()]
        betas = pd.read_csv('beta_results.csv', index_col=0)
    except FileNotFoundError:
        print("Error: Could not find 'german_market_data.csv' or 'beta_results.csv'.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Calculate Average Excess Returns (The 'Y' Variable)
    print("2. Calculating average returns...")
    
    # Calculate Daily Returns
    daily_returns = prices.pct_change()
    
    # Calculate Excess Returns (Return - RF)
    # Note: We must align the RF column with the stock columns
    rf = prices['RF']
    excess_returns = daily_returns.sub(rf, axis=0)
    
    # Drop Market and RF from the stock list
    if 'Market' in excess_returns.columns:
        excess_returns = excess_returns.drop(columns=['Market', 'RF'], errors='ignore')

    # Calculate the Mean Daily Excess Return for each stock
    # multiplying by 252 to annualize it for easier reading in the chart
    avg_excess_return = excess_returns.mean() * 252

    # 3. Merge with Beta (The 'X' Variable)
    # We only keep stocks that exist in both files
    capm_data = pd.DataFrame(avg_excess_return, columns=['Avg_Return'])
    capm_data = capm_data.join(betas['Beta'], how='inner')
    
    print(f"   Matched {len(capm_data)} stocks for the test.")

    # 4. Run Cross-Sectional Regression
    # Equation: Avg_Return = Intercept + Slope * Beta
    print("3. Running regression...")
    
    Y = capm_data['Avg_Return']
    X = capm_data['Beta']
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(Y, X_with_const).fit()
    
    print("\nRegression Results:")
    print(f"Slope (Market Risk Premium): {model.params['Beta']:.4f} (T-stat: {model.tvalues['Beta']:.2f})")
    print(f"Intercept (Alpha):           {model.params['const']:.4f} (T-stat: {model.tvalues['const']:.2f})")
    print(f"R-squared:                   {model.rsquared:.4f}")

    # 5. Plotting the Security Market Line (SML)
    print("4. Generating SML Plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of stocks
    sns.scatterplot(data=capm_data, x='Beta', y='Avg_Return', alpha=0.6)
    
    # Add the Regression Line (The SML)
    x_range = pd.Series([capm_data['Beta'].min(), capm_data['Beta'].max()])
    y_pred = model.params['const'] + model.params['Beta'] * x_range
    plt.plot(x_range, y_pred, color='red', linewidth=2, label='Fitted SML')
    
    # Label outliers (Top 3 and Bottom 3 performers)
    capm_data['Residuals'] = model.resid
    top_performers = capm_data.nlargest(3, 'Residuals')
    worst_performers = capm_data.nsmallest(3, 'Residuals')
    
    for stock in pd.concat([top_performers, worst_performers]).index:
        plt.text(capm_data.loc[stock, 'Beta'], 
                 capm_data.loc[stock, 'Avg_Return'], 
                 stock, fontsize=9, fontweight='bold')

    plt.title('Security Market Line (SML): German Market (2015-2025)', fontsize=14)
    plt.xlabel('Market Beta (Systematic Risk)', fontsize=12)
    plt.ylabel('Annualized Average Excess Return', fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('sml_plot.png')
    print("Saved plot to 'sml_plot.png'")

if __name__ == "__main__":
    run_capm_test()