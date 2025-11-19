import pandas as pd
import numpy as np
import statsmodels.api as sm

def estimate_betas():
    print("============================================================")
    print("BETA ESTIMATION (Robust Version)")
    print("============================================================")

    # 1. Load Data
    print("1. Loading data...")
    try:
        # FIX: Use index_col=0 to automatically use the first column (Dates) as the index
        data = pd.read_csv('german_market_data.csv', index_col=0, parse_dates=True)
        print(f"   Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return

    # 2. IMMEDIATE DATA CLEANING: Remove duplicate columns
    print("2. Cleaning data (removing duplicates)...")
    data = data.loc[:, ~data.columns.duplicated()]
    print(f"   After removing duplicates: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Check for Market and RF
    if 'Market' not in data.columns or 'RF' not in data.columns:
        print("   CRITICAL ERROR: 'Market' or 'RF' column missing.")
        print(f"   Available columns: {data.columns.tolist()}")
        return

    # Calculate Returns
    returns = data.pct_change()
    
    # Handle infinite values
    returns = returns.replace([np.inf, -np.inf], np.nan)

    # Separate Market and RF
    market_returns = returns['Market'].copy()
    rf_returns = returns['RF'].copy()
    
    # Drop Market and RF from the stock list to get just the assets
    stock_returns = returns.drop(columns=['Market', 'RF'])
    
    print(f"   Calculating Excess Returns for {len(stock_returns.columns)} assets...")

    # 3. Run Regressions
    print("3. Running OLS regressions...")
    
    results_list = []

    for stock in stock_returns.columns:
        try:
            # Create a mini dataframe for this specific stock
            # We need to align the Stock, Market, and RF for this specific asset
            df_temp = pd.DataFrame({
                'Stock': stock_returns[stock],
                'Market': market_returns,
                'RF': rf_returns
            })
            
            # Drop rows where THIS stock (or market/RF) has missing data
            # This handles the "Unbalanced Panel" (IPOs/Delistings)
            df_temp = df_temp.dropna()
            
            # Skip if too little data (less than 6 months approx)
            if len(df_temp) < 120:
                continue

            # Calculate Excess Returns
            # Y = Stock - RF
            # X = Market - RF
            Y = (df_temp['Stock'] - df_temp['RF']).rename('Stock_Excess')
            X = (df_temp['Market'] - df_temp['RF']).rename('Market_Excess')
            
            # Add constant (Alpha)
            X_with_const = sm.add_constant(X)
            
            # Run Regression
            model = sm.OLS(Y, X_with_const).fit()
            
            # Check R-squared: If > 0.99, it's a data error (perfect fake match - Market Clone)
            if model.rsquared > 0.99:
                print(f"   Skipping Market Clone: {stock} (R² = {model.rsquared:.4f})")
                continue
            
            # Store Results
            results_list.append({
                'Stock': stock,
                'Beta': model.params['Market_Excess'],
                'Alpha': model.params['const'],
                'Beta_tstat': model.tvalues['Market_Excess'],
                'Beta_pvalue': model.pvalues['Market_Excess'],
                'R_squared': model.rsquared,
                'N_observations': len(df_temp)
            })
            
        except Exception as e:
            # Skip stocks with errors (SVD, insufficient data, etc.)
            print(f"   Skipping {stock}: {type(e).__name__}")
            continue

    print(f"   Completed regressions for {len(results_list)} stocks")

    # 4. Output
    print("4. Preparing output...")
    
    if len(results_list) == 0:
        print("   ERROR: No successful regressions!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Sort by Beta (descending)
    results_df = results_df.sort_values('Beta', ascending=False)
    
    # Save
    results_df.to_csv('beta_results.csv', index=False)
    print(f"   Results saved to 'beta_results.csv'")
    
    # Print Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average Beta: {results_df['Beta'].mean():.2f}")
    print(f"Highest Beta: {results_df.iloc[0]['Stock']} ({results_df.iloc[0]['Beta']:.2f})")
    print(f"Lowest Beta: {results_df.iloc[-1]['Stock']} ({results_df.iloc[-1]['Beta']:.2f})")
    print("=" * 60)
    
    return results_df

if __name__ == "__main__":
    estimate_betas()
