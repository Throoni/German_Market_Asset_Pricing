"""
German Market Asset Pricing - Beta Estimation Script (Robust Version)
Calculates Market Beta for every stock using OLS regression with error handling
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

def estimate_betas():
    print("=" * 60)
    print("BETA ESTIMATION (Robust Version)")
    print("=" * 60)
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("\n1. Loading data...")
    data = pd.read_csv('german_market_data.csv', index_col='Date', parse_dates=True)
    
    # Ensure index is Datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    print(f"   Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # ============================================================================
    # 2. PRE-CLEANING
    # ============================================================================
    print("\n2. Pre-cleaning data...")
    
    # Calculate returns using pct_change()
    returns = data.pct_change()
    
    # Replace all infinite values with np.nan
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Drop the 'RF' column from the stock list (but keep it for calculation)
    rf_series = returns['RF'].copy()
    stock_columns = [col for col in returns.columns if col not in ['RF', 'Market']]
    market_returns = returns['Market'].copy()
    
    print(f"   Processing {len(stock_columns)} stocks")
    
    # ============================================================================
    # 3. THE ROBUST LOOP
    # ============================================================================
    print("\n3. Running OLS regressions...")
    
    results = []
    skipped_count = 0
    
    # Loop through each stock
    for stock in stock_columns:
        # Construct temporary dataframe with columns: ['Y', 'X', 'RF']
        df_temp = pd.DataFrame({
            'Y': returns[stock],
            'X': market_returns,
            'RF': rf_series
        })
        
        # Calculate Excess Returns inside this loop
        Y_excess = df_temp['Y'] - df_temp['RF']
        X_excess = df_temp['X'] - df_temp['RF']
        
        # Update df_temp with excess returns
        df_temp['Y_excess'] = Y_excess
        df_temp['X_excess'] = X_excess
        
        # CRITICAL STEP: Apply dropna() to remove any row where any of these values are missing
        df_temp = df_temp[['Y_excess', 'X_excess']].dropna()
        
        # Check length: If less than 60 observations, skip the stock
        if len(df_temp) < 60:
            skipped_count += 1
            continue
        
        # ============================================================================
        # 4. REGRESSION WITH ERROR HANDLING
        # ============================================================================
        try:
            # Define Y and X for regression
            Y = df_temp['Y_excess']
            X = df_temp['X_excess']
            
            # Add constant to X (for Alpha)
            X_with_const = sm.add_constant(X)
            
            # Run OLS Regression
            model = sm.OLS(Y, X_with_const).fit()
            
            # Extract results
            alpha = model.params['const']
            beta = model.params['X_excess']
            beta_tstat = model.tvalues['X_excess']
            beta_pvalue = model.pvalues['X_excess']
            r_squared = model.rsquared
            
            # Append to results
            results.append({
                'Stock': stock,
                'Beta': beta,
                'Alpha': alpha,
                'Beta_tstat': beta_tstat,
                'Beta_pvalue': beta_pvalue,
                'R_squared': r_squared,
                'N_observations': len(df_temp)
            })
            
        except Exception as e:
            # If regression fails (SVD error or other math error), skip
            print(f"   Skipping {stock} due to math error: {type(e).__name__}")
            skipped_count += 1
            continue
    
    print(f"   Completed regressions for {len(results)} stocks")
    if skipped_count > 0:
        print(f"   Skipped {skipped_count} stocks (insufficient data or errors)")
    
    # ============================================================================
    # 5. OUTPUT
    # ============================================================================
    print("\n4. Preparing output...")
    
    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Beta (descending)
    results_df = results_df.sort_values('Beta', ascending=False)
    
    # Save successful results to CSV
    results_df.to_csv('beta_results.csv', index=False)
    print(f"   Results saved to 'beta_results.csv'")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(results_df) > 0:
        print(f"\nAverage Beta: {results_df['Beta'].mean():.2f}")
        
        print("\nTop 5 Highest Betas:")
        for idx, row in results_df.head(5).iterrows():
            print(f"  {row['Stock']}: {row['Beta']:.2f}")
        
        print("\nTop 5 Lowest Betas:")
        for idx, row in results_df.tail(5).iterrows():
            print(f"  {row['Stock']}: {row['Beta']:.2f}")
    else:
        print("No successful regressions completed.")
    
    print("=" * 60)
    
    return results_df

if __name__ == "__main__":
    estimate_betas()
