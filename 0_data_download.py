import yfinance as yf
import pandas as pd
import numpy as np

def download_data():
    print("Downloading historical stock data...")
    
    # Settings
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    market_ticker = '^GDAXI'
    
    # 88 Tickers (DAX + MDAX + SDAX)
    tickers = [
        # DAX
        'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'AIR.DE', 'BMW.DE', 'MBG.DE', 'BAS.DE', 
        'IFX.DE', 'MUV2.DE', 'DHL.DE', 'VOW3.DE', 'DB1.DE', 'RWE.DE', 'EOAN.DE', 'BAYN.DE', 
        'BEI.DE', 'DBK.DE', 'HEI.DE', 'MTX.DE', 'HNR1.DE', 'CON.DE', 'DTG.DE', 'QIAGEN.DE', 
        'SY1.DE', 'FRE.DE', 'CBK.DE', 'ZAL.DE', 'ENR.DE', 'SHL.DE', 'VNA.DE', 'PUM.DE',
        # MDAX
        'LHA.DE', 'LEG.DE', 'KRN.DE', 'G1A.DE', 'FPE.DE', 'CTSA.DE', 'EVT.DE', 'FRA.DE', 
        'GXI.DE', 'HAG.DE', 'HOT.DE', 'KCO.DE', 'KGX.DE', 'MDG1.DE', 'NEM.DE', 'RHM.DE', 
        'RRTL.DE', 'SOW.DE', 'TEG.DE', 'TKA.DE', 'UN01.DE', 'WCH.DE', 'AFX.DE', 'BOSS.DE',
        # SDAX
        'DUE.DE', 'SDF.DE', 'DEZ.DE', 'WAF.DE', 'SIX2.DE', 'S92.DE', 'AOX.DE', 'BVB.DE', 
        'CEV.DE', 'DWO.DE', 'EVD.DE', 'NDX1.DE', 'O2D.DE', 'PAT.DE', 'PBB.DE', 'PFV.DE',
        'PV1.DE', 'QIA.DE', 'RHK.DE', 'SANT.DE', 'SGL.DE', 'STM.DE', 'SZG.DE', 'SZU.DE', 
        'TLX.DE', 'TTK.DE', 'UTDI.DE', 'VAR1.DE', 'WOS.DE', 'WSU.DE', 'ZO1.DE'
    ]

    # 1. Download Stocks (YFinance)
    print("1. Fetching Stocks from Yahoo...")
    all_tickers = [market_ticker] + tickers
    stock_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    
    if market_ticker in stock_data.columns:
        stock_data = stock_data.rename(columns={market_ticker: 'Market'})

    # 2. Download Real Risk-Free Rate (Direct from FRED URL)
    print("2. Fetching German 10Y Bund Yield from FRED...")
    # URL for Series: IRLTLT01DEM156N (Long-Term Govt Bond Yields: 10-year: Main for Germany)
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01DEM156N"
    
    try:
        rf_data = pd.read_csv(fred_url, index_col=0, parse_dates=True)
        # Rename the column to 'RF_Annual'
        rf_data.columns = ['RF_Annual']
        
        # Resample to Daily (Forward Fill) to match stock data
        rf_daily = rf_data.resample('D').ffill()
        
        # Filter to our date range
        rf_daily = rf_daily[(rf_daily.index >= start_date) & (rf_daily.index <= end_date)]
        
        # Convert Annual Percent (e.g., 2.5) to Daily Decimal (e.g., 0.000098)
        # Formula: (1 + r/100)^(1/252) - 1
        rf_daily['RF'] = (1 + rf_daily['RF_Annual'] / 100)**(1/252) - 1
        
        print("   Success: Downloaded Real Rates.")
        
    except Exception as e:
        print(f"   Error downloading FRED data: {e}")
        print("   Fallback: Using 2.5% constant approximation.")
        # Fallback logic just in case
        rf_daily = pd.DataFrame(index=stock_data.index)
        rf_daily['RF'] = (1 + 0.025)**(1/252) - 1

    # 3. Merge
    print("3. Merging Data...")
    # Join stocks with the daily RF column
    final_data = stock_data.join(rf_daily['RF'], how='left')
    
    # Fill any missing RF days (holidays)
    final_data['RF'] = final_data['RF'].ffill().bfill()

    # 4. Clean
    # Drop columns that are mostly empty
    final_data = final_data.dropna(axis=1, thresh=int(len(final_data)*0.1))
    
    # Fill small internal gaps
    final_data = final_data.ffill(limit=5)

    print(f"Download complete.")
    print(f"Shape: {final_data.shape}")
    print(f"Date range: {final_data.index.min().date()} to {final_data.index.max().date()}")
    
    final_data.to_csv('german_market_data.csv')
    print("Saved to 'german_market_data.csv'")

if __name__ == "__main__":
    download_data()
