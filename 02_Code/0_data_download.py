import yfinance as yf
import pandas as pd
import numpy as np

def download_data():
    print("Downloading historical stock data (Safe Mode)...")
    
    # Settings
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    
    # 1. Download Market Index SEPARATELY (To ensure it exists)
    print("1. Fetching Market Index (^GDAXI)...")
    market_data = yf.download(['^GDAXI'], start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # Handle MultiIndex (yfinance quirk)
    if isinstance(market_data.columns, pd.MultiIndex):
        try:
            market_data = market_data['Adj Close']
        except KeyError:
            market_data = market_data['Close']
            
    # Rename to 'Market'
    if '^GDAXI' in market_data.columns:
        market_data = market_data.rename(columns={'^GDAXI': 'Market'})
    elif len(market_data.columns) == 1:
        market_data.columns = ['Market']
        
    # Sanity Check 1
    if 'Market' not in market_data.columns:
        print("WARNING: Market index download failed. Using dummy data for testing.")
        market_data = pd.DataFrame({'Market': 10000}, index=pd.date_range(start_date, end_date))

    # 2. Download Stocks
    print("2. Fetching 88 Stocks...")
    tickers = [
        'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'AIR.DE', 'BMW.DE', 'MBG.DE', 'BAS.DE', 
        'IFX.DE', 'MUV2.DE', 'DHL.DE', 'VOW3.DE', 'DB1.DE', 'RWE.DE', 'EOAN.DE', 'BAYN.DE', 
        'BEI.DE', 'DBK.DE', 'HEI.DE', 'MTX.DE', 'HNR1.DE', 'CON.DE', 'DTG.DE', 'QIAGEN.DE', 
        'SY1.DE', 'FRE.DE', 'CBK.DE', 'ZAL.DE', 'ENR.DE', 'SHL.DE', 'VNA.DE', 'PUM.DE',
        'LHA.DE', 'LEG.DE', 'KRN.DE', 'G1A.DE', 'FPE.DE', 'CTSA.DE', 'EVT.DE', 'FRA.DE', 
        'GXI.DE', 'HAG.DE', 'HOT.DE', 'KCO.DE', 'KGX.DE', 'MDG1.DE', 'NEM.DE', 'RHM.DE', 
        'RRTL.DE', 'SOW.DE', 'TEG.DE', 'TKA.DE', 'UN01.DE', 'WCH.DE', 'AFX.DE', 'BOSS.DE',
        'DUE.DE', 'SDF.DE', 'DEZ.DE', 'WAF.DE', 'SIX2.DE', 'S92.DE', 'AOX.DE', 'BVB.DE', 
        'CEV.DE', 'DWO.DE', 'EVD.DE', 'NDX1.DE', 'O2D.DE', 'PAT.DE', 'PBB.DE', 'PFV.DE',
        'PV1.DE', 'QIA.DE', 'RHK.DE', 'SANT.DE', 'SGL.DE', 'STM.DE', 'SZG.DE', 'SZU.DE', 
        'TLX.DE', 'TTK.DE', 'UTDI.DE', 'VAR1.DE', 'WOS.DE', 'WSU.DE', 'ZO1.DE'
    ]
    
    stock_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # Handle MultiIndex again
    if isinstance(stock_data.columns, pd.MultiIndex):
        try:
            stock_data = stock_data['Adj Close']
        except KeyError:
            stock_data = stock_data['Close']

    # 3. Download Risk-Free Rate (Direct from FRED URL)
    print("3. Fetching German 10Y Bund Yield from FRED...")
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01DEM156N"
    
    try:
        rf_data = pd.read_csv(fred_url, index_col=0, parse_dates=True)
        rf_data.columns = ['RF_Annual']
        rf_daily = rf_data.resample('D').ffill()
        # Filter to date range
        rf_daily = rf_daily[(rf_daily.index >= start_date) & (rf_daily.index <= end_date)]
        # Convert Annual % to Daily Decimal
        rf_daily['RF'] = (1 + rf_daily['RF_Annual'] / 100)**(1/252) - 1
        print("   Success: Downloaded Real Rates.")
    except Exception as e:
        print(f"   Error with FRED: {e}. Using constant 2.5%.")
        rf_daily = pd.DataFrame(index=market_data.index)
        rf_daily['RF'] = (1 + 0.025)**(1/252) - 1

    # 4. Merge Everything
    print("4. Merging Data...")
    final_data = pd.concat([stock_data, market_data, rf_daily['RF']], axis=1)
    
    # Fill RF gaps
    final_data['RF'] = final_data['RF'].ffill().bfill()
    
    # 5. Clean
    # Drop columns that are >90% empty (failed downloads)
    final_data = final_data.dropna(axis=1, thresh=int(len(final_data)*0.1))
    # Fill small internal gaps
    final_data = final_data.ffill(limit=5)
    
    print(f"Download complete.")
    print(f"Shape: {final_data.shape}")
    final_data.to_csv('german_market_data.csv')
    print("Saved to 'german_market_data.csv'")

if __name__ == "__main__":
    download_data()
