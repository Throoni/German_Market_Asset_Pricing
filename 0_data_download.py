import yfinance as yf
import pandas as pd
import numpy as np

def download_data():
    print("Downloading historical stock data...")
    
    # Settings
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    market_ticker = '^GDAXI'
    
    # DAX (40), MDAX (Selected), SDAX (Selected) - Total 88
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

    # 1. Download Data
    # combining market ticker with stock tickers for one request
    all_tickers = [market_ticker] + tickers
    
    data = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False  # We want Adj Close usually, but auto_adjust=False gives us 'Adj Close' column explicitly
    )['Adj Close']

    # 2. Rename Market Column
    if market_ticker in data.columns:
        data = data.rename(columns={market_ticker: 'Market'})
    
    # 3. Create Risk-Free Rate (Approximation)
    # Create empty Series with same index
    rf_series = pd.Series(index=data.index, dtype=float)
    
    # Logic: 0% before July 2022, 2.5% after
    cutoff_date = pd.Timestamp('2022-07-01').tz_localize(data.index.dtype.tz) if data.index.tz else pd.Timestamp('2022-07-01')
    
    mask_zero = data.index < cutoff_date
    mask_hike = data.index >= cutoff_date
    
    rf_annual_zero = 0.0
    rf_annual_hike = 0.025
    
    # Convert to daily: (1+r)^(1/252) - 1
    rf_daily_zero = (1 + rf_annual_zero)**(1/252) - 1
    rf_daily_hike = (1 + rf_annual_hike)**(1/252) - 1
    
    rf_series[mask_zero] = rf_daily_zero
    rf_series[mask_hike] = rf_daily_hike
    
    data['RF'] = rf_series

    # 4. Clean Data
    # Drop columns that are almost entirely empty (failed downloads)
    data = data.dropna(axis=1, thresh=int(len(data)*0.1))
    
    # Fill small gaps (holidays) but keep leading NaNs (pre-IPO)
    data = data.ffill(limit=5)

    print(f"Download complete.")
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    
    # Save
    data.to_csv('german_market_data.csv')
    print("Saved to 'german_market_data.csv'")

if __name__ == "__main__":
    download_data()
