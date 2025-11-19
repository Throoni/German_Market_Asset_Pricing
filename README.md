# Empirical Analysis of German Asset Pricing (DAX, MDAX, SDAX)

## Project Overview
This repository contains the empirical analysis for the [Course Name] assignment. The project tests standard asset pricing theories (CAPM) on the German equity market.

## Objectives
1.  Estimate Market Betas for stocks in the DAX40, MDAX, and SDAX.
2.  Test the Capital Asset Pricing Model (CAPM) using cross-sectional regression.
3.  Construct an Efficient Frontier and Minimum Variance Portfolio.
4.  Analyze Alpha parameters for value/size effects.

## Data
* **Market Proxy:** DAX 40 Index
* **Risk-Free Rate:** German Bund Yield / Euro Short-Term Rate
* **Stock Universe:** Top components of DAX, MDAX, and SDAX
* **Source:** Yahoo Finance (via yfinance API)

## Methodology
The analysis is conducted using Python.
* `1_beta_estimation.py`: Time-series regressions.
* `2_capm_test.py`: Cross-sectional analysis.
* `3_efficient_frontier.py`: Portfolio optimization.

## Authors
* Messiah Gord
