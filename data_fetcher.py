"""
FILE: data_fetcher.py
Purpose:
Fetches options data and extracts trading signals for biotech companies ahead of key events.

Main Features:
Connects to the Polygon.io API with error handling and rate limits
Retrieves full options chains including implied volatility (IV), Greeks, open interest, etc.
Uses a list of known biotech catalyst events (e.g., FDA decisions, trial data releases)
Calculates 10 types of trading signals from the raw data (e.g., IV skew, unusual volume, call/put ratio)
Supports both historical and forward-looking event analysis
Can compute implied volatility using Black-Scholes if not provided by the API

Role in System:
This is the core data pipeline component, feeding data into signal_calculator.py for computations and analysis.py for visualization.

Copyright (c) 2024 Nathan Dougherty
All rights reserved.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon import RESTClient
from dotenv import load_dotenv
import logging
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
import re
import json
import time
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Event configuration: ticker, event name, event date, type (historical/future)
EVENTS = [
    # Future-looking events
    {"ticker": "NUVB", "event": "Nuvation Bio PDUFA (NSCLC)", "date": "2025-06-23", "type": "future"},
    {"ticker": "CAPR", "event": "Capricor Therapeutics PDUFA (Duchenne MD)", "date": "2025-08-31", "type": "future"},
    {"ticker": "ATYR", "event": "Atyr Pharma Phase 3 (sarcoidosis)", "date": "2025-09-30", "type": "future"},
    # Historical events (examples, adjust tickers as needed)
    {"ticker": "SLDB", "event": "SLDB Duchenne MD Phase 1/2", "date": "2025-02-15", "type": "historical"},
    {"ticker": "PEPG", "event": "PEPG Myotonic Dystrophy 1 Phase 1/2", "date": "2025-02-15", "type": "historical"},
    {"ticker": "PEPG", "event": "PEPG data release", "date": "2024-01-15", "type": "historical"},
    {"ticker": "AKRO", "event": "AKRO data release", "date": "2025-01-15", "type": "historical"},
    {"ticker": "NOVOB", "event": "NOVOB obesity readout", "date": "2025-12-15", "type": "historical"},
    {"ticker": "RNA", "event": "RNA data announcement", "date": "2024-06-15", "type": "historical"},
    {"ticker": "RGNX", "event": "RGNX data announcement", "date": "2024-11-15", "type": "historical"},
    {"ticker": "RGNX", "event": "RGNX data announcement", "date": "2025-03-15", "type": "historical"},
    {"ticker": "ALNY", "event": "ALNY data announcement", "date": "2024-06-15", "type": "historical"},
    {"ticker": "ALNY", "event": "ALNY FDA announcement", "date": "2025-03-15", "type": "historical"},
    {"ticker": "BBIO", "event": "BBIO data announcement", "date": "2024-09-15", "type": "historical"},
    {"ticker": "BBIO", "event": "BBIO FDA decision", "date": "2024-11-15", "type": "historical"},
    {"ticker": "QURE", "event": "QURE regulatory update", "date": "2024-07-15", "type": "historical"},
    {"ticker": "QURE", "event": "QURE regulatory update", "date": "2024-12-15", "type": "historical"},
    {"ticker": "QURE", "event": "QURE regulatory update", "date": "2025-04-15", "type": "historical"},
]

# Helper function to get 2-week intervals for 3 months before event
def get_intervals(event_date_str):
    event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
    intervals = []
    for i in range(6, 0, -1):  # 6 intervals, 2 weeks apart
        interval_date = event_date - timedelta(weeks=2*i)
        intervals.append(interval_date.strftime("%Y-%m-%d"))
    return intervals

# connects to Polygon.IO API
class OptionsDataFetcher:
    def __init__(self):
        """Initialize the OptionsDataFetcher with Polygon.io API client."""
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
        self.client = RESTClient(self.api_key)
        self.rate_limit_delay = 0.2  # 200ms delay between requests
        self.max_retries = 3

    def _make_api_call(self, func, *args, **kwargs):
        """Helper method to make API calls with rate limiting and retries."""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)  # Rate limiting
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                raise e

    def get_options_chain(self, ticker: str, expiration_date: str) -> pd.DataFrame:
        """
        Fetch options chain for a given ticker and expiration date, including IV, OI, and Greeks.
        """
        try:
            logger.info(f"Fetching options chain for {ticker} with expiration {expiration_date}")
            options = list(self._make_api_call(
                self.client.list_options_contracts,
                underlying_ticker=ticker,
                expiration_date=expiration_date,
                limit=1000
            ))
            logger.info(f"Received {len(options)} options contracts")
            options_data = []
            for option in options:
                try:
                    # Use snapshot endpoint to get IV, OI, Greeks
                    snapshot = self._make_api_call(
                        self.client.get_option_snapshot,
                        option.ticker
                    )
                    options_data.append({
                        'ticker': option.ticker,
                        'strike_price': option.strike_price,
                        'expiration_date': option.expiration_date,
                        'option_type': option.contract_type,
                        'underlying_ticker': option.underlying_ticker,
                        'implied_volatility': getattr(snapshot, 'implied_volatility', None),
                        'open_interest': getattr(snapshot, 'open_interest', None),
                        'delta': getattr(snapshot.greeks, 'delta', None) if getattr(snapshot, 'greeks', None) else None,
                        'gamma': getattr(snapshot.greeks, 'gamma', None) if getattr(snapshot, 'greeks', None) else None,
                        'theta': getattr(snapshot.greeks, 'theta', None) if getattr(snapshot, 'greeks', None) else None,
                        'vega': getattr(snapshot.greeks, 'vega', None) if getattr(snapshot, 'greeks', None) else None,
                    })
                except Exception as e:
                    logger.warning(f"Error fetching snapshot for {option.ticker}: {str(e)}")
                    options_data.append({
                        'ticker': option.ticker,
                        'strike_price': option.strike_price,
                        'expiration_date': option.expiration_date,
                        'option_type': option.contract_type,
                        'underlying_ticker': option.underlying_ticker,
                        'implied_volatility': None,
                        'open_interest': None,
                        'delta': None,
                        'gamma': None,
                        'theta': None,
                        'vega': None,
                    })
            df = pd.DataFrame(options_data)
            logger.info(f"Created DataFrame with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error fetching options chain for {ticker}: {str(e)}")
            logger.exception("Full traceback:")
            return pd.DataFrame()

    def calculate_signals(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """
        Calculate various options signals from the options chain data.
        If current_price is None, skip price-dependent signals.
        """
        try:
            signals = {}
            # Filter for ATM options (within 5% of current price)
            if current_price:
                atm_range = (current_price * 0.95, current_price * 1.05)
                atm_options = df[
                    (df['strike_price'] >= atm_range[0]) & 
                    (df['strike_price'] <= atm_range[1])
                ]
                atm_iv = atm_options['implied_volatility'].mean() if 'implied_volatility' in atm_options else None
                signals['ATM_IV'] = atm_iv
            else:
                signals['ATM_IV'] = None
            # IV percentile
            all_ivs = df['implied_volatility'].dropna() if 'implied_volatility' in df else []
            signals['IV_percentile'] = np.percentile(all_ivs, 50) if len(all_ivs) > 0 else None
            # Implied move
            if current_price and signals['ATM_IV']:
                signals['Implied_Move'] = current_price * signals['ATM_IV'] * np.sqrt(30/365)
            else:
                signals['Implied_Move'] = None
            # IV skew
            if current_price:
                otm_calls = df[(df['option_type'] == 'call') & (df['strike_price'] > current_price)]
                otm_puts = df[(df['option_type'] == 'put') & (df['strike_price'] < current_price)]
                signals['IV_Skew'] = (
                    otm_calls['implied_volatility'].mean() - 
                    otm_puts['implied_volatility'].mean()
                ) if len(otm_calls) > 0 and len(otm_puts) > 0 else None
            else:
                signals['IV_Skew'] = None
            # Call/Put volume ratio
            call_volume = df[df['option_type'] == 'call']['volume'].sum() if 'volume' in df and 'option_type' in df else None
            put_volume = df[df['option_type'] == 'put']['volume'].sum() if 'volume' in df and 'option_type' in df else None
            signals['CallPut_Volume_Ratio'] = call_volume / put_volume if put_volume and put_volume > 0 else None
            # Risk reversal, unusual options volume, OI, etc. (leave as before)
            signals['Risk_Reversal'] = None
            signals['Unusual_Options_Volume'] = None
            signals['Total_OI'] = df['open_interest'].sum() if 'open_interest' in df and not df['open_interest'].isnull().all() else None
            signals['Max_OI'] = df['open_interest'].max() if 'open_interest' in df and not df['open_interest'].isnull().all() else None
            signals['Calendar_Spread'] = None
            signals['Large_Trades_Sweeps'] = None
            return signals
        except Exception as e:
            logger.error(f"Error calculating signals: {str(e)}")
            return {}

    def get_current_price(self, ticker: str) -> float:
        """Get current stock price. Try Polygon.io first, then Yahoo Finance. If not available, return None and log a warning."""
        try:
            last_trade = self._make_api_call(
                self.client.get_last_trade,
                ticker
            )
            logger.info(f"Fetched current price for {ticker} from Polygon.io: {last_trade.price}")
            return last_trade.price
        except Exception as e:
            logger.warning(f"Could not get current price for {ticker} from Polygon.io. Trying Yahoo Finance. Error: {str(e)}")
            try:
                import yfinance as yf
                price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                logger.info(f"Fetched current price for {ticker} from Yahoo Finance: {price}")
                return price
            except Exception as yf_e:
                logger.warning(f"Could not get current price for {ticker} from Yahoo Finance: {str(yf_e)}")
                return None

    def _calculate_implied_volatility(self, strike_price: float, stock_price: float, option_type: str, 
                                    time_to_expiry: float, risk_free_rate: float = 0.05) -> float:
        """
        Calculate implied volatility using Black-Scholes model.
        
        Args:
            strike_price (float): Option strike price
            stock_price (float): Current stock price
            option_type (str): 'call' or 'put'
            time_to_expiry (float): Time to expiration in years
            risk_free_rate (float): Risk-free interest rate (default: 5%)
            
        Returns:
            float: Implied volatility estimate
        """
        try:
            # Initial guess for volatility
            sigma = 0.3
            
            # Maximum number of iterations
            max_iter = 100
            tolerance = 0.0001
            
            for i in range(max_iter):
                # Calculate d1 and d2
                d1 = (np.log(stock_price/strike_price) + (risk_free_rate + 0.5 * sigma**2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
                d2 = d1 - sigma * np.sqrt(time_to_expiry)
                
                if option_type.lower() == 'call':
                    price = stock_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
                    vega = stock_price * np.sqrt(time_to_expiry) * norm.pdf(d1)
                else:  # put
                    price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
                    vega = stock_price * np.sqrt(time_to_expiry) * norm.pdf(d1)
                
                # Newton-Raphson method
                diff = price - stock_price
                if abs(diff) < tolerance:
                    return sigma
                
                sigma = sigma - diff/vega
                
                # Ensure sigma stays within reasonable bounds
                sigma = max(0.001, min(5.0, sigma))
            
            return sigma
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {str(e)}")
            return 0.3  # Return default value in case of error

    def get_implied_volatility(self, ticker: str, expiration_date: str) -> pd.DataFrame:
        """
        Calculate implied volatility for options of a given ticker and expiration date.
        """
        try:
            # Get options chain
            options_chain = self.get_options_chain(ticker, expiration_date)
            if options_chain.empty:
                logger.warning("No options chain data available. This might be due to API subscription limitations.")
                return pd.DataFrame()
            try:
                # Use get_aggs to get previous close
                today = datetime.now().strftime('%Y-%m-%d')
                aggs = self.client.get_aggs(ticker, 1, 'day', today, today, limit=1)
                if hasattr(aggs, 'results') and aggs.results:
                    stock_price = aggs.results[0].c
                else:
                    logger.error("No previous close data available for stock price.")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error fetching stock price: {str(e)}")
                return pd.DataFrame()
            # Calculate time to expiration in years
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            current_date = datetime.now()
            time_to_expiry = (exp_date - current_date).days / 365.0
            # Calculate implied volatility for each option
            options_chain['implied_volatility'] = options_chain.apply(
                lambda x: self._calculate_implied_volatility(
                    x['strike_price'],
                    stock_price,
                    x['option_type'],
                    time_to_expiry
                ),
                axis=1
            )
            # Add additional useful columns
            options_chain['moneyness'] = options_chain['strike_price'] / stock_price
            options_chain['days_to_expiry'] = (exp_date - current_date).days
            return options_chain
        except Exception as e:
            if "NOT_AUTHORIZED" in str(e):
                logger.error("API subscription does not include access to options data. Please upgrade your Polygon.io subscription.")
            else:
                logger.error(f"Error calculating implied volatility for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_available_expiration_dates(self, ticker: str) -> list:
        """
        List all available expiration dates for a given ticker.
        Args:
            ticker (str): Stock ticker symbol
        Returns:
            list: List of unique expiration dates (as strings)
        """
        try:
            logger.info(f"Fetching available expiration dates for {ticker}")
            # Get current date
            current_date = datetime.now().strftime('%Y-%m-%d')
            # Get date 1 year from now
            future_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Fetch only a small sample of contracts to get expiration dates
            options = self.client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date_gte=current_date,
                expiration_date_lte=future_date,
                limit=100  # Reduced limit since we only need a sample
            )
            
            expiration_dates = set()
            for option in options:
                expiration_dates.add(option.expiration_date)
            sorted_dates = sorted(expiration_dates)
            logger.info(f"Found {len(sorted_dates)} unique expiration dates for {ticker}")
            return sorted_dates
        except Exception as e:
            logger.error(f"Error fetching expiration dates for {ticker}: {str(e)}")
            logger.exception("Full traceback:")
            return []

    def get_option_metrics(self, ticker: str, expiration_date: str) -> pd.DataFrame:
        """
        Fetch options chain and add open interest and volume if available.
        """
        df = self.get_options_chain(ticker, expiration_date)
        if df.empty:
            return df
        # If OI is present in the API response, it will be in the DataFrame already
        return df

def safe_filename(s):
    return re.sub(r'[^\w\-_\. ]', '_', s)

def compute_signals(df, historical_ivs=None):
    signals = {}
    # 1. Absolute IV (mean ATM IV)
    if 'implied_volatility' in df.columns and 'strike_price' in df.columns and not df.empty:
        # ATM = strike closest to moneyness 1
        df['atm_dist'] = abs(df['moneyness'] - 1)
        atm = df.loc[df['atm_dist'].idxmin()]
        signals['ATM_IV'] = atm['implied_volatility']
    else:
        signals['ATM_IV'] = None
    # 2. IV Percentile / Rank (vs. historical)
    if historical_ivs is not None and signals['ATM_IV'] is not None:
        signals['IV_percentile'] = (historical_ivs < signals['ATM_IV']).mean() * 100
    else:
        signals['IV_percentile'] = None
    # 3. Implied Move (ATM straddle / spot)
    if not df.empty and 'option_type' in df.columns and 'strike_price' in df.columns:
        atm_strike = df.loc[df['atm_dist'].idxmin()]['strike_price'] if 'atm_dist' in df.columns else None
        if atm_strike is not None:
            calls = df[(df['option_type'] == 'call') & (df['strike_price'] == atm_strike)]
            puts = df[(df['option_type'] == 'put') & (df['strike_price'] == atm_strike)]
            if not calls.empty and not puts.empty:
                # Use mid price if available, else placeholder
                call_price = calls.iloc[0].get('mid', None) or calls.iloc[0].get('ask', None) or calls.iloc[0].get('bid', None)
                put_price = puts.iloc[0].get('mid', None) or puts.iloc[0].get('ask', None) or puts.iloc[0].get('bid', None)
                spot = atm_strike  # Approximate spot as ATM strike if no better data
                if call_price and put_price and spot:
                    signals['Implied_Move'] = (call_price + put_price) / spot
                else:
                    signals['Implied_Move'] = None
            else:
                signals['Implied_Move'] = None
        else:
            signals['Implied_Move'] = None
    else:
        signals['Implied_Move'] = None
    # 4. IV Skew / Smirk (OTM Put IV minus OTM Call IV)
    if 'implied_volatility' in df.columns and 'option_type' in df.columns and not df.empty:
        otm_puts = df[(df['option_type'] == 'put') & (df['moneyness'] < 1)]
        otm_calls = df[(df['option_type'] == 'call') & (df['moneyness'] > 1)]
        if not otm_puts.empty and not otm_calls.empty:
            put_iv = otm_puts.iloc[0]['implied_volatility']
            call_iv = otm_calls.iloc[0]['implied_volatility']
            signals['IV_Skew'] = put_iv - call_iv
        else:
            signals['IV_Skew'] = None
    else:
        signals['IV_Skew'] = None
    # 5. Call/Put Volume Ratio (if available)
    if 'volume' in df.columns and 'option_type' in df.columns and not df.empty:
        call_vol = df[df['option_type'] == 'call']['volume'].sum() if not df[df['option_type'] == 'call'].empty else 0
        put_vol = df[df['option_type'] == 'put']['volume'].sum() if not df[df['option_type'] == 'put'].empty else 0
        if put_vol > 0:
            signals['CallPut_Volume_Ratio'] = call_vol / put_vol
        else:
            signals['CallPut_Volume_Ratio'] = None
    else:
        signals['CallPut_Volume_Ratio'] = None
    # 6. Risk Reversals (stub)
    signals['Risk_Reversal'] = None  # Needs OTM call/put prices
    # 7. Unusual Options Volume (stub)
    signals['Unusual_Options_Volume'] = None  # Needs historical volume
    # 8. Open Interest (OI) - Example: total OI, max OI at a strike, or OI buildup
    if 'open_interest' in df.columns and not df['open_interest'].isnull().all():
        signals['Total_OI'] = df['open_interest'].sum()
        signals['Max_OI'] = df['open_interest'].max()
        # You can add more sophisticated OI signals here
    else:
        signals['Total_OI'] = None
        signals['Max_OI'] = None
    # 9. Calendar Spreads (stub)
    signals['Calendar_Spread'] = None  # Needs IV for multiple expiries
    # 10. Large Trades/Sweeps (stub)
    signals['Large_Trades_Sweeps'] = None  # Needs trade-level data
    return signals

def analyze_historical_event(fetcher, event):
    print(f"\nAnalyzing historical event: {event['event']} ({event['ticker']}) on {event['date']}")
    intervals = get_intervals(event['date'])
    all_metrics = []
    all_signals = []
    historical_ivs = []
    for interval in intervals:
        print(f"  Fetching metrics for {interval}...")
        expiration_dates = fetcher.get_available_expiration_dates(event['ticker'])
        exp = next((d for d in expiration_dates if d >= interval), None)
        if not exp:
            print(f"    No expiration found after {interval}")
            continue
        metrics = fetcher.get_option_metrics(event['ticker'], exp)
        metrics['interval_date'] = interval
        metrics['event'] = event['event']
        metrics['event_date'] = event['date']
        all_metrics.append(metrics)
        # For IV percentile
        if 'implied_volatility' in metrics.columns:
            historical_ivs.extend(metrics['implied_volatility'].dropna().tolist())
        # Compute signals for this interval
        signals = compute_signals(metrics, historical_ivs=historical_ivs if historical_ivs else None)
        signals['interval_date'] = interval
        all_signals.append(signals)
        print(f"    {len(metrics)} contracts fetched for expiration {exp}")
        # Save to CSV for later analysis
        outdir = 'event_metrics'
        os.makedirs(outdir, exist_ok=True)
        fname = f"{event['ticker']}_{safe_filename(event['event'])}_{interval}.csv"
        metrics.to_csv(os.path.join(outdir, fname), index=False)
    # Save signals summary
    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        fname = f"{event['ticker']}_{safe_filename(event['event'])}_signals.csv"
        signals_df.to_csv(os.path.join('event_metrics', fname), index=False)
        print(f"  Signals summary saved to {fname}")
    if all_metrics:
        df_all = pd.concat(all_metrics, ignore_index=True)
        # Only use intervals with IV and moneyness data
        valid = df_all.dropna(subset=['implied_volatility', 'moneyness'], how='any') if 'implied_volatility' in df_all.columns and 'moneyness' in df_all.columns else pd.DataFrame()
        if not valid.empty:
            # Sort by interval date
            valid = valid.sort_values('interval_date')
            # Use mean IV and moneyness per interval
            iv_by_interval = valid.groupby('interval_date')['implied_volatility'].mean()
            moneyness_by_interval = valid.groupby('interval_date')['moneyness'].mean()
            print("  IV trend (mean implied volatility by interval):")
            print(iv_by_interval)
            print("  Moneyness trend (mean moneyness by interval):")
            print(moneyness_by_interval)
            # Pre-event (first and last interval)
            iv_start, iv_end = iv_by_interval.iloc[0], iv_by_interval.iloc[-1]
            mny_start, mny_end = moneyness_by_interval.iloc[0], moneyness_by_interval.iloc[-1]
            iv_change = 100 * (iv_end - iv_start) / iv_start if iv_start else float('nan')
            mny_change = 100 * (mny_end - mny_start) / mny_start if mny_start else float('nan')
            print(f"  IV change: {iv_start:.3f} -> {iv_end:.3f} ({iv_change:+.1f}%)")
            print(f"  Moneyness change: {mny_start:.3f} -> {mny_end:.3f} ({mny_change:+.1f}%)")
        else:
            print("  No valid IV or moneyness data for trend analysis.")
        # Post-event price move (using previous close)
        try:
            aggs = fetcher.client.get_aggs(event['ticker'], 1, 'day', event['date'], event['date'], limit=1)
            if hasattr(aggs, 'results') and aggs.results:
                post_event_price = aggs.results[0].c
                # Try to get pre-event price from last interval
                pre_event_date = intervals[-1]
                pre_aggs = fetcher.client.get_aggs(event['ticker'], 1, 'day', pre_event_date, pre_event_date, limit=1)
                if hasattr(pre_aggs, 'results') and pre_aggs.results:
                    pre_event_price = pre_aggs.results[0].c
                    price_move = 100 * (post_event_price - pre_event_price) / pre_event_price if pre_event_price else float('nan')
                    print(f"  Pre-event close: {pre_event_price:.2f}, Post-event close: {post_event_price:.2f}, Move: {price_move:+.2f}%")
                    # Simple predictivity: did IV rise before a big move?
                    if not valid.empty and abs(price_move) > 5:
                        if iv_change > 0:
                            print("  Signal: IV rose before a large move (potentially predictive)")
                        else:
                            print("  Signal: IV did not rise before a large move")
                else:
                    print("  No pre-event price data available.")
            else:
                print("  No post-event price data available.")
        except Exception as e:
            print(f"  Could not fetch post-event price: {e}")
    print("  Metrics saved to CSV in 'event_metrics/' directory.")

def summarize_upcoming_event(fetcher, event):
    print(f"\nSummarizing upcoming event: {event['event']} ({event['ticker']}) on {event['date']}")
    expiration_dates = fetcher.get_available_expiration_dates(event['ticker'])
    exp = next((d for d in expiration_dates if d >= event['date']), None)
    if not exp:
        print(f"  No expiration found after event date {event['date']}")
        return
    # Use get_implied_volatility to ensure IV is present
    metrics = fetcher.get_implied_volatility(event['ticker'], exp)
    if metrics.empty:
        print(f"  No options data available for {event['ticker']} {exp}")
        return
    metrics['event'] = event['event']
    metrics['event_date'] = event['date']
    outdir = 'event_metrics'
    os.makedirs(outdir, exist_ok=True)
    fname = f"{event['ticker']}_{safe_filename(event['event'])}_upcoming.csv"
    metrics.to_csv(os.path.join(outdir, fname), index=False)
    print(f"  {len(metrics)} contracts fetched for expiration {exp}")
    # Summary: show IV and moneyness distribution if present
    if 'implied_volatility' in metrics.columns:
        print("  IV summary:")
        print(metrics['implied_volatility'].describe())
    else:
        print("  No implied volatility data available.")
    if 'moneyness' in metrics.columns:
        print("  Moneyness summary:")
        print(metrics['moneyness'].describe())
    else:
        print("  No moneyness data available.")
    print("  Metrics saved to CSV in 'event_metrics/' directory.")

def main():
    """Main function to run the options data analysis."""
    try:
        fetcher = OptionsDataFetcher()
        
        # Process each event
        for event in EVENTS:
            ticker = event['ticker']
            event_name = event['event']
            event_date = event['date']
            event_type = event['type']
            
            logger.info(f"\nAnalyzing {event_type} event: {event_name} ({ticker}) on {event_date}")
            
            # Get current price
            current_price = fetcher.get_current_price(ticker)
            if current_price is None:
                logger.warning(f"Proceeding without current price for {ticker}. Price-dependent signals will be skipped.")
                continue
            
            # Get intervals
            intervals = get_intervals(event_date)
            
            # Process each interval
            all_signals = []
            for interval_date in intervals:
                logger.info(f"  Fetching metrics for {interval_date}...")
                
                # Get available expiration dates
                expirations = fetcher.get_available_expiration_dates(ticker)
                if not expirations:
                    logger.warning(f"No expiration dates found for {ticker}")
                    continue
                
                # Use the expiration date closest to event date
                target_expiration = min(expirations, key=lambda x: abs(
                    datetime.strptime(x, "%Y-%m-%d") - 
                    datetime.strptime(event_date, "%Y-%m-%d")
                ))
                
                # Get options chain
                options_df = fetcher.get_options_chain(ticker, target_expiration)
                if options_df.empty:
                    logger.warning(f"No options data found for {ticker} on {interval_date}")
                    continue
                
                # Calculate signals
                signals = fetcher.calculate_signals(options_df, current_price)
                signals['interval_date'] = interval_date
                all_signals.append(signals)
                
                # Save detailed data
                options_df['interval_date'] = interval_date
                options_df['event'] = event_name
                options_df['event_date'] = event_date
                
                filename = safe_filename(f"{ticker}_{event_name}_{interval_date}.csv")
                options_df.to_csv(f"event_metrics/{filename}", index=False)
                logger.info(f"    {len(options_df)} contracts fetched for expiration {target_expiration}")
            
            # Save signals summary
            if all_signals:
                signals_df = pd.DataFrame(all_signals)
                signals_filename = safe_filename(f"{ticker}_{event_name}_signals.csv")
                signals_df.to_csv(f"event_metrics/{signals_filename}", index=False)
                logger.info(f"  Signals summary saved to {signals_filename}")
                
                # Analyze trends if we have enough data
                if len(all_signals) > 1:
                    try:
                        # Calculate trend metrics
                        iv_trend = np.polyfit(range(len(all_signals)), 
                                            [s['ATM_IV'] for s in all_signals if s['ATM_IV']], 1)[0]
                        volume_trend = np.polyfit(range(len(all_signals)), 
                                                [s['Total_OI'] for s in all_signals if s['Total_OI']], 1)[0]
                        
                        logger.info(f"  IV Trend: {iv_trend:.4f}")
                        logger.info(f"  Volume Trend: {volume_trend:.4f}")
                    except Exception as e:
                        logger.warning(f"  No valid IV or moneyness data for trend analysis.")
                
                # For historical events, try to get post-event price
                if event_type == 'historical':
                    try:
                        post_event_price = fetcher.get_current_price(ticker)
                        if post_event_price:
                            price_change = (post_event_price - current_price) / current_price * 100
                            logger.info(f"  Post-event price change: {price_change:.2f}%")
                    except Exception as e:
                        logger.warning(f"  Could not fetch post-event price: {str(e)}")
            
            logger.info("  Metrics saved to CSV in 'event_metrics/' directory.")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    main() 