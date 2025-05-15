"""
FILE: signal_calculator.py
PURPOSE: Specialized options signal calculation algorithms and metrics generation.

This file contains the OptionsSignalCalculator class which implements various mathematical 
and statistical methods for calculating options-based signals and metrics. It serves as the
quantitative engine of the analysis pipeline, providing sophisticated calculations for:

Key functionality:
- IV (Implied Volatility) percentile calculations across historical data
- Implied move predictions based on ATM IV and days to expiry
- IV skew measurements to gauge market sentiment (put vs call bias)
- Risk reversal calculations to identify directional bias in options pricing
- Call/put ratio analysis for volume and open interest
- Unusual volume detection algorithms with customizable thresholds
- Calendar spread calculations for volatility term structure analysis

The class handles robust error prevention with try/except blocks and comprehensive
logging. This module is typically used by the analysis.py and analyze_signals.py 
files which integrate these calculations into higher-level analysis workflows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsSignalCalculator:
    def __init__(self):
        """Initialize the OptionsSignalCalculator."""
        pass

    def calculate_iv_percentile(self, historical_iv: pd.Series) -> float:
        """
        Calculate the IV percentile for a given historical IV series.
        
        Args:
            historical_iv (pd.Series): Historical implied volatility data
            
        Returns:
            float: IV percentile (0-100)
        """
        try:
            current_iv = historical_iv.iloc[-1]
            iv_percentile = (historical_iv < current_iv).mean() * 100
            return iv_percentile
        except Exception as e:
            logger.error(f"Error calculating IV percentile: {str(e)}")
            return np.nan

    def calculate_implied_move(self, current_price: float, atm_iv: float, days_to_expiry: int) -> float:
        """
        Calculate the implied move for a given stock price and ATM IV.
        
        Args:
            current_price (float): Current stock price
            atm_iv (float): At-the-money implied volatility
            days_to_expiry (int): Days until options expiration
            
        Returns:
            float: Implied move as a percentage
        """
        try:
            # Convert annualized IV to daily
            daily_iv = atm_iv / np.sqrt(252)
            # Calculate implied move for the period
            implied_move = daily_iv * np.sqrt(days_to_expiry) * 100
            return implied_move
        except Exception as e:
            logger.error(f"Error calculating implied move: {str(e)}")
            return np.nan

    def calculate_iv_skew(self, options_chain: pd.DataFrame) -> float:
        """
        Calculate the IV skew (difference between OTM put and ATM call IV).
        
        Args:
            options_chain (pd.DataFrame): Options chain data with IV
            
        Returns:
            float: IV skew
        """
        try:
            # Get ATM call IV
            atm_calls = options_chain[
                (options_chain['option_type'] == 'call') & 
                (options_chain['strike_price'] >= options_chain['underlying_price'])
            ]
            atm_iv = atm_calls.iloc[0]['implied_volatility']
            
            # Get OTM put IV
            otm_puts = options_chain[
                (options_chain['option_type'] == 'put') & 
                (options_chain['strike_price'] < options_chain['underlying_price'])
            ]
            otm_iv = otm_puts.iloc[0]['implied_volatility']
            
            # Calculate skew
            skew = otm_iv - atm_iv
            return skew
        except Exception as e:
            logger.error(f"Error calculating IV skew: {str(e)}")
            return np.nan

    def calculate_risk_reversal(self, options_chain: pd.DataFrame) -> float:
        """
        Calculate the risk reversal (difference between OTM call and OTM put prices).
        
        Args:
            options_chain (pd.DataFrame): Options chain data
            
        Returns:
            float: Risk reversal value
        """
        try:
            # Get OTM call and put prices
            otm_calls = options_chain[
                (options_chain['option_type'] == 'call') & 
                (options_chain['strike_price'] > options_chain['underlying_price'])
            ]
            otm_puts = options_chain[
                (options_chain['option_type'] == 'put') & 
                (options_chain['strike_price'] < options_chain['underlying_price'])
            ]
            
            # Calculate risk reversal
            risk_reversal = otm_calls.iloc[0]['price'] - otm_puts.iloc[0]['price']
            return risk_reversal
        except Exception as e:
            logger.error(f"Error calculating risk reversal: {str(e)}")
            return np.nan

    def calculate_call_put_ratio(self, options_chain: pd.DataFrame) -> float:
        """
        Calculate the call/put ratio based on volume or open interest.
        
        Args:
            options_chain (pd.DataFrame): Options chain data
            
        Returns:
            float: Call/put ratio
        """
        try:
            # Calculate total volume for calls and puts
            call_volume = options_chain[options_chain['option_type'] == 'call']['volume'].sum()
            put_volume = options_chain[options_chain['option_type'] == 'put']['volume'].sum()
            
            # Calculate ratio
            ratio = call_volume / put_volume if put_volume > 0 else np.nan
            return ratio
        except Exception as e:
            logger.error(f"Error calculating call/put ratio: {str(e)}")
            return np.nan

    def identify_unusual_volume(self, options_chain: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify options with unusual volume.
        
        Args:
            options_chain (pd.DataFrame): Options chain data
            threshold (float): Volume threshold multiplier
            
        Returns:
            pd.DataFrame: Options with unusual volume
        """
        try:
            # Calculate average volume
            avg_volume = options_chain['volume'].mean()
            
            # Identify unusual volume
            unusual_volume = options_chain[options_chain['volume'] > (avg_volume * threshold)]
            return unusual_volume
        except Exception as e:
            logger.error(f"Error identifying unusual volume: {str(e)}")
            return pd.DataFrame()

    def calculate_calendar_spread(self, options_chain: pd.DataFrame, near_date: str, far_date: str) -> float:
        """
        Calculate the calendar spread value.
        
        Args:
            options_chain (pd.DataFrame): Options chain data
            near_date (str): Near-term expiration date
            far_date (str): Far-term expiration date
            
        Returns:
            float: Calendar spread value
        """
        try:
            # Get options for both dates
            near_options = options_chain[options_chain['expiration_date'] == near_date]
            far_options = options_chain[options_chain['expiration_date'] == far_date]
            
            # Calculate spread
            spread = far_options['price'].mean() - near_options['price'].mean()
            return spread
        except Exception as e:
            logger.error(f"Error calculating calendar spread: {str(e)}")
            return np.nan

def main():
    """Main function to demonstrate usage."""
    calculator = OptionsSignalCalculator()
    
    # Example usage with sample data
    sample_data = pd.DataFrame({
        'option_type': ['call', 'put', 'call', 'put'],
        'strike_price': [100, 100, 105, 95],
        'implied_volatility': [0.3, 0.35, 0.28, 0.32],
        'volume': [1000, 800, 500, 600],
        'price': [5.0, 4.0, 3.0, 2.0],
        'underlying_price': [100, 100, 100, 100]
    })
    
    # Calculate signals
    iv_percentile = calculator.calculate_iv_percentile(sample_data['implied_volatility'])
    implied_move = calculator.calculate_implied_move(100, 0.3, 30)
    iv_skew = calculator.calculate_iv_skew(sample_data)
    risk_reversal = calculator.calculate_risk_reversal(sample_data)
    call_put_ratio = calculator.calculate_call_put_ratio(sample_data)
    
    print("Sample Calculations:")
    print(f"IV Percentile: {iv_percentile:.2f}%")
    print(f"Implied Move: {implied_move:.2f}%")
    print(f"IV Skew: {iv_skew:.4f}")
    print(f"Risk Reversal: {risk_reversal:.2f}")
    print(f"Call/Put Ratio: {call_put_ratio:.2f}")

if __name__ == "__main__":
    main() 