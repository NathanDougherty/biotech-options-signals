# Biotech Options Signal Analysis

This project analyzes options pricing signals for binary biotech catalysts to identify market expectations, investor sentiment, and predictive patterns. The analysis focuses on both historical biotech events and upcoming binary catalysts.

## Project Overview

This repository contains a comprehensive analysis of options data surrounding biotech catalyst events, including FDA decisions, clinical trial readouts, and data announcements. The project aims to identify predictive patterns in options pricing that can help anticipate market reactions to binary events.

## Key Files

- **biotech_options_analysis_report.md**: Detailed analysis report with key findings
- **streamlit_app.py**: Interactive web application for exploring the data
- **analyze_signals.py**: Core analysis logic and signal processing
- **data_fetcher.py**: Data collection utilities
- **signal_calculator.py**: Algorithms for computing various options signals

## Detailed Analysis Report

For a comprehensive analysis of our findings, please refer to the [Biotech Options Analysis Report](biotech_options_analysis_report.md). The report covers:

1. **Predictive Signals in Historical Events**: Analysis of which signals had the strongest predictive power
2. **Sentiment Assessment for Future Events**: Evaluation of bullish, bearish, and neutral signals for upcoming catalysts
3. **Signal Trends and Key Metrics**: Detailed visualization and analysis of how signals evolve leading up to events

## Key Findings

- Phase 1/2 trial results show the highest implied volatility (IV) levels (~220%), reflecting greater uncertainty
- IV skew is most pronounced before FDA decisions and general data announcements
- Call/put volume ratios are consistently lower for Phase 3 data readouts compared to earlier-stage trials
- Signal correlations with subsequent price movements are strongest for IV skew and call/put volume ratio

## Running the Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Signal Types Analyzed

The analysis focuses on various options pricing signals:

1. **Implied Volatility (IV)**: Expected magnitude of future stock movement
2. **IV Skew**: Directional sentiment bias (put vs call IV relationship)
3. **Call/Put Volume Ratio**: Skew in bullish or bearish sentiment
4. **Implied Move**: Market-implied price swing by expiration
5. **Open Interest Patterns**: Market consensus on landing zones

## Calculation Methodology

This project uses precise financial modeling techniques to calculate the options metrics. Here's how each metric is derived:

### Implied Volatility (IV)
Calculated using the Black-Scholes model with a Newton-Raphson iterative method (implemented in `data_fetcher.py`):

```python
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
```

This method:
1. Starts with an initial volatility estimate (σ = 0.3 or 30%)
2. Calculates d₁ and d₂ parameters using the standard Black-Scholes formula
3. Computes the theoretical option price and vega (sensitivity to volatility)
4. Uses the Newton-Raphson method to iteratively improve the volatility estimate
5. Converges when the theoretical price matches the market price (within tolerance)
6. Bounds volatility between 0.1% and 500% to ensure realistic values

### Options Greeks
The primary Greeks are calculated as follows:
- **Delta**: Partial derivative of option price with respect to underlying price
- **Gamma**: Second derivative of option price with respect to underlying price (δ²V/δS²)
- **Theta**: Rate of time decay in option value (δV/δt)
- **Vega**: Option's sensitivity to changes in volatility (δV/δσ)

### IV Percentile
Calculated by comparing current IV to historical values:
- Percentile = (% of historical IVs less than current IV) × 100
- Uses 30-day lookback period by default

### Implied Move
Calculated using:
- For event-based: Current price × ATM IV × √(days to event/365)
- For daily volatility: ATM IV ÷ √252 × √days × 100

### IV Skew
Two calculation methods:
1. Ratio method: OTM put IV ÷ OTM call IV
2. Difference method: OTM put IV - ATM call IV

### Put/Call Ratio
Based on trading volume:
- Call/Put Ratio = Total call volume ÷ Total put volume
- Values > 1 indicate bullish sentiment, < 1 indicate bearish sentiment

### Risk Reversal
Calculated as the difference between equidistant OTM call and put IVs:
- Risk Reversal = OTM call IV - OTM put IV
- Positive values suggest bullish sentiment, negative values suggest bearish sentiment

### Open Interest
Directly obtained from market data, representing the total number of outstanding contracts not yet closed out.

Data is primarily sourced from financial APIs with error handling for missing data points and API limitations.

## Future Improvements

1. Real-time signal monitoring for upcoming biotech catalysts
2. Integration with automated trading strategies
3. Enhanced visualization and reporting capabilities
4. Machine learning to improve signal predictiveness 