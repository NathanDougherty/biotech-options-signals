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

## Future Improvements

1. Real-time signal monitoring for upcoming biotech catalysts
2. Integration with automated trading strategies
3. Enhanced visualization and reporting capabilities
4. Machine learning to improve signal predictiveness 