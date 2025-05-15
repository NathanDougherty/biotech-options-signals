"""
FILE: analysis.py
PURPOSE: Core analysis functionality for options data visualization and metrics calculations.

This file contains the OptionsAnalyzer class which provides comprehensive tools for analyzing
options data from biotech companies with upcoming catalysts. The class implements various
visualization methods and analytical techniques to extract insights from options data:

Key functionality:
- Event signal analysis for specific catalysts (e.g., FDA approvals, data releases)
- Visualization of implied volatility surfaces across strike prices and expiration dates
- Interactive dashboarding with Plotly for exploring options metrics
- Timeline analysis of signals leading up to catalyst events
- Summary report generation for options analysis findings

This class works in conjunction with signal_calculator.py and data_fetcher.py to form
the complete analysis pipeline. It focuses on the visualization and insight generation
aspects of the pipeline rather than data acquisition or signal calculation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsAnalyzer:
    def __init__(self):
        """Initialize the OptionsAnalyzer."""
        # Set style for plots
        plt.style.use('seaborn')
        sns.set_theme(style="whitegrid")

    def analyze_event_signals(self, event_data: pd.DataFrame, event_date: str) -> Dict:
        """
        Analyze options signals for a specific event.
        
        Args:
            event_data (pd.DataFrame): Options data for the event
            event_date (str): Event date in YYYY-MM-DD format
            
        Returns:
            Dict: Analysis results
        """
        try:
            results = {
                'iv_percentile': self._calculate_iv_percentile(event_data),
                'implied_move': self._calculate_implied_move(event_data),
                'iv_skew': self._calculate_iv_skew(event_data),
                'risk_reversal': self._calculate_risk_reversal(event_data),
                'call_put_ratio': self._calculate_call_put_ratio(event_data),
                'unusual_volume': self._identify_unusual_volume(event_data)
            }
            return results
        except Exception as e:
            logger.error(f"Error analyzing event signals: {str(e)}")
            return {}

    def plot_iv_surface(self, options_data: pd.DataFrame, title: str = "IV Surface") -> None:
        """
        Plot the implied volatility surface.
        
        Args:
            options_data (pd.DataFrame): Options data with IV
            title (str): Plot title
        """
        try:
            # Create pivot table for IV surface
            iv_surface = options_data.pivot_table(
                values='implied_volatility',
                index='days_to_expiry',
                columns='moneyness',
                aggfunc='mean'
            )
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.heatmap(iv_surface, cmap='viridis', annot=True, fmt='.2f')
            plt.title(title)
            plt.xlabel('Moneyness')
            plt.ylabel('Days to Expiry')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting IV surface: {str(e)}")

    def plot_signal_timeline(self, historical_data: pd.DataFrame, signal: str) -> None:
        """
        Plot the timeline of a specific signal.
        
        Args:
            historical_data (pd.DataFrame): Historical options data
            signal (str): Signal to plot (e.g., 'iv_percentile', 'call_put_ratio')
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(historical_data['date'], historical_data[signal])
            plt.title(f'{signal.replace("_", " ").title()} Timeline')
            plt.xlabel('Date')
            plt.ylabel(signal.replace("_", " ").title())
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting signal timeline: {str(e)}")

    def create_interactive_dashboard(self, options_data: pd.DataFrame) -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            options_data (pd.DataFrame): Options data
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'IV Surface', 'Call/Put Ratio',
                    'IV Skew', 'Risk Reversal'
                )
            )
            
            # Add IV Surface
            iv_surface = options_data.pivot_table(
                values='implied_volatility',
                index='days_to_expiry',
                columns='moneyness',
                aggfunc='mean'
            )
            fig.add_trace(
                go.Heatmap(z=iv_surface.values, x=iv_surface.columns, y=iv_surface.index),
                row=1, col=1
            )
            
            # Add Call/Put Ratio
            fig.add_trace(
                go.Scatter(x=options_data['date'], y=options_data['call_put_ratio']),
                row=1, col=2
            )
            
            # Add IV Skew
            fig.add_trace(
                go.Scatter(x=options_data['date'], y=options_data['iv_skew']),
                row=2, col=1
            )
            
            # Add Risk Reversal
            fig.add_trace(
                go.Scatter(x=options_data['date'], y=options_data['risk_reversal']),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text="Options Analysis Dashboard",
                showlegend=False
            )
            
            # Show plot
            fig.show()
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {str(e)}")

    def generate_summary_report(self, analysis_results: Dict) -> str:
        """
        Generate a summary report of the analysis results.
        
        Args:
            analysis_results (Dict): Analysis results
            
        Returns:
            str: Summary report
        """
        try:
            report = "Options Analysis Summary Report\n"
            report += "=" * 30 + "\n\n"
            
            # Add key metrics
            report += "Key Metrics:\n"
            report += f"IV Percentile: {analysis_results.get('iv_percentile', 'N/A'):.2f}%\n"
            report += f"Implied Move: {analysis_results.get('implied_move', 'N/A'):.2f}%\n"
            report += f"IV Skew: {analysis_results.get('iv_skew', 'N/A'):.4f}\n"
            report += f"Risk Reversal: {analysis_results.get('risk_reversal', 'N/A'):.2f}\n"
            report += f"Call/Put Ratio: {analysis_results.get('call_put_ratio', 'N/A'):.2f}\n\n"
            
            # Add unusual volume analysis
            unusual_volume = analysis_results.get('unusual_volume', [])
            if unusual_volume:
                report += "Unusual Volume Detected:\n"
                for item in unusual_volume:
                    report += f"- {item}\n"
            
            return report
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return "Error generating report"

def main():
    """Main function to demonstrate usage."""
    analyzer = OptionsAnalyzer()
    
    # Example usage with sample data
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=30),
        'implied_volatility': np.random.normal(0.3, 0.05, 30),
        'call_put_ratio': np.random.normal(1.2, 0.2, 30),
        'iv_skew': np.random.normal(0.05, 0.02, 30),
        'risk_reversal': np.random.normal(0.1, 0.05, 30),
        'days_to_expiry': np.random.randint(1, 90, 30),
        'moneyness': np.random.uniform(0.8, 1.2, 30)
    })
    
    # Analyze signals
    results = analyzer.analyze_event_signals(sample_data, '2024-02-01')
    
    # Generate plots
    analyzer.plot_iv_surface(sample_data)
    analyzer.plot_signal_timeline(sample_data, 'call_put_ratio')
    
    # Create interactive dashboard
    analyzer.create_interactive_dashboard(sample_data)
    
    # Generate summary report
    report = analyzer.generate_summary_report(results)
    print(report)

if __name__ == "__main__":
    main() 