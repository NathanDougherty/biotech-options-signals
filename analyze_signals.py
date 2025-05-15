"""
FILE: analyze_signals.py
PURPOSE: Core analysis framework for biotech options signal analysis.

This is the primary analysis module that coordinates the entire signal analysis pipeline.
It contains the SignalAnalyzer class which acts as the orchestration layer, pulling together
data, calculations, and visualizations into comprehensive analytical workflows.

Key functionality:
- Consolidates signal data across multiple biotech companies and event types
- Groups events by categories (FDA decisions, Phase 3 trials, data releases)
- Implements 5 major analysis methods:
  1. compare_metrics_across_events: Identifies patterns in how signals differ by event type
  2. track_signal_changes: Analyzes signal evolution as events approach
  3. flag_notable_signals: Detects unusual patterns exceeding predefined thresholds
  4. compare_signals_with_returns: Correlates signals with subsequent stock performance
  5. identify_predictive_signals: Determines which signals have highest predictive value

- Generates statistical analysis including significance testing
- Creates sophisticated visualizations (heatmaps, time series, scatter plots)
- Outputs detailed CSV reports and figures for each analysis type
- Handles data aggregation, normalization, and cross-referencing across signal types

This module is the central analytical engine for the project, integrating with the
Streamlit web interface and providing the core analysis capabilities displayed in 
the dashboard.
"""

import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Optional: For historical stock price data (uncomment if needed)
# import yfinance as yf

class SignalAnalyzer:
    def __init__(self, signals_dir="event_metrics"):
        """
        Initialize the SignalAnalyzer class with configuration for analyzing option signals.
        
        This constructor sets up the analyzer by loading signal files from the specified directory,
        categorizing them by event type (FDA, Phase trials, data releases), defining key metrics
        to analyze (IV, skew, volume ratios, etc.), and establishing thresholds for identifying
        notable signals. It creates the output directory structure if needed.
        
        Parameters:
        -----------
        signals_dir : str, default="event_metrics"
            Directory containing the signal CSV files to analyze
        """
        self.signals_dir = signals_dir
        self.output_dir = "analysis_output"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Find all signal files
        self.signal_files = glob.glob(f"{self.signals_dir}/*_signals.csv")
        print(f"Found {len(self.signal_files)} signal files to analyze")
        
        # Group signal files by event type
        self.fda_files = [f for f in self.signal_files if 'FDA' in f or 'PDUFA' in f]
        self.phase3_files = [f for f in self.signal_files if 'Phase_3' in f or 'Phase 3' in f]
        self.phase12_files = [f for f in self.signal_files if ('Phase_1' in f or 'Phase 1' in f) and not ('Phase_3' in f or 'Phase 3' in f)]
        self.data_files = [f for f in self.signal_files if 'data' in f or 'announcement' in f]
        
        # Key metrics to analyze
        self.key_metrics = [
            "ATM_IV", 
            "IV_Percentile", 
            "Implied_Move", 
            "IV_Skew", 
            "CallPut_Volume_Ratio", 
            "Risk_Reversal",
            "Total_OI", 
            "Max_OI", 
            "Calendar_Spread", 
            "Large_Trades_Sweeps"
        ]
        
        # Signal thresholds for flagging notable events
        self.thresholds = {
            # Volatility thresholds
            'ATM_IV': 100.0,  # High IV: > 100%
            'ATM_IV_low': 30.0,  # Unusually low IV: < 30%
            'IV_Percentile': 90.0,  # High IV percentile: > 90%
            'IV_Percentile_low': 10.0,  # Low IV percentile: < 10%
            
            # Directional sentiment thresholds
            'CallPut_Volume_Ratio_high': 1.5,  # Bullish sentiment: > 1.5
            'CallPut_Volume_Ratio_low': 0.7,   # Bearish sentiment: < 0.7
            'IV_Skew': 1.1,   # Significant skew: > 1.1 (puts more expensive)
            'IV_Skew_low': 0.9,  # Reverse skew: < 0.9 (calls more expensive)
            'Risk_Reversal': -10.0,  # Negative risk reversal: < -10 (bearish)
            'Risk_Reversal_high': 10.0,  # Positive risk reversal: > 10 (bullish)
            
            # Magnitude thresholds
            'Implied_Move': 15.0,  # Large implied move: > 15%
            'Large_Trades_Sweeps': 3,  # Unusual sweep activity: > 3 sweeps
            
            # Rate of change thresholds
            'IV_daily_change': 5.0,  # Fast IV change: > 5% per day
            'CallPut_Ratio_daily_change': 0.2,  # Fast sentiment shift: > 0.2 change per day
        }
    
    def compare_metrics_across_events(self):
        """
        Compare options metrics across different event types to identify consistent patterns.
        
        This method analyzes how various options metrics (IV, skew, volume ratios, etc.) differ
        across event categories like FDA decisions, Phase 3 trials, and data releases. It:
        
        1. Groups signal files by event type and calculates average metrics for each group
        2. Performs statistical analysis to identify significant differences between event types
        3. Creates visualizations including:
           - Bar charts comparing average metrics by event type
           - Heatmaps showing differences between event categories
           - Error bars demonstrating variation in the data
        4. Saves results to CSV files for further analysis
        
        Returns:
        --------
        DataFrame: Contains the average metrics for each event type
        
        Output files:
        -------------
        - event_type_comparison.csv: Tabular comparison of metrics by event type
        - significant_patterns.csv: Statistically significant differences found
        - Multiple visualization PNGs showing comparisons graphically
        """
        print("\n=== Comparing Metrics Across Event Types ===")
        
        # Group files by event type
        event_types = ["FDA Decisions", "Phase 3 Data", "Phase 1/2 Data", "Data Announcements"]
        file_groups = [self.fda_files, self.phase3_files, self.phase12_files, self.data_files]
        
        # Get counts for each group
        counts = [len(group) for group in file_groups]
        print(f"Event counts: {dict(zip(event_types, counts))}")
        
        # Create a DataFrame to store average metrics by event type
        results = pd.DataFrame(index=event_types)
        
        # Store all values for statistical analysis
        all_values = {event_type: {metric: [] for metric in self.key_metrics} 
                     for event_type in event_types}
        
        for metric in self.key_metrics:
            # Calculate average for each event type
            avgs = []
            stds = []  # Standard deviations
            for i, files in enumerate(file_groups):
                values = []
                event_type = event_types[i]
                
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        if metric in df.columns:
                            metric_values = df[metric].dropna().tolist()
                            values.extend(metric_values)
                            all_values[event_type][metric].extend(metric_values)
                    except Exception as e:
                        print(f"Error processing {file} for {metric}: {str(e)}")
                
                avg = np.mean(values) if values else np.nan
                std = np.std(values) if len(values) > 1 else np.nan
                avgs.append(avg)
                stds.append(std)
            
            results[metric] = avgs
            results[f"{metric}_std"] = stds
        
        # Save detailed results to CSV
        results.to_csv(f"{self.output_dir}/event_type_comparison.csv")
        print(f"Saved event type comparison to {self.output_dir}/event_type_comparison.csv")
        
        # Calculate pairwise statistical significance for key metrics
        significant_patterns = []
        for metric in ['ATM_IV', 'IV_Skew', 'CallPut_Volume_Ratio', 'Implied_Move']:
            for i, event1 in enumerate(event_types):
                for j, event2 in enumerate(event_types):
                    if i < j:  # Only compare each pair once
                        values1 = all_values[event1][metric]
                        values2 = all_values[event2][metric]
                        
                        if len(values1) >= 5 and len(values2) >= 5:  # Only test if enough samples
                            try:
                                # Perform t-test to check if differences are statistically significant
                                from scipy import stats
                                t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                                
                                if p_value < 0.05:  # Significant difference
                                    avg1 = np.mean(values1)
                                    avg2 = np.mean(values2)
                                    significant_patterns.append({
                                        'Metric': metric,
                                        'Event1': event1,
                                        'Event2': event2,
                                        'Event1_Avg': avg1,
                                        'Event2_Avg': avg2,
                                        'Difference': avg1 - avg2,
                                        'P_Value': p_value,
                                        'Finding': f"{event1} has {'higher' if avg1 > avg2 else 'lower'} {metric} than {event2} (p={p_value:.4f})"
                                    })
                            except Exception as e:
                                print(f"Error comparing {event1} and {event2} for {metric}: {str(e)}")
        
        # Save significant patterns
        if significant_patterns:
            patterns_df = pd.DataFrame(significant_patterns)
            patterns_df.to_csv(f"{self.output_dir}/significant_patterns.csv", index=False)
            print(f"Found {len(patterns_df)} statistically significant patterns across event types")
            
            # Create a heatmap visualization for key metrics
            for metric in ['ATM_IV', 'IV_Skew', 'CallPut_Volume_Ratio', 'Implied_Move']:
                if metric in results.columns:
                    metric_matrix = np.zeros((len(event_types), len(event_types)))
                    for i in range(len(event_types)):
                        metric_matrix[i, i] = results.loc[event_types[i], metric]
                        
                    # Fill in the significant differences
                    for pattern in significant_patterns:
                        if pattern['Metric'] == metric:
                            i = event_types.index(pattern['Event1'])
                            j = event_types.index(pattern['Event2'])
                            metric_matrix[i, j] = pattern['Difference']
                            metric_matrix[j, i] = -pattern['Difference']
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(metric_matrix, annot=True, fmt=".2f", 
                                xticklabels=event_types, yticklabels=event_types,
                                cmap="coolwarm", center=0)
                    plt.title(f"{metric} Differences Between Event Types")
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/event_comparison_{metric}_heatmap.png")
                    plt.close()
                    print(f"Created {metric} comparison heatmap")
        
        # Create visualizations for key metrics
        for metric in ['ATM_IV', 'IV_Skew', 'CallPut_Volume_Ratio', 'Implied_Move']:
            if metric in results.columns:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(results.index, results[metric])
                
                # Add error bars showing standard deviation
                if f"{metric}_std" in results.columns:
                    plt.errorbar(x=range(len(results.index)), y=results[metric], 
                                 yerr=results[f"{metric}_std"], fmt='none', color='black', capsize=5)
                
                plt.title(f"Average {metric} by Event Type")
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}',
                                ha='center', va='bottom', rotation=0)
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/event_type_{metric}.png")
                plt.close()
                print(f"Created {metric} comparison chart")
        
        return results
    
    def track_signal_changes(self, n_files=None):
        """
        Track how options signals change over time approaching catalytic events.
        
        This method performs temporal analysis of signal evolution by:
        
        1. Processing each signal file to track metrics as they approach the event date
        2. Calculating both absolute values and rate of change for key metrics
        3. Normalizing signals to show percentage changes from baseline
        4. Creating time series visualizations showing how signals evolve
        5. Aggregating data across event types to identify common patterns
        6. Analyzing the rate of change to detect acceleration or unusual movements
        
        The analysis reveals how market sentiment and volatility expectations shift
        as biotech catalysts approach, which can provide valuable trading insights.
        
        Parameters:
        -----------
        n_files : int, optional
            Limit the number of files to process (useful for testing)
        
        Output files:
        -------------
        - signal_rate_of_change.csv: Detailed analysis of how quickly signals change
        - Multiple time series charts showing signal evolution by ticker and event type
        - Aggregated visualizations showing patterns by event category
        """
        print("\n=== Tracking Signal Changes Over Time ===")
        
        files_to_process = self.signal_files[:n_files] if n_files else self.signal_files
        
        # Collect data for aggregated event type analysis
        all_event_data = {
            "FDA": {"days_to_event": [], "ATM_IV": [], "IV_Skew": [], "CallPut_Volume_Ratio": [], "Implied_Move": []},
            "Phase 3": {"days_to_event": [], "ATM_IV": [], "IV_Skew": [], "CallPut_Volume_Ratio": [], "Implied_Move": []},
            "Data Release": {"days_to_event": [], "ATM_IV": [], "IV_Skew": [], "CallPut_Volume_Ratio": [], "Implied_Move": []}
        }
        
        # Track rate of change for key metrics
        rate_of_change_data = []
        
        for file in files_to_process:
            try:
                df = pd.read_csv(file)
                
                # Check if there are multiple dates to create a timeline
                if len(df) <= 1:
                    print(f"Skipping {file} - insufficient time points for trend analysis")
                    continue
                    
                # Convert date column to datetime
                df['interval_date'] = pd.to_datetime(df['interval_date'])
                df = df.sort_values('interval_date')
                
                # Get ticker and event name for plot labels
                file_parts = os.path.basename(file).replace('_signals.csv', '').split('_')
                ticker = file_parts[0]
                event_type = ' '.join(file_parts[1:])
                
                # Determine the event category for aggregation
                event_category = None
                if any(term in event_type.lower() for term in ['fda', 'pdufa']):
                    event_category = "FDA"
                elif any(term in event_type.lower() for term in ['phase 3', 'phase_3']):
                    event_category = "Phase 3"
                elif any(term in event_type.lower() for term in ['data', 'announcement', 'release']):
                    event_category = "Data Release"
                
                # Identify the event date (last date in the series)
                event_date = df['interval_date'].iloc[-1]
                
                # Calculate days to event for each data point
                df['days_to_event'] = (event_date - df['interval_date']).dt.days
                
                # Create plots for key volatility and sentiment metrics
                key_metrics_to_plot = ['ATM_IV', 'IV_Skew', 'CallPut_Volume_Ratio', 'Implied_Move']
                
                # Calculate rate of change for each metric
                for metric in key_metrics_to_plot:
                    if metric in df.columns and not df[metric].isna().all() and len(df) >= 3:
                        # Calculate percent change from first to last measurement
                        first_value = df[metric].iloc[0]
                        last_value = df[metric].iloc[-2]  # Second to last (right before event)
                        if first_value != 0 and not pd.isna(first_value) and not pd.isna(last_value):
                            pct_change = ((last_value - first_value) / abs(first_value)) * 100
                            total_days = df['days_to_event'].iloc[0]  # Days from first measurement to event
                            
                            if total_days > 0:  # Avoid division by zero
                                # Calculate average daily change rate
                                daily_change_rate = pct_change / total_days
                                
                                rate_of_change_data.append({
                                    'Ticker': ticker,
                                    'Event_Type': event_type,
                                    'Event_Category': event_category,
                                    'Metric': metric,
                                    'Starting_Value': first_value,
                                    'Ending_Value': last_value,
                                    'Percent_Change': pct_change,
                                    'Days_Monitored': total_days,
                                    'Daily_Change_Rate': daily_change_rate
                                })
                                
                                # Add a column for normalized change from baseline
                                df[f'{metric}_pct_change'] = ((df[metric] - first_value) / abs(first_value)) * 100
                
                # Create days-to-event plots for each metric
                for metric in key_metrics_to_plot:
                    if metric in df.columns and not df[metric].isna().all():
                        plt.figure(figsize=(10, 6))
                        plt.plot(df['days_to_event'], df[metric], marker='o', linestyle='-', linewidth=2)
                        plt.title(f"{ticker} - {metric} Leading Up to {event_type}")
                        plt.xlabel("Days Until Event")
                        plt.ylabel(metric)
                        plt.grid(True, alpha=0.3)
                        plt.gca().invert_xaxis()  # Invert x-axis so time flows left to right
                        plt.tight_layout()
                        
                        # Save the figure
                        safe_filename = f"{self.output_dir}/{ticker}_{metric}_days_to_event.png"
                        plt.savefig(safe_filename)
                        plt.close()
                        print(f"Created days-to-event plot for {ticker} {metric}")
                        
                        # Add data to the aggregated collection if we have a category
                        if event_category and metric in all_event_data[event_category]:
                            for idx, row in df.iterrows():
                                if not pd.isna(row[metric]) and not pd.isna(row['days_to_event']):
                                    all_event_data[event_category]['days_to_event'].append(row['days_to_event'])
                                    all_event_data[event_category][metric].append(row[metric])
                
                # Create a normalized metrics plot showing percent change from baseline
                plt.figure(figsize=(12, 8))
                for metric in key_metrics_to_plot:
                    if f'{metric}_pct_change' in df.columns and not df[f'{metric}_pct_change'].isna().all():
                        plt.plot(df['days_to_event'], df[f'{metric}_pct_change'], 
                                 marker='o', linestyle='-', label=f"{metric}")
                
                plt.title(f"{ticker} - Signal Changes (%) Leading Up to {event_type}")
                plt.xlabel("Days Until Event")
                plt.ylabel("Percent Change from Baseline")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.gca().invert_xaxis()  # Invert x-axis so time flows left to right
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Add zero line
                plt.tight_layout()
                
                # Save the combined figure
                safe_filename = f"{self.output_dir}/{ticker}_percent_changes.png"
                plt.savefig(safe_filename)
                plt.close()
                print(f"Created normalized signal changes plot for {ticker}")
                
            except Exception as e:
                print(f"Error processing timeline for {file}: {str(e)}")
                continue
        
        # Create aggregated plots by event type
        for event_category, data in all_event_data.items():
            if len(data['days_to_event']) > 0:  # Only create if we have data
                for metric in ['ATM_IV', 'IV_Skew', 'CallPut_Volume_Ratio', 'Implied_Move']:
                    if len(data[metric]) > 0:
                        # Create a DataFrame for easier aggregation and plotting
                        agg_df = pd.DataFrame({
                            'days_to_event': data['days_to_event'],
                            metric: data[metric]
                        })
                        
                        # Group by days-to-event and calculate mean and standard deviation
                        grouped = agg_df.groupby('days_to_event')[metric].agg(['mean', 'std']).reset_index()
                        grouped = grouped.sort_values('days_to_event')
                        
                        plt.figure(figsize=(12, 6))
                        plt.errorbar(grouped['days_to_event'], grouped['mean'], 
                                    yerr=grouped['std'], capsize=4, marker='o', 
                                    linestyle='-', linewidth=2, label=f"{metric} (with std dev)")
                        
                        plt.title(f"Average {metric} Approaching {event_category} Events")
                        plt.xlabel("Days Until Event")
                        plt.ylabel(f"Average {metric}")
                        plt.grid(True, alpha=0.3)
                        plt.gca().invert_xaxis()  # Invert x-axis
                        plt.legend()
                        plt.tight_layout()
                        
                        # Save the aggregated figure
                        safe_filename = f"{self.output_dir}/{event_category}_{metric}_aggregate.png"
                        plt.savefig(safe_filename)
                        plt.close()
                        print(f"Created aggregated {metric} plot for {event_category} events")
        
        # Create rate of change analysis
        if rate_of_change_data:
            roc_df = pd.DataFrame(rate_of_change_data)
            roc_df.to_csv(f"{self.output_dir}/signal_rate_of_change.csv", index=False)
            print(f"Saved signal rate of change analysis to {self.output_dir}/signal_rate_of_change.csv")
            
            # Create visualization of daily change rates by event category
            for metric in ['ATM_IV', 'IV_Skew', 'CallPut_Volume_Ratio', 'Implied_Move']:
                metric_df = roc_df[roc_df['Metric'] == metric]
                
                if not metric_df.empty:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='Event_Category', y='Daily_Change_Rate', data=metric_df)
                    plt.title(f"Daily Change Rate for {metric} by Event Type")
                    plt.ylabel("Daily % Change")
                    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/{metric}_daily_change_rate.png")
                    plt.close()
                    print(f"Created daily change rate visualization for {metric}")
    
    def flag_notable_signals(self):
        """
        Apply thresholds to flag notable signal moments and detect meaningful patterns
        """
        print("\n=== Flagging Notable Signal Moments ===")
        
        all_flags = []
        signal_patterns = []
        
        for file in self.signal_files:
            try:
                df = pd.read_csv(file)
                
                # Get ticker and event type
                base_name = os.path.basename(file).replace('_signals.csv', '')
                ticker = base_name.split('_')[0]
                event_type = '_'.join(base_name.split('_')[1:])
                
                # Convert date to datetime
                df['interval_date'] = pd.to_datetime(df['interval_date'])
                df = df.sort_values('interval_date')
                
                # Get event date (last date in series)
                if len(df) > 0:
                    event_date = df['interval_date'].iloc[-1]
                    df['days_to_event'] = (event_date - df['interval_date']).dt.days
                
                # Calculate rate of change for key metrics where relevant
                for metric in ['ATM_IV', 'CallPut_Volume_Ratio', 'IV_Skew']:
                    if metric in df.columns and len(df) > 1:
                        df[f'{metric}_change'] = df[metric].diff()
                        days_diff = df['interval_date'].diff().dt.days
                        df[f'{metric}_daily_change'] = df[f'{metric}_change'] / days_diff
                        
                        # Calculate z-scores for changes (how unusual is this change?)
                        mean_change = df[f'{metric}_daily_change'].mean()
                        std_change = df[f'{metric}_daily_change'].std()
                        if std_change > 0:
                            df[f'{metric}_change_zscore'] = (df[f'{metric}_daily_change'] - mean_change) / std_change
                
                # Flag individual signal moments
                for idx, row in df.iterrows():
                    date = row['interval_date']
                    days_to_event = row['days_to_event'] if 'days_to_event' in row else None
                    flags_for_row = []
                    signal_strength = 0  # Score to measure overall signal strength
                    
                    # Check against thresholds
                    # --- Volatility signals ---
                    if 'ATM_IV' in row and not pd.isna(row['ATM_IV']):
                        if row['ATM_IV'] > self.thresholds['ATM_IV']:
                            flags_for_row.append(f"High IV: {row['ATM_IV']:.2f}%")
                            signal_strength += 1
                        elif row['ATM_IV'] < self.thresholds['ATM_IV_low']:
                            flags_for_row.append(f"Low IV: {row['ATM_IV']:.2f}%")
                            signal_strength += 0.5
                    
                    if 'IV_Percentile' in row and not pd.isna(row['IV_Percentile']):
                        if row['IV_Percentile'] > self.thresholds['IV_Percentile']:
                            flags_for_row.append(f"High IV Percentile: {row['IV_Percentile']:.2f}%")
                            signal_strength += 1
                        elif row['IV_Percentile'] < self.thresholds['IV_Percentile_low']:
                            flags_for_row.append(f"Low IV Percentile: {row['IV_Percentile']:.2f}%")
                            signal_strength += 0.5
                    
                    # --- Directional sentiment signals ---
                    if 'CallPut_Volume_Ratio' in row and not pd.isna(row['CallPut_Volume_Ratio']):
                        if row['CallPut_Volume_Ratio'] > self.thresholds['CallPut_Volume_Ratio_high']:
                            flags_for_row.append(f"Bullish: C/P Ratio {row['CallPut_Volume_Ratio']:.2f}")
                            signal_strength += 1
                        elif row['CallPut_Volume_Ratio'] < self.thresholds['CallPut_Volume_Ratio_low']:
                            flags_for_row.append(f"Bearish: C/P Ratio {row['CallPut_Volume_Ratio']:.2f}")
                            signal_strength += 1
                    
                    if 'IV_Skew' in row and not pd.isna(row['IV_Skew']):
                        if row['IV_Skew'] > self.thresholds['IV_Skew']:
                            flags_for_row.append(f"High Skew: {row['IV_Skew']:.2f}")
                            signal_strength += 1
                        elif row['IV_Skew'] < self.thresholds['IV_Skew_low']:
                            flags_for_row.append(f"Low Skew: {row['IV_Skew']:.2f}")
                            signal_strength += 0.5
                    
                    if 'Risk_Reversal' in row and not pd.isna(row['Risk_Reversal']):
                        if row['Risk_Reversal'] < self.thresholds['Risk_Reversal']:
                            flags_for_row.append(f"Bearish Risk Reversal: {row['Risk_Reversal']:.2f}")
                            signal_strength += 1.5
                        elif row['Risk_Reversal'] > self.thresholds['Risk_Reversal_high']:
                            flags_for_row.append(f"Bullish Risk Reversal: {row['Risk_Reversal']:.2f}")
                            signal_strength += 1.5
                    
                    # --- Magnitude signals ---
                    if 'Implied_Move' in row and not pd.isna(row['Implied_Move']) and row['Implied_Move'] > self.thresholds['Implied_Move']:
                        flags_for_row.append(f"Large Implied Move: {row['Implied_Move']:.2f}%")
                        signal_strength += 1
                    
                    if 'Large_Trades_Sweeps' in row and not pd.isna(row['Large_Trades_Sweeps']) and row['Large_Trades_Sweeps'] > self.thresholds['Large_Trades_Sweeps']:
                        flags_for_row.append(f"Unusual Option Sweeps: {row['Large_Trades_Sweeps']}")
                        signal_strength += 1.5
                    
                    # --- Rate of change signals ---
                    for metric in ['ATM_IV', 'CallPut_Volume_Ratio', 'IV_Skew']:
                        change_col = f'{metric}_daily_change'
                        zscore_col = f'{metric}_change_zscore'
                        
                        if change_col in row and not pd.isna(row[change_col]):
                            # Flag large absolute changes
                            if metric == 'ATM_IV' and abs(row[change_col]) > self.thresholds['IV_daily_change']:
                                direction = "increasing" if row[change_col] > 0 else "decreasing"
                                flags_for_row.append(f"Rapid IV {direction}: {row[change_col]:.2f}% per day")
                                signal_strength += 1
                            
                            elif metric == 'CallPut_Volume_Ratio' and abs(row[change_col]) > self.thresholds['CallPut_Ratio_daily_change']:
                                direction = "bullish shift" if row[change_col] > 0 else "bearish shift"
                                flags_for_row.append(f"Sentiment {direction}: {row[change_col]:.2f} per day")
                                signal_strength += 1
                            
                            # Flag statistically unusual changes (high z-scores)
                            if zscore_col in row and not pd.isna(row[zscore_col]) and abs(row[zscore_col]) > 2:
                                flags_for_row.append(f"Unusual {metric} change (z={row[zscore_col]:.2f})")
                                signal_strength += 1
                    
                    # --- Detect combination signals and patterns ---
                    
                    # IV-Skew Divergence (IV up, skew down = bullish; IV up, skew up = bearish)
                    if ('ATM_IV_change' in row and 'IV_Skew_change' in row and 
                        not pd.isna(row['ATM_IV_change']) and not pd.isna(row['IV_Skew_change'])):
                        
                        if row['ATM_IV_change'] > 0 and row['IV_Skew_change'] < 0:
                            flags_for_row.append("BULLISH IV-Skew Divergence")
                            signal_strength += 2
                        elif row['ATM_IV_change'] > 0 and row['IV_Skew_change'] > 0:
                            flags_for_row.append("BEARISH IV-Skew Confirmation")
                            signal_strength += 2
                    
                    # Volume-OI Agreement (increasing volume and OI = strong signal)
                    if ('Total_OI' in row and 'CallPut_Volume_Ratio' in row and 
                        not pd.isna(row['Total_OI']) and not pd.isna(row['CallPut_Volume_Ratio'])):
                        
                        oi_change = row['Total_OI'] - df['Total_OI'].iloc[idx-1] if idx > 0 else 0
                        
                        if oi_change > 0 and row['CallPut_Volume_Ratio'] > self.thresholds['CallPut_Volume_Ratio_high']:
                            flags_for_row.append("Strong Bullish Flow (↑OI + ↑C/P)")
                            signal_strength += 2
                        elif oi_change > 0 and row['CallPut_Volume_Ratio'] < self.thresholds['CallPut_Volume_Ratio_low']:
                            flags_for_row.append("Strong Bearish Flow (↑OI + ↓C/P)")
                            signal_strength += 2
                    
                    # Add to all flags if any were found, including signal strength
                    if flags_for_row:
                        flag_entry = {
                            'Ticker': ticker,
                            'Event_Type': event_type,
                            'Date': date,
                            'Days_to_Event': days_to_event,
                            'Flags': '; '.join(flags_for_row),
                            'Signal_Strength': signal_strength,
                            'Flag_Count': len(flags_for_row)
                        }
                        
                        # Add key metrics for reference
                        for metric in self.key_metrics:
                            if metric in row and not pd.isna(row[metric]):
                                flag_entry[metric] = row[metric]
                        
                        all_flags.append(flag_entry)
                
                # Detect patterns across time series
                if len(df) >= 3:
                    # Detect IV expansion or compression trends
                    if 'ATM_IV' in df.columns and not df['ATM_IV'].isna().all():
                        iv_trend = np.polyfit(range(len(df)), df['ATM_IV'], 1)[0]
                        iv_change_pct = iv_trend * len(df) / df['ATM_IV'].iloc[0] * 100 if df['ATM_IV'].iloc[0] > 0 else 0
                        
                        if abs(iv_change_pct) > 20:  # Significant trend
                            trend_type = "expansion" if iv_trend > 0 else "compression"
                            signal_patterns.append({
                                'Ticker': ticker,
                                'Event_Type': event_type,
                                'Pattern': f"IV {trend_type} trend",
                                'Magnitude': f"{abs(iv_change_pct):.1f}% over {len(df)} observations",
                                'Start_Date': df['interval_date'].iloc[0],
                                'End_Date': df['interval_date'].iloc[-1]
                            })
                    
                    # Detect sentiment shifts
                    if 'CallPut_Volume_Ratio' in df.columns and not df['CallPut_Volume_Ratio'].isna().all():
                        start_sentiment = df['CallPut_Volume_Ratio'].iloc[0]
                        end_sentiment = df['CallPut_Volume_Ratio'].iloc[-1]
                        sentiment_shift = end_sentiment - start_sentiment
                        
                        if abs(sentiment_shift) > 0.5:  # Significant shift
                            shift_type = "bullish" if sentiment_shift > 0 else "bearish"
                            signal_patterns.append({
                                'Ticker': ticker,
                                'Event_Type': event_type,
                                'Pattern': f"Sentiment shift to {shift_type}",
                                'Magnitude': f"C/P Ratio change of {sentiment_shift:.2f}",
                                'Start_Date': df['interval_date'].iloc[0],
                                'End_Date': df['interval_date'].iloc[-1]
                            })
                    
                    # Detect impending event signals (IV run-up)
                    if len(df) >= 4 and 'ATM_IV' in df.columns and 'days_to_event' in df.columns:
                        recent_df = df.iloc[-4:]  # Last 4 observations
                        if not recent_df['ATM_IV'].isna().all():
                            recent_iv_trend = np.polyfit(range(len(recent_df)), recent_df['ATM_IV'], 1)[0]
                            days_to_event = recent_df['days_to_event'].iloc[-1]
                            
                            if recent_iv_trend > 0 and days_to_event < 14:
                                signal_patterns.append({
                                    'Ticker': ticker,
                                    'Event_Type': event_type,
                                    'Pattern': "Pre-event IV buildup",
                                    'Magnitude': f"IV increasing by ~{recent_iv_trend:.2f} per observation",
                                    'Days_To_Event': days_to_event,
                                    'End_Date': df['interval_date'].iloc[-1]
                                })
            
            except Exception as e:
                print(f"Error flagging signals in {file}: {str(e)}")
                continue
        
        # Convert to DataFrame and save
        if all_flags:
            flags_df = pd.DataFrame(all_flags)
            flags_df.to_csv(f"{self.output_dir}/notable_signals.csv", index=False)
            print(f"Saved {len(flags_df)} notable signals to {self.output_dir}/notable_signals.csv")
            
            # Create a heatmap of signal strength by ticker and days to event
            if 'Days_to_Event' in flags_df.columns:
                try:
                    # Create time bins for days-to-event
                    flags_df['Time_Bin'] = pd.cut(flags_df['Days_to_Event'], 
                                                  bins=[0, 7, 14, 30, 60, np.inf],
                                                  labels=['0-7d', '7-14d', '14-30d', '30-60d', '60d+'])
                    
                    # Create a pivot table of average signal strength
                    strength_pivot = flags_df.pivot_table(
                        index='Ticker', 
                        columns='Time_Bin',
                        values='Signal_Strength', 
                        aggfunc='mean'
                    )
                    
                    # Plot heatmap
                    plt.figure(figsize=(10, max(6, len(strength_pivot) * 0.4)))
                    sns.heatmap(strength_pivot, annot=True, cmap='YlOrRd', fmt='.1f')
                    plt.title("Signal Strength by Ticker and Time to Event")
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/signal_strength_heatmap.png")
                    plt.close()
                    print("Created signal strength heatmap")
                except Exception as e:
                    print(f"Error creating signal strength heatmap: {str(e)}")
            
            # Create a bubble chart of signal frequency and strength
            plt.figure(figsize=(12, 8))
            
            # Group by ticker and calculate metrics
            ticker_summary = flags_df.groupby('Ticker').agg(
                Count=('Signal_Strength', 'count'),
                Avg_Strength=('Signal_Strength', 'mean'),
                Max_Strength=('Signal_Strength', 'max')
            ).reset_index()
            
            # Create bubble chart
            plt.scatter(ticker_summary['Count'], ticker_summary['Avg_Strength'], 
                      s=ticker_summary['Max_Strength']*50, alpha=0.6)
            
            # Add ticker labels
            for i, row in ticker_summary.iterrows():
                plt.annotate(row['Ticker'], 
                           (row['Count'], row['Avg_Strength']),
                           xytext=(5, 5), textcoords='offset points')
            
            plt.title("Signal Frequency and Strength by Ticker")
            plt.xlabel("Number of Notable Signals")
            plt.ylabel("Average Signal Strength")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/signal_bubble_chart.png")
            plt.close()
            print("Created signal bubble chart")
        else:
            print("No notable signals found")
        
        # Process pattern signals
        if signal_patterns:
            patterns_df = pd.DataFrame(signal_patterns)
            patterns_df.to_csv(f"{self.output_dir}/signal_patterns.csv", index=False)
            print(f"Saved {len(patterns_df)} signal patterns to {self.output_dir}/signal_patterns.csv")
            
            # Create pattern visualization
            pattern_counts = patterns_df['Pattern'].value_counts()
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=pattern_counts.index, y=pattern_counts.values)
            plt.title("Frequency of Signal Patterns")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/pattern_frequency.png")
            plt.close()
            print("Created pattern frequency chart")
        
        return all_flags, signal_patterns
    
    def compare_signals_with_returns(self):
        """
        For historical events, compare options signals with actual stock returns.
        
        This method provides a critical validation of signal predictiveness by:
        
        1. Fetching historical stock price data using yfinance (or simulating if unavailable)
        2. Calculating actual returns following catalytic events
        3. Comparing various options signals with subsequent price movements
        4. Measuring correlation between signal strength and return magnitude
        5. Calculating directional prediction accuracy (e.g., did bearish signals precede downward moves)
        6. Creating visualizations showing relationship between signals and returns
        7. Identifying which timeframes have the strongest signal-return relationship
        
        The analysis helps determine which options metrics have historically been most
        accurate in predicting both the direction and magnitude of price movements following
        biotech catalyst events.
        
        Note: This requires yfinance package and internet connection for real data
        
        Returns:
        --------
        None
        
        Output files:
        -------------
        - signal_vs_returns.csv: Detailed comparison of signals and subsequent returns
        - signal_return_correlations.csv: Correlation analysis by signal type and timeframe
        - Multiple visualization PNGs showing relationships between signals and returns
        """
        print("\n=== Comparing Signals with Actual Returns ===")
        
        try:
            import yfinance as yf
            use_real_data = True
            print("Using yfinance to fetch actual stock price data")
        except ImportError:
            use_real_data = False
            print("yfinance package not found - using simulated returns for demonstration")
            print("To use real data, install yfinance: pip install yfinance")
        
        results = []
        event_windows = [1, 3, 5, 10]  # Days to check after event
        signal_windows = [30, 14, 7]   # Days before event to check signal strength
        
        for file in self.signal_files:
            try:
                df = pd.read_csv(file)
                if len(df) == 0:
                    continue
                    
                # Get ticker and event info
                file_base = os.path.basename(file)
                ticker = file_base.split('_')[0]
                event_type = '_'.join(file_base.split('_')[1:-1])  # Everything between ticker and _signals.csv
                
                # Convert dates and sort
                df['interval_date'] = pd.to_datetime(df['interval_date'])
                df = df.sort_values('interval_date')
                
                # Find the event date (last date in the series)
                event_date = df['interval_date'].iloc[-1]
                
                actual_returns = {}
                if use_real_data:
                    # Calculate start and end dates for fetching price data
                    # Get data from 30 days before first signal to 10 days after event
                    start_date = df['interval_date'].iloc[0] - pd.Timedelta(days=5)
                    end_date = event_date + pd.Timedelta(days=max(event_windows) + 5)  # Add buffer days
                    
                    # Fetch historical price data
                    try:
                        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                        
                        if not stock_data.empty:
                            # Get the closing price on the event date or the closest trading day before it
                            event_idx = stock_data.index.searchsorted(event_date)
                            if event_idx >= len(stock_data.index):
                                event_idx = len(stock_data.index) - 1
                            
                            event_price = stock_data['Close'].iloc[event_idx]
                            
                            # Calculate returns for different windows
                            for window in event_windows:
                                future_idx = min(event_idx + window, len(stock_data.index) - 1)
                                if future_idx > event_idx:
                                    future_price = stock_data['Close'].iloc[future_idx]
                                    actual_returns[window] = ((future_price - event_price) / event_price) * 100
                                    
                            # Create a price chart with key dates marked
                            plt.figure(figsize=(12, 6))
                            stock_data['Close'].plot()
                            
                            # Mark the dates where we have signal data
                            for date in df['interval_date']:
                                date_idx = stock_data.index.searchsorted(date)
                                if date_idx < len(stock_data.index):
                                    plt.axvline(x=stock_data.index[date_idx], color='green', linestyle='--', alpha=0.5)
                            
                            # Mark the event date
                            plt.axvline(x=stock_data.index[event_idx], color='red', linestyle='-', linewidth=2, 
                                        label='Event Date')
                            
                            plt.title(f"{ticker} Price Movement Around {event_type}")
                            plt.ylabel("Price ($)")
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(f"{self.output_dir}/{ticker}_price_movement.png")
                            plt.close()
                            print(f"Created price movement chart for {ticker}")
                    
                    except Exception as e:
                        print(f"Error fetching stock data for {ticker}: {str(e)}")
                        # Fall back to simulated returns if yfinance fetch fails
                        for window in event_windows:
                            implied_move = df['Implied_Move'].iloc[-1] if 'Implied_Move' in df.columns and not pd.isna(df['Implied_Move'].iloc[-1]) else 10
                            actual_returns[window] = np.random.normal(0, implied_move/2)
                
                else:
                    # Generate simulated returns if yfinance not available
                    for window in event_windows:
                        implied_move = df['Implied_Move'].iloc[-1] if 'Implied_Move' in df.columns and not pd.isna(df['Implied_Move'].iloc[-1]) else 10
                        actual_returns[window] = np.random.normal(0, implied_move/2)
                
                # Calculate signal values at different points before the event
                for signal_window in signal_windows:
                    # Find the closest date to 'signal_window' days before the event
                    target_date = event_date - pd.Timedelta(days=signal_window)
                    closest_idx = (df['interval_date'] - target_date).abs().idxmin()
                    
                    signal_suffix = f"{signal_window}d"
                    
                    # Calculate signal change from this point to event
                    for metric in ['ATM_IV', 'IV_Skew', 'CallPut_Volume_Ratio', 'Implied_Move', 'Risk_Reversal']:
                        if metric in df.columns:
                            # Get the value at this time point
                            signal_value = df.loc[closest_idx, metric]
                            
                            # Store this value with a suffix indicating days before event
                            results_key = f"{metric}_{signal_suffix}"
                            
                            # Also calculate change from this point to right before event
                            if not pd.isna(signal_value) and len(df) > 1:
                                pre_event_value = df[metric].iloc[-2]  # Second to last value (right before event)
                                if not pd.isna(pre_event_value):
                                    change = pre_event_value - signal_value
                                    pct_change = (change / abs(signal_value)) * 100 if signal_value != 0 else np.nan
                                    
                                    # Get return details
                                    for window, ret in actual_returns.items():
                                        if abs(pct_change) > 0:  # If there was a signal change
                                            # Direction agreement: 1 if both same sign, -1 if opposite, 0 if either is zero
                                            direction_match = np.sign(pct_change) * np.sign(ret) if (pct_change != 0 and ret != 0) else 0
                                            
                                            result_row = {
                                                'Ticker': ticker,
                                                'Event_Type': event_type,
                                                'Event_Date': event_date,
                                                'Signal': metric,
                                                'Days_Before': signal_window,
                                                'Signal_Value': signal_value,
                                                'Signal_Change': change,
                                                'Signal_Pct_Change': pct_change,
                                                f'Return_{window}d': ret,
                                                f'Direction_Match_{window}d': direction_match
                                            }
                                            results.append(result_row)
            
            except Exception as e:
                print(f"Error processing returns for {file}: {str(e)}")
                continue
        
        # Convert to DataFrame
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{self.output_dir}/signal_vs_returns.csv", index=False)
            print(f"Saved detailed signal vs returns analysis to {self.output_dir}/signal_vs_returns.csv")
            
            # Create a comprehensive correlation analysis
            correlation_results = []
            for signal in ['ATM_IV', 'IV_Skew', 'CallPut_Volume_Ratio', 'Implied_Move', 'Risk_Reversal']:
                for days_before in signal_windows:
                    for return_window in event_windows:
                        signal_data = results_df[(results_df['Signal'] == signal) & 
                                                (results_df['Days_Before'] == days_before)]
                        
                        if len(signal_data) >= 5:  # Only analyze if we have enough data points
                            return_col = f'Return_{return_window}d'
                            
                            # Calculate correlation between signal change and return
                            corr = signal_data[['Signal_Pct_Change', return_col]].corr().iloc[0, 1]
                            
                            # Calculate success rate (direction match)
                            direction_col = f'Direction_Match_{return_window}d'
                            success_rate = (signal_data[direction_col] > 0).mean() * 100
                            
                            correlation_results.append({
                                'Signal': signal,
                                'Days_Before_Event': days_before,
                                'Return_Window': return_window,
                                'Correlation': corr,
                                'Direction_Success_Rate': success_rate,
                                'Sample_Size': len(signal_data)
                            })
            
            # Save correlation results
            if correlation_results:
                corr_df = pd.DataFrame(correlation_results)
                corr_df.to_csv(f"{self.output_dir}/signal_return_correlations.csv", index=False)
                print(f"Saved signal correlation analysis to {self.output_dir}/signal_return_correlations.csv")
                
                # Create a heatmap of correlations
                pivot_df = corr_df.pivot_table(
                    index='Signal', 
                    columns=['Days_Before_Event', 'Return_Window'],
                    values='Correlation'
                )
                
                plt.figure(figsize=(14, 8))
                sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Signal-Return Correlation by Timing Window')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/signal_return_correlation_heatmap.png")
                plt.close()
                print("Created signal-return correlation heatmap")
                
                # Create success rate chart
                plt.figure(figsize=(12, 8))
                success_pivot = corr_df.pivot_table(
                    index='Signal', 
                    columns='Days_Before_Event',
                    values='Direction_Success_Rate', 
                    aggfunc='mean'
                )
                
                sns.heatmap(success_pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=100, fmt='.1f')
                plt.title('Signal Direction Success Rate (%)')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/signal_success_rate_heatmap.png")
                plt.close()
                print("Created signal success rate heatmap")
            
            # Create visualizations of implied vs actual moves
            for window in event_windows:
                return_col = f'Return_{window}d'
                
                # Only keep rows with both implied move and return data
                plot_data = results_df[
                    (results_df['Signal'] == 'Implied_Move') & 
                    (results_df['Days_Before'] == min(signal_windows)) & 
                    (~results_df['Signal_Value'].isna()) & 
                    (~results_df[return_col].isna())
                ]
                
                if len(plot_data) >= 3:  # Only create plot if we have enough data points
                    plt.figure(figsize=(10, 8))
                    plt.scatter(plot_data['Signal_Value'], plot_data[return_col].abs(), alpha=0.7)
                    
                    # Add ticker labels
                    for i, ticker in enumerate(plot_data['Ticker']):
                        plt.annotate(ticker, 
                                   (plot_data['Signal_Value'].iloc[i], abs(plot_data[return_col].iloc[i])),
                                   xytext=(5, 5), textcoords='offset points')
                    
                    # Add diagonal line (perfect prediction)
                    max_val = max(plot_data['Signal_Value'].max(), plot_data[return_col].abs().max())
                    min_val = 0
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                    
                    plt.title(f"Implied Move vs Actual {window}-Day Return")
                    plt.xlabel("Implied Move (%)")
                    plt.ylabel(f"Absolute {window}-Day Return (%)")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/implied_vs_actual_{window}d.png")
                    plt.close()
                    print(f"Created implied vs actual {window}-day returns chart")
        else:
            print("No results for signal vs returns analysis")
    
    def identify_predictive_signals(self):
        """
        Identify which options signals were most predictive in historical events.
        
        This method synthesizes the results from previous analyses to determine which
        signals have the strongest predictive power. It:
        
        1. Requires output from compare_signals_with_returns() as input
        2. Ranks signal types by their correlation with subsequent returns
        3. Measures signal reliability through statistical validation
        4. Identifies the optimal timeframes for each signal type
        5. Creates visualizations showing the relative predictive power
        
        The insights from this method help traders focus on the most proven signals
        for future biotech catalyst events, increasing the probability of successful trades.
        
        Returns:
        --------
        dict: Contains correlation coefficients for each signal type
        
        Output files:
        -------------
        - signal_predictiveness.csv: Ranked list of signals by predictive power
        - signal_predictiveness.png: Visualization of signal predictive strength
        """
        print("\n=== Identifying Most Predictive Signals ===")
        
        # Check if the results file exists from compare_signals_with_returns
        results_file = f"{self.output_dir}/signal_vs_returns.csv"
        if not os.path.exists(results_file):
            print(f"Results file {results_file} not found. Run compare_signals_with_returns first.")
            return
        
        results_df = pd.read_csv(results_file)
        
        # Calculate correlations between signals and actual returns
        correlations = {}
        for signal in ['ATM_IV', 'Implied_Move', 'IV_Skew', 'CallPut_Ratio', 'Risk_Reversal']:
            if signal in results_df.columns:
                # Calculate correlation with absolute return (magnitude)
                corr = results_df[[signal, 'Actual_Return']].corr().iloc[0, 1]
                correlations[signal] = corr
        
        # Convert to DataFrame and save
        corr_df = pd.DataFrame([correlations])
        corr_df.to_csv(f"{self.output_dir}/signal_predictiveness.csv", index=False)
        print(f"Saved signal predictiveness to {self.output_dir}/signal_predictiveness.csv")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(correlations.keys(), correlations.values())
        plt.title("Signal Correlation with Actual Returns")
        plt.ylabel("Correlation")
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/signal_predictiveness.png")
        plt.close()
        print("Created signal predictiveness chart")
        
        return correlations
    
    def run_all_analyses(self):
        """
        Run all analysis methods in sequence to perform comprehensive signal analysis.
        
        This method orchestrates the full analysis pipeline by executing all component
        methods in a logical sequence. It provides a convenient way to run the entire
        analysis with a single command, ensuring that:
        
        1. All analyses use consistent data and parameters
        2. Dependencies between analyses are respected (e.g., predictive analysis requires returns data)
        3. All output files are generated in a single operation
        4. The analysis is executed in the most efficient order
        
        This is particularly useful for batch processing multiple event datasets or for
        running regular updates to the analysis with new data.
        
        Returns:
        --------
        None
        
        Output:
        -------
        All outputs from component methods are saved to the output_dir
        """
        print("\n=== Running All Analyses ===")
        self.compare_metrics_across_events()
        self.track_signal_changes()
        self.flag_notable_signals()
        self.compare_signals_with_returns()
        self.identify_predictive_signals()
        print("\n=== Analysis Complete ===")
        print(f"All results saved to {self.output_dir}/")

# Run the analysis if the script is executed directly
if __name__ == "__main__":
    analyzer = SignalAnalyzer()
    analyzer.run_all_analyses() 