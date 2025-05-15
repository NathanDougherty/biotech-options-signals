"""
FILE: streamlit_app.py
PURPOSE: Interactive web dashboard for biotech options signal analysis.

This file creates a Streamlit web application that provides a user-friendly interface
for exploring and visualizing the options signal analysis for biotech companies with
upcoming catalysts. It serves as the front-end for the analysis pipeline.

Key functionality:
- Creates an interactive dashboard with sidebar controls for filtering data
- Organizes content into three tabs: Analysis Results, Visualizations, and Raw Data
- Provides filtering capabilities by ticker symbol and event type
- Implements all five major analysis types from the SignalAnalyzer class:
  1. Compare Metrics: Shows differences in metrics across event types
  2. Track Signal Changes: Displays how signals evolve over time
  3. Flag Notable Signals: Highlights unusual or significant option activity
  4. Compare with Returns: Correlates signals with subsequent stock performance
  5. Identify Predictive Signals: Shows which signals have highest predictive value

- Features interactive data tables, charts, and visualizations
- Includes file upload functionality when data is missing
- Implements caching for performance optimization
- Provides progress bars and status updates during analysis

This application integrates with analyze_signals.py and serves as the main user
interface for exploring the biotech options signal analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from analyze_signals import SignalAnalyzer
import time
import io
import base64

st.set_page_config(
    page_title="Biotech Options Signal Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Biotech Options Signal Analysis Dashboard")
st.markdown("""
This tool analyzes options signals for biotech companies with upcoming catalysts
to identify predictive patterns and market sentiment.
""")

# Sidebar for controls
st.sidebar.header("Analysis Controls")

# Load data and run analysis
@st.cache_data
def load_signal_files():
    signals_dir = "event_metrics"
    if not os.path.exists(signals_dir):
        os.makedirs(signals_dir)
        
    # Get all signal files
    signal_files = glob.glob(f"{signals_dir}/*_signals.csv")
    
    # Extract ticker names
    tickers = sorted(list(set([os.path.basename(f).split('_')[0] for f in signal_files])))
    
    # Extract event types
    event_types = []
    for f in signal_files:
        base_name = os.path.basename(f).replace('_signals.csv', '')
        parts = base_name.split('_')
        if len(parts) > 1:
            event_type = '_'.join(parts[1:])
            if event_type not in event_types:
                event_types.append(event_type)
    
    return signal_files, tickers, sorted(event_types)

# Check if data exists and provide upload functionality if not
signal_files = glob.glob("event_metrics/*_signals.csv")
if not signal_files:
    st.warning("No signal files found in the event_metrics directory.")
    
    st.info("""
    ### Data Missing
    
    This app requires signal data files in the event_metrics directory. You have two options:
    
    1. Upload CSV files using the uploader below
    2. View the pre-computed analysis results in the Analysis Results tab
    
    Note: On Streamlit Cloud, uploaded files are temporary and will be lost when the app is redeployed.
    """)
    
    uploaded_files = st.file_uploader("Upload signal files", type=['csv'], accept_multiple_files=True)
    
    if uploaded_files:
        if not os.path.exists("event_metrics"):
            os.makedirs("event_metrics")
            
        for uploaded_file in uploaded_files:
            with open(os.path.join("event_metrics", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.success(f"Uploaded {len(uploaded_files)} files! Please refresh the page.")
        signal_files = glob.glob("event_metrics/*_signals.csv")
    
    # Add a fallback to show pre-computed analysis
    st.subheader("Pre-computed Analysis Results")
    
    # Check if we have any pre-computed results to display
    precomputed_files = [
        "analysis_output/event_type_comparison.csv",
        "analysis_output/notable_signals.csv",
        "analysis_output/significant_patterns.csv"
    ]
    
    precomputed_exists = False
    for file in precomputed_files:
        if os.path.exists(file):
            precomputed_exists = True
            break
    
    if precomputed_exists:
        st.write("The following pre-computed analysis is available:")
        
        if os.path.exists("analysis_output/event_type_comparison.csv"):
            st.subheader("Event Type Comparison")
            event_comparison = pd.read_csv("analysis_output/event_type_comparison.csv")
            st.dataframe(event_comparison)
            
        if os.path.exists("analysis_output/notable_signals.csv"):
            st.subheader("Notable Signals")
            notable = pd.read_csv("analysis_output/notable_signals.csv")
            st.dataframe(notable)
            
        if os.path.exists("analysis_output/significant_patterns.csv"):
            st.subheader("Significant Patterns")
            patterns = pd.read_csv("analysis_output/significant_patterns.csv")
            st.dataframe(patterns)
            
        # Show the report
        if os.path.exists("biotech_options_analysis_report.md"):
            st.subheader("Analysis Report")
            with open("biotech_options_analysis_report.md", "r") as f:
                report_content = f.read()
            st.markdown(report_content)
    else:
        st.error("No pre-computed analysis files found. Please upload data files or contact the administrator.")

if signal_files:
    signal_files, available_tickers, available_event_types = load_signal_files()
    
    # Sidebar controls
    selected_tickers = st.sidebar.multiselect("Select Tickers", available_tickers)
    
    event_type_filter = st.sidebar.selectbox(
        "Filter by Event Type", 
        ["All"] + ["FDA/PDUFA", "Phase 3", "Data Release"] + available_event_types
    )
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Compare Metrics", "Track Signal Changes", "Flag Notable Signals", 
         "Compare with Returns", "Identify Predictive Signals"]
    )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Analysis Results", "Visualizations", "Raw Data"])
    
    # Initialize analyzer
    analyzer = SignalAnalyzer()
    
    # Filter by ticker if selected
    if selected_tickers:
        analyzer.signal_files = [f for f in analyzer.signal_files 
                           if any(t in os.path.basename(f) for t in selected_tickers)]
    
    # Filter by event type if selected
    if event_type_filter != "All":
        if event_type_filter == "FDA/PDUFA":
            event_terms = ["FDA", "PDUFA"]
        elif event_type_filter == "Phase 3":
            event_terms = ["Phase_3", "Phase 3"]
        elif event_type_filter == "Data Release":
            event_terms = ["data", "announcement", "release"]
        else:
            event_terms = [event_type_filter]
            
        analyzer.signal_files = [f for f in analyzer.signal_files 
                          if any(term in f for term in event_terms)]
    
    # Show selected files
    with tab3:
        st.subheader("Selected Signal Files")
        for file in analyzer.signal_files:
            st.write(os.path.basename(file))
    
    # Run analysis based on selection
    with tab1:
        if analyzer.signal_files:
            st.subheader(f"Running {analysis_type}")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create in-memory output for plots
            if not os.path.exists(analyzer.output_dir):
                os.makedirs(analyzer.output_dir)
            
            # Run the selected analysis
            if analysis_type == "Compare Metrics":
                status_text.text("Comparing metrics across events...")
                progress_bar.progress(25)
                results = analyzer.compare_metrics_across_events()
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                st.subheader("Metrics Comparison by Event Type")
                st.dataframe(results)
                
                # Show significant patterns if any
                sig_patterns_file = f"{analyzer.output_dir}/significant_patterns.csv"
                if os.path.exists(sig_patterns_file):
                    sig_patterns = pd.read_csv(sig_patterns_file)
                    if not sig_patterns.empty:
                        st.subheader("Statistically Significant Patterns")
                        st.dataframe(sig_patterns)
            
            elif analysis_type == "Track Signal Changes":
                status_text.text("Tracking signal changes over time...")
                progress_bar.progress(25)
                analyzer.track_signal_changes()
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Show rate of change analysis
                roc_file = f"{analyzer.output_dir}/signal_rate_of_change.csv"
                if os.path.exists(roc_file):
                    roc_df = pd.read_csv(roc_file)
                    if not roc_df.empty:
                        st.subheader("Signal Rate of Change Analysis")
                        st.dataframe(roc_df)
            
            elif analysis_type == "Flag Notable Signals":
                status_text.text("Flagging notable signals...")
                progress_bar.progress(25)
                all_flags, patterns = analyzer.flag_notable_signals()
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                if all_flags:
                    flags_df = pd.DataFrame(all_flags)
                    st.subheader("Notable Signals")
                    st.dataframe(flags_df)
                    
                    if 'Signal_Strength' in flags_df.columns:
                        st.subheader("Signal Strength by Ticker")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=flags_df['Ticker'].value_counts().index, 
                                  y=flags_df['Ticker'].value_counts().values, ax=ax)
                        plt.title("Number of Notable Signals by Ticker")
                        plt.xlabel("Ticker")
                        plt.ylabel("Count")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                else:
                    st.info("No notable signals found.")
                    
                if patterns:
                    patterns_df = pd.DataFrame(patterns)
                    st.subheader("Signal Patterns")
                    st.dataframe(patterns_df)
            
            elif analysis_type == "Compare with Returns":
                status_text.text("Comparing signals with returns...")
                progress_bar.progress(25)
                analyzer.compare_signals_with_returns()
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                results_file = f"{analyzer.output_dir}/signal_vs_returns.csv"
                if os.path.exists(results_file):
                    results_df = pd.read_csv(results_file)
                    if not results_df.empty:
                        st.subheader("Signal vs. Returns Analysis")
                        st.dataframe(results_df)
                        
                        corr_file = f"{analyzer.output_dir}/signal_return_correlations.csv"
                        if os.path.exists(corr_file):
                            corr_df = pd.read_csv(corr_file)
                            if not corr_df.empty:
                                st.subheader("Signal Correlation with Returns")
                                st.dataframe(corr_df)
                else:
                    st.info("No results for signal vs returns analysis.")
            
            elif analysis_type == "Identify Predictive Signals":
                status_text.text("Identifying predictive signals...")
                progress_bar.progress(25)
                analyzer.identify_predictive_signals()
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                pred_file = f"{analyzer.output_dir}/signal_predictiveness.csv"
                if os.path.exists(pred_file):
                    pred_df = pd.read_csv(pred_file)
                    if not pred_df.empty:
                        st.subheader("Signal Predictiveness")
                        st.dataframe(pred_df)
                else:
                    st.info("No predictiveness analysis results found. Run 'Compare with Returns' first.")
        
        else:
            st.info("No signal files match your selection criteria.")
    
    # Display visualizations
    with tab2:
        st.subheader("Visualizations")
        
        # Find all PNG files in the output directory
        if os.path.exists(analyzer.output_dir):
            plot_files = glob.glob(f"{analyzer.output_dir}/*.png")
            
            if plot_files:
                # Create a gallery of plots
                num_cols = 2
                num_plots = len(plot_files)
                
                # Create rows of plots
                for i in range(0, num_plots, num_cols):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i + j
                        if idx < num_plots:
                            plot_file = plot_files[idx]
                            plot_name = os.path.basename(plot_file).replace(".png", "")
                            cols[j].image(plot_file, caption=plot_name)
            else:
                st.info("No visualizations generated yet. Run an analysis to create visualizations.")
        else:
            st.info("Output directory not found.")
    
    # Display raw data
    with tab3:
        if analyzer.signal_files:
            for file in analyzer.signal_files:
                with st.expander(f"View {os.path.basename(file)}"):
                    try:
                        df = pd.read_csv(file)
                        st.dataframe(df)
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
        
        # Thresholds used for analysis
        st.subheader("Signal Thresholds")
        thresholds_df = pd.DataFrame([analyzer.thresholds]).T.reset_index()
        thresholds_df.columns = ["Threshold", "Value"]
        st.dataframe(thresholds_df)

# Add download link for the README and documentation
if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        readme_content = f.read()
    
    st.sidebar.markdown("### Documentation")
    st.sidebar.download_button(
        label="Download README",
        data=readme_content,
        file_name="README.md",
        mime="text/markdown"
    )

# Add footer
st.markdown("---")
st.caption("Biotech Options Signal Analysis Tool | © 2024") 