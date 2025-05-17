"""
Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pandas as pd

# Create directory if it doesn't exist
os.makedirs("static/screenshots/dashboard", exist_ok=True)

# Create a dark-themed dashboard overview mockup that matches the actual UI
plt.style.use('dark_background')

# Create figure with specific aspect ratio to match the dashboard
fig = plt.figure(figsize=(13, 8), facecolor='#0E1117')

# Set up the grid
gs = gridspec.GridSpec(20, 24)

# Left sidebar (takes 1/5 of the width)
sidebar = plt.subplot(gs[:, :5])
sidebar.axis('off')
sidebar.set_facecolor('#0E1117')

# Main area
main_area = plt.subplot(gs[:, 5:])
main_area.axis('off')
main_area.set_facecolor('#0E1117')

# Add sidebar elements
sidebar.text(0.5, 0.95, 'Analysis Controls', fontsize=16, fontweight='bold', ha='center')
sidebar.text(0.5, 0.9, 'Select Tickers', fontsize=14, ha='center')
sidebar.add_patch(patches.Rectangle((0.1, 0.85), 0.8, 0.04, facecolor='#262730', edgecolor='gray', linewidth=1, alpha=0.8))
sidebar.text(0.5, 0.865, 'Choose an option', fontsize=12, ha='center', color='#D3D3D3')

sidebar.text(0.5, 0.8, 'Filter by Event Type', fontsize=14, ha='center')
sidebar.add_patch(patches.Rectangle((0.1, 0.75), 0.8, 0.04, facecolor='#262730', edgecolor='gray', linewidth=1, alpha=0.8))
sidebar.text(0.5, 0.765, 'All', fontsize=12, ha='center', color='#D3D3D3')

sidebar.text(0.5, 0.7, 'Analysis Type', fontsize=14, ha='center')
sidebar.add_patch(patches.Rectangle((0.1, 0.65), 0.8, 0.04, facecolor='#262730', edgecolor='gray', linewidth=1, alpha=0.8))
sidebar.text(0.5, 0.665, 'Compare Metrics', fontsize=12, ha='center', color='#D3D3D3')

sidebar.text(0.5, 0.45, 'Documentation', fontsize=16, fontweight='bold', ha='center')
sidebar.add_patch(patches.Rectangle((0.1, 0.4), 0.8, 0.04, facecolor='#262730', edgecolor='gray', linewidth=1, alpha=0.8))
sidebar.text(0.5, 0.415, 'Download README', fontsize=12, ha='center', color='#D3D3D3')

# Main content area - Title
main_area.text(0.5, 0.95, 'Biotech Options Signal Analysis Dashboard', fontsize=24, fontweight='bold', ha='center')

main_area.text(0.5, 0.9, 'This tool analyzes options signals for biotech companies with upcoming catalysts to identify predictive patterns and market sentiment.', fontsize=12, ha='center')

# Tabs
tab_y = 0.84
tab_height = 0.02
main_area.add_patch(patches.Rectangle((0.15, tab_y), 0.1, tab_height, facecolor='#E61E50', edgecolor=None))
main_area.text(0.2, tab_y+0.01, 'Analysis Results', fontsize=12, ha='center', va='center')

main_area.add_patch(patches.Rectangle((0.25, tab_y), 0.15, tab_height, facecolor='#0E1117', edgecolor=None))
main_area.text(0.325, tab_y+0.01, 'Visualizations', fontsize=12, ha='center', va='center', color='#D3D3D3')

main_area.add_patch(patches.Rectangle((0.4, tab_y), 0.1, tab_height, facecolor='#0E1117', edgecolor=None))
main_area.text(0.45, tab_y+0.01, 'Raw Data', fontsize=12, ha='center', va='center', color='#D3D3D3')

main_area.axhline(y=tab_y, xmin=0.01, xmax=0.99, color='#E61E50', linewidth=1)

# Running section
main_area.text(0.5, 0.78, 'Running Compare Metrics', fontsize=18, fontweight='bold', ha='center')

# Progress bar - completed
main_area.add_patch(patches.Rectangle((0.1, 0.74), 0.8, 0.02, facecolor='#3A79C4', edgecolor=None))

main_area.text(0.5, 0.7, 'Analysis complete!', fontsize=14, ha='center')

# Table header
main_area.text(0.5, 0.65, 'Metrics Comparison by Event Type', fontsize=18, fontweight='bold', ha='center')

# Create table data
data = [
    ['FDA Decisions', '9678.7764', '7877.8633', '38.4615', '25.2209', '324.8421', '239.'],
    ['Phase 3 Data', '9368.5761', '9246.2261', '25', '25', '300.8328', '299.'],
    ['Phase 1/2 Data', '21982.0642', '198.1907', '50', '0', '709.2737', '6.'],
    ['Data Announcements', '8356.995', '7387.2621', '37.7551', '23.756', '282.8643', '223.']
]

# Table headers
headers = ['', 'ATM_IV', 'ATM_IV_std', 'IV_Percentile', 'IV_Percentile_std', 'Implied_Move', 'Implied_Move_std']

# Calculate cell dimensions
table_width = 0.8
table_height = 0.25
n_rows = len(data) + 1  # +1 for header
n_cols = len(headers)
cell_width = table_width / n_cols
cell_height = table_height / n_rows

# Draw table
table_top = 0.6
table_left = 0.1

# Draw header row with background
for i, header in enumerate(headers):
    cell_x = table_left + i * cell_width
    cell_y = table_top
    if i > 0:  # Skip the first empty cell
        main_area.add_patch(patches.Rectangle((cell_x, cell_y), cell_width, cell_height, 
                                              facecolor='#262730', edgecolor='#3B3B3B', linewidth=0.5))
        main_area.text(cell_x + cell_width/2, cell_y + cell_height/2, header, 
                        ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        main_area.add_patch(patches.Rectangle((cell_x, cell_y), cell_width, cell_height, 
                                              facecolor='#262730', edgecolor='#3B3B3B', linewidth=0.5))

# Draw data rows
for i, row in enumerate(data):
    for j, cell in enumerate(row):
        cell_x = table_left + j * cell_width
        cell_y = table_top - (i + 1) * cell_height
        main_area.add_patch(patches.Rectangle((cell_x, cell_y), cell_width, cell_height, 
                                             facecolor='#1E262D', edgecolor='#3B3B3B', linewidth=0.5))
        main_area.text(cell_x + cell_width/2, cell_y + cell_height/2, cell, 
                       ha='center', va='center', fontsize=10)

plt.tight_layout(pad=0)
plt.savefig("static/screenshots/dashboard/actual_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()

print("Actual-style dashboard image created at: static/screenshots/dashboard/actual_dashboard.png") 