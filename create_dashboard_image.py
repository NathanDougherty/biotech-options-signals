"""
Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Create directory if it doesn't exist
os.makedirs("static/screenshots/dashboard", exist_ok=True)

# Create a dark-themed dashboard overview mockup
plt.style.use('dark_background')

# Create figure with specific aspect ratio to match the dashboard
fig = plt.figure(figsize=(12, 8), facecolor='#1E1E1E')
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[1, 1])

# Left sidebar area
ax_sidebar = plt.subplot(gs[:, 0])
ax_sidebar.axis('off')
sidebar_color = '#0E1117'
sidebar_rect = patches.Rectangle((0, 0), 1, 1, facecolor=sidebar_color)
ax_sidebar.add_patch(sidebar_rect)

# Title area
ax_title = plt.subplot(gs[0, 1])
ax_title.axis('off')
ax_title.text(0.5, 0.6, 'Biotech Options Signal Analysis', ha='center', fontsize=24, fontweight='bold')
ax_title.text(0.5, 0.3, 'Dashboard', ha='center', fontsize=24, fontweight='bold')

# Main content area
ax_content = plt.subplot(gs[1, 1])
ax_content.axis('off')

# Add sidebar elements
ax_sidebar.text(0.5, 0.95, 'Analysis Controls', ha='center', fontsize=14, fontweight='bold')
ax_sidebar.text(0.5, 0.9, 'Select Tickers', ha='center', fontsize=12)
ax_sidebar.text(0.5, 0.85, '[Dropdown]', ha='center', fontsize=10, bbox=dict(facecolor='#262730', edgecolor='gray', boxstyle='round,pad=0.5'))

ax_sidebar.text(0.5, 0.75, 'Filter by Event Type', ha='center', fontsize=12)
ax_sidebar.text(0.5, 0.7, 'All', ha='center', fontsize=10, bbox=dict(facecolor='#262730', edgecolor='gray', boxstyle='round,pad=0.5'))

ax_sidebar.text(0.5, 0.6, 'Analysis Type', ha='center', fontsize=12)
ax_sidebar.text(0.5, 0.55, 'Compare Metrics', ha='center', fontsize=10, bbox=dict(facecolor='#262730', edgecolor='gray', boxstyle='round,pad=0.5'))

ax_sidebar.text(0.5, 0.35, 'Documentation', ha='center', fontsize=14, fontweight='bold')
ax_sidebar.text(0.5, 0.3, 'Download README', ha='center', fontsize=10, bbox=dict(facecolor='#262730', edgecolor='gray', boxstyle='round,pad=0.5'))

# Add tabs
tabs_y = 0.95
ax_content.text(0.2, tabs_y, 'Analysis Results', ha='center', fontweight='bold', fontsize=12, color='white')
ax_content.text(0.4, tabs_y, 'Visualizations', ha='center', fontsize=12, color='gray')
ax_content.text(0.6, tabs_y, 'Raw Data', ha='center', fontsize=12, color='gray')
ax_content.axhline(y=tabs_y-0.05, xmin=0.05, xmax=0.95, color='gray', alpha=0.3)

# Add table headers
ax_content.text(0.5, 0.85, 'Metrics Comparison by Event Type', ha='center', fontsize=14, fontweight='bold')

# Create a sample table
n_rows, n_cols = 5, 6
table_data = np.zeros((n_rows, n_cols))

# Header row
header_labels = ['', 'ATM_IV', 'IV_std', 'IV_Percentile', 'Implied_Move', 'IV_Skew']
for i, label in enumerate(header_labels):
    ax_content.text(0.1 + i*0.15, 0.75, label, ha='center', fontsize=10, fontweight='bold')

# Data rows
row_labels = ['FDA Decisions', 'Phase 3 Data', 'Phase 1/2 Data', 'Data Announcements']
for i, label in enumerate(row_labels):
    ax_content.text(0.1, 0.7 - i*0.1, label, ha='left', fontsize=10)

# Sample data values
for i in range(4):
    for j in range(5):
        val = np.random.randint(10, 100) if j > 0 else ''
        if j == 1:  # ATM_IV
            val = f"{np.random.randint(80, 150)}"
        elif j == 2:  # IV_std
            val = f"{np.random.randint(5, 20)}"
        elif j == 3:  # IV_Percentile
            val = f"{np.random.randint(10, 99)}"
        elif j == 4:  # Implied_Move
            val = f"{np.random.randint(200, 500)}"
        elif j == 5:  # IV_Skew
            val = f"{np.random.randint(80, 120)/100}"
        ax_content.text(0.1 + (j+1)*0.15, 0.7 - i*0.1, str(val), ha='center', fontsize=10)

# Add table grid lines
for i in range(n_rows+1):
    ax_content.axhline(y=0.75 - 0.1*i, xmin=0.05, xmax=0.95, color='gray', alpha=0.3)
for i in range(n_cols+1):
    ax_content.axvline(x=0.05 + 0.15*i, ymin=0.35, ymax=0.75, color='gray', alpha=0.3)

# Add progress bar
ax_content.text(0.5, 0.2, 'Analysis complete!', ha='center', fontsize=12)
rect = patches.Rectangle((0.1, 0.25), 0.8, 0.03, facecolor='#0078ff')
ax_content.add_patch(rect)

plt.tight_layout()
plt.savefig("static/screenshots/dashboard/dashboard_overview.png", dpi=150, bbox_inches='tight')
plt.close()

print("Dashboard overview image created at: static/screenshots/dashboard/dashboard_overview.png") 