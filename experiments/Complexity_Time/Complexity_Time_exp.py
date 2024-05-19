#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

import sys
import os

try:
    # This will work when the script is run as a file
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Use an alternative method to set the current_dir when __file__ is not defined
    current_dir = os.getcwd()  # Gets the current working directory

project_root = os.path.dirname(os.path.dirname(current_dir))  # Two levels up
sys.path.append(project_root)

from pate.PATE_metric import PATE
from pate.PATE_utils import convert_events_to_array_PATE
from metrics.vus.metrics import get_range_vus_roc #VUS
from metrics.AUCs_Compute import *


def generate_anomalies(time_series_length, anomaly_size, ratio):
    """
    Generate anomaly ranges based on the specified ratio of the time series length.
    """
    num_anomalies = int((time_series_length * ratio) // anomaly_size)
    anomaly_ranges = []
    for i in range(num_anomalies):
        start = i * anomaly_size + i * 10  # Simple spacing between anomalies
        end = start + anomaly_size
        anomaly_ranges.append([start, end])
    return anomaly_ranges

def generate_synthetic_data(anomaly_ranges, predicted_ranges, time_series_length):
    """
    Generate synthetic binary labels and predictions for a given time series length and anomaly ranges.
    """
    label_array = convert_events_to_array_PATE(anomaly_ranges, time_series_length)
    predicted_array = convert_events_to_array_PATE(predicted_ranges, time_series_length)
    return label_array, predicted_array


def run_experiments(time_series_lengths, anomaly_ratios, anomaly_size, e_buffer, d_buffer):
    results = []
    for length in time_series_lengths:
        for ratio in anomaly_ratios:
            anomaly_ranges = generate_anomalies(length, anomaly_size, ratio)
            predicted_ranges = anomaly_ranges  # Perfect prediction scenario
            
            label_array, predicted_array = generate_synthetic_data(anomaly_ranges, predicted_ranges, length)
            
            # Measure PATE computation time
            start_time = time.time()
            _ = PATE(label_array, predicted_array, e_buffer, d_buffer, Big_Data=False, n_jobs=1, num_splits_MaxBuffer=1, include_zero=False, binary_scores=False)
            pate_duration = time.time() - start_time
            
            # Measure AUC-PR computation time
            start_time = time.time()
            _ = compute_auprc(label_array, predicted_array)
            auc_pr_duration = time.time() - start_time
            
            # Measure VUS computation time
            start_time = time.time()
            _ = get_range_vus_roc(predicted_array, label_array, 20)  # Adjust vus_zone_size as needed
            vus_duration = time.time() - start_time
            
            print(f"Length: {length}, Ratio: {ratio}, Anomalies: {len(anomaly_ranges)}")
            print(f"PATE computation time: {pate_duration:.4f} seconds")
            print(f"AUC-PR computation time: {auc_pr_duration:.4f} seconds")
            print(f"VUS computation time: {vus_duration:.4f} seconds")
            print("---")
            
            results.append((length, ratio, pate_duration, auc_pr_duration, vus_duration))
    return results


# Parameters
time_series_lengths = [1000, 10000, 100000]
anomaly_ratios = [0.02, 0.05, 0.1]
anomaly_size = 20  # Fixed size of each anomaly event
e_buffer = 20
d_buffer = 20

# Run the experiments
experiment_results = run_experiments(time_series_lengths, anomaly_ratios, anomaly_size, e_buffer, d_buffer)


# Save experiment results to a pickle file
# results_file = 'Time_Complexity_Data_Job1.pkl'
# with open(results_file, 'wb') as file:  # Note the 'wb' mode for writing binary files
#     pickle.dump(experiment_results, file)


# # Load experiment results from the pickle file
# with open(results_file, 'rb') as file:  # Note the 'rb' mode for reading binary files
#     loaded_results = pickle.load(file)



# Initialize a dictionary to store data for plotting
plot_data = {}

# Adjusting the results appending section to include AUC-PR and VUS times
for length, ratio, pate_duration, auc_pr_duration, vus_duration in experiment_results:
    key = f'PATE {ratio}'
    if key not in plot_data:
        plot_data[key] = {'lengths': [], 'times': []}
    plot_data[key]['lengths'].append(length)
    plot_data[key]['times'].append(pate_duration)
    
    key = f'AUC-PR {ratio}'
    if key not in plot_data:
        plot_data[key] = {'lengths': [], 'times': []}
    plot_data[key]['lengths'].append(length)
    plot_data[key]['times'].append(auc_pr_duration)
    
    key = f'VUS {ratio}'
    if key not in plot_data:
        plot_data[key] = {'lengths': [], 'times': []}
    plot_data[key]['lengths'].append(length)
    plot_data[key]['times'].append(vus_duration)

# Sorting data by length for each key (PATE, AUC-PR, VUS for each ratio)
for key in plot_data:
    lengths, times = zip(*sorted(zip(plot_data[key]['lengths'], plot_data[key]['times'])))
    plot_data[key] = {'lengths': lengths, 'times': times}



# Define colors for each anomaly ratio for clarity
ratio_colors = {0.02: 'red', 0.05: 'green', 0.1: 'blue'}

# Define line styles for each method
method_styles = {'PATE': '-', 'AUC-PR': '--', 'VUS': '-.'}

# Define markers for each method
method_markers = {'PATE': 'o', 'AUC-PR': 's', 'VUS': '^'}


plt.figure(figsize=(10, 6))

# Set sizes for text elements
font_size = 14  # For axis labels and title
legend_font_size = 14  # For legend text
marker_size = 8  # For marker size
tick_label_size = 12  # For the tick labels on both axes

for key, data in plot_data.items():
    # Split the key to get the method and ratio
    method, str_ratio = key.split()
    ratio = float(str_ratio)
    
    # Get the corresponding color, line style, and markers
    color = ratio_colors[ratio]
    linestyle = method_styles[method]
    markers = method_markers[method]
    
    # Construct the label
    label = f'{method} Ratio {ratio}'
    
    # Plot data with specified markers and sizes
    plt.plot(data['lengths'], data['times'], label=label, color=color, linestyle=linestyle, marker=markers, markersize=marker_size)

# Set log scale for axes
plt.xscale('log')
plt.yscale('log')

# Set axis labels with a larger font size
plt.xlabel('Time Series Length', fontsize=font_size)
plt.ylabel('Computation Time (seconds)', fontsize=font_size)

# Customize the tick labels on both axes to increase their size
plt.tick_params(axis='both', which='major', labelsize=tick_label_size)

# Adjust the legend to incorporate a larger font size, and position it below the plot
plt.legend(bbox_to_anchor=(0.5, -0.13), loc='upper center', ncol=3, fontsize=legend_font_size)

# Enable grid
plt.grid(True)

# Adjust layout to accommodate the legend below the plot
plt.tight_layout()

# Save the figure to a PDF file
# plt.savefig('Time_Complexity.pdf', format='pdf')

# Display the plot
plt.show()


