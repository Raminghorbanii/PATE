#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
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
from metrics.vus.metrics import get_range_vus_roc #VUS
from metrics.AUCs_Compute import *

# Label = Load your labels from the dataset
# scores = Load your anomaly scores

sum_of_ones = sum(Label)
total_elements = len(Label)
ratio_of_ones = sum_of_ones / total_elements
print("Ratio of 1s in the array:", ratio_of_ones)

# Measure AUC-PR computation time
start_time = time.time()
_ = compute_auprc(Label, scores)
auc_pr_duration = time.time() - start_time
print(f"AUC-PR computation time: {auc_pr_duration:.4f} seconds")


# Measure PATE computation time
e_buffer = 100
d_buffer = 100
start_time = time.time()
_ = PATE(Label, scores, e_buffer, d_buffer, Big_Data=False, n_jobs=1, num_splits_MaxBuffer=1, include_zero=False, binary_scores=False)
pate_duration = time.time() - start_time
print(f"PATE computation time: {pate_duration:.4f} seconds")

          
# Measure VUS computation time
start_time = time.time()
_ = get_range_vus_roc(scores, Label, 100)  # Adjust vus_zone_size as needed
vus_duration = time.time() - start_time
print(f"VUS computation time: {vus_duration:.4f} seconds")
