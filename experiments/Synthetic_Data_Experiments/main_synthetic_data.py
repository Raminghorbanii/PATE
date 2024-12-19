#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################

from utils_Synthetic_exp import evaluate_all_metrics, synthetic_generator

# Synthetic Data Experiments (Size of anomaly range is selected as 20 in the original paper)
label_anomaly_ranges = [[40,59]]  # Multiple actual anomaly ranges - you can select a range that you want to have as actual anomaly (A_k) 
predicted_ranges = [[30, 49]]  # Multiple predicted anomaly ranges - you can select a range as detection output (S)
vus_zone_size = e_buffer = d_buffer = 20

experiment_results = synthetic_generator(label_anomaly_ranges, predicted_ranges, vus_zone_size, e_buffer, d_buffer)
predicted_array = experiment_results["predicted_array"]
label_array = experiment_results["label_array"]


score_list_simple = evaluate_all_metrics(predicted_array, label_array, vus_zone_size, e_buffer, d_buffer)
print(score_list_simple)



# scenarios tested in the original paper:
# label_anomaly_ranges = [[40,59]]
# vus_zone_size = e_buffer = d_buffer = 20
    
# s_1 : predicted_ranges = [[20, 39]] 
# s_2 : predicted_ranges = [[30, 49]] 
# s_3 : predicted_ranges = [[40, 59]] 
# s_4 : predicted_ranges = [[50, 69]] 
# s_5 : predicted_ranges = [[60, 79]] 
# s_6 : predicted_ranges = [[30, 69]] 
# s_7 : predicted_ranges = [[40, 49]] 
# s_8 : predicted_ranges = [[50, 59]] 
# s_9 : predicted_ranges = [[40, 54]] 
# s_10 : predicted_ranges = [[45, 59]] 
