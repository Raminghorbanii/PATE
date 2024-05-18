#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import random

from utils_Real_exp import load_data, create_windows, plotFigures, Example_evaluation_scores
from models_Real_exp import mvn_model, AE_model
from sklearn.preprocessing import MinMaxScaler

#############################

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


#############################
#############################
#############################

set_seed()
filename = 'TSB-UAD-Public/ECG/MBA_ECG14046_data_0.out' # Example of ECG data

train_data, test_data, test_label, window_size = load_data(filename, train_data_ends = None)

test_data = test_data[24250:25300] 
test_label = test_label[24250:25300]

window_size = 100
train_window = create_windows(train_data, window_size)
test_window = create_windows(test_data, window_size)


#############################
#############################
#############################

# iforest_anomaly_scores, _ = iforest_model(train_window, test_window, n_estimators=100)
# lof_anomaly_scores, _ = LOF_model(train_window, test_window, n_neighbors=10)
# pca_anomaly_scores, _ = pca_model(train_window, test_window)
# ocsvm_anomaly_scores, _ = ocsvm_model(train_window, test_window)

# For this example the following two models are selected. 
mvn_anomaly_scores, mvn_anomaly_scores_plot = mvn_model(train_window, test_window, window_size, n_components=4, random_state= 42)
AE_anomaly_scores, AE_anomaly_scores_plot = AE_model(train_data, test_data, window_size)

# Random uniform scores
random_scores = np.random.uniform(0, 1, len(test_data))

# Perfect model (the same as the test anomaly labels)
perfect_scores = test_label

#############################
#############################
#############################


test_data_plot = MinMaxScaler(feature_range=(0,1)).fit_transform(test_data.reshape(-1,1)).ravel()

plotFigures(test_data_plot, test_label, perfect_scores, mvn_anomaly_scores_plot, AE_anomaly_scores_plot, random_scores,  window_size, color_box = 0.4, save_plot=False, plot_1_name='ECG Data', plot_3_name='Model 1 (MVN)', plot_4_name='Model 2 (AE)') 


#############################
#############################
#############################

vus_zone_size = e_buffer = d_buffer = 50

perfect_evaluations = Example_evaluation_scores(perfect_scores, test_label, vus_zone_size, e_buffer, d_buffer)
mvn_evaluations = Example_evaluation_scores(mvn_anomaly_scores, test_label, vus_zone_size, e_buffer, d_buffer)
AE_evaluations = Example_evaluation_scores(AE_anomaly_scores, test_label, vus_zone_size, e_buffer, d_buffer)
Random_evaluations = Example_evaluation_scores(random_scores, test_label, vus_zone_size, e_buffer, d_buffer)

print(f"Perfect Score: {perfect_evaluations}")
print(f"MVN: {mvn_evaluations}")
print(f"AE: {AE_evaluations}")
print(f"Random: {Random_evaluations}")

