#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

from metrics.precision_at_k import *
from metrics.vus.metrics import get_range_vus_roc #Vus
from metrics.AUCs_Compute import *
from PATE_metric import PATE



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



# determine sliding window (period) based on ACF
def find_length(data):
    data = data.reshape(data.shape[0],)
    
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 100
        return local_max[max_local_max]+base
    except:
        return 100
    
    
# Function to create windows from the data
def create_windows(data, window_size = None, step=1):
    
    if window_size == None:
        window_size = find_length(data)
        print(f"window Size: {window_size}")
        
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    
    windows = np.array(windows)
    if step == 1:
        windows = windows.reshape(windows.shape[0], windows.shape[1])
        
    return windows


def window_to_points_Scores(data, window_scores, window_size = None):
    
    if window_size == None:
        window_size = find_length(data)
        
    accumulated_scores = np.zeros(len(data))
    window_count = np.zeros(len(data))
    
    for i, score in enumerate(window_scores):
        accumulated_scores[i:i+window_size] += score
        window_count[i:i+window_size] += 1
        
    average_scores = accumulated_scores / window_count
    return average_scores



def load_data(filename, train_data_ends = None):

    # Load data from the file
    with open(filename, 'r') as file:
        data = pd.read_csv(file, header=None).to_numpy()

    window_size = find_length(data[:,0])
    print(f"window Size: {window_size}")
    
    if train_data_ends == None:
        
        all_data_label = data[:, 1]
        first_anomaly_index = np.where(all_data_label == 1)[0][0]
        train_data_ends = first_anomaly_index - 500
        print(f"First Anomaly Indx: {first_anomaly_index}, Train Data ends at:{train_data_ends}")
        
    # Split data into training and testing sets
    train_data = data[:train_data_ends, 0].astype(float)
    test_data = data[train_data_ends:, 0].astype(float)
    test_label = data[train_data_ends:, 1]
    
    # Normalize training and testing data using StandardScaler
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data.reshape(-1, 1))
    test_data = scaler.transform(test_data.reshape(-1, 1))
    
    return train_data, test_data, test_label, window_size




def range_convers_new(label):
    '''
    input: arrays of binary values 
    output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    '''
    L = []
    i = 0
    j = 0 
    while j < len(label):
        # print(i)
        while label[i] == 0:
            i+=1
            if i >= len(label):
                break
        j = i+1
        # print('j'+str(j))
        if j >= len(label):
            if j==len(label):
                L.append((i,j-1))

            break
        while label[j] != 0:
            j+=1
            if j >= len(label):
                L.append((i,j-1))
                break
        if j >= len(label):
            break
        L.append((i, j-1))
        i = j
    return L
    


def plotFigures(data, label, score, score2, score3, score4, slidingWindow, color_box, plotRange=None, save_plot = False, plot_1_name = 'Real Data',  plot_2_name = 'Perfect Model',  plot_3_name = 'Model 1 (MVN)',  plot_4_name = 'Model 2 (AE)',  plot_5_name = 'Random Score'):
    
    range_anomaly = range_convers_new(label)
    
    max_length = len(score)
    if plotRange is None:
        plotRange = [0, max_length]
    
    fig3 = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig3.add_gridspec(5, 1)  # Adjusted grid for 5 rows
    

    # Plotting the data with anomalies highlighted and labeling it as 'Dataset'
    f3_ax1 = fig3.add_subplot(gs[0, 0])
    plt.tick_params(labelbottom=False)
    plt.plot(data[:max_length], 'k')
    for r in range_anomaly:
        f3_ax1.axvspan(r[0], r[1], color='red', alpha=color_box)
    plt.xlim(plotRange)
    f3_ax1.text(0.02, 0.90, plot_1_name, transform=f3_ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    
    
    # Function to plot each anomaly score
    def plot_anomaly_score(ax, score, label_text):
        ax.plot(score[:max_length])
        for r in range_anomaly:
            ax.axvspan(r[0], r[1], color='red', alpha=color_box)
        ax.set_ylabel('score')
        ax.set_xlim(plotRange)
        ax.text(0.02, 0.90, label_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Plotting the anomaly scores in separate subplots
    f3_ax2 = fig3.add_subplot(gs[1, 0])
    plot_anomaly_score(f3_ax2, score, plot_2_name)

    f3_ax3 = fig3.add_subplot(gs[2, 0])
    plot_anomaly_score(f3_ax3, score2, plot_3_name)

    f3_ax4 = fig3.add_subplot(gs[3, 0])
    plot_anomaly_score(f3_ax4, score3, plot_4_name)

    f3_ax5 = fig3.add_subplot(gs[4, 0])
    plot_anomaly_score(f3_ax5, score4, plot_5_name)


    plt.show()
    
    # Optionally save the figure
    if save_plot:
        fig3.savefig(f'Real_World_Data_Example_Scores_{plot_1_name}.pdf', format='pdf')
        
    return fig3




def Example_evaluation_scores(pred, labels, vus_zone_size, e_buffer, d_buffer):
    
    #Vus
    vus_results = get_range_vus_roc(pred, labels, vus_zone_size) 
    
    #Auc
    auc_result = compute_auc(labels, pred)
    auc_pr = compute_auprc(labels, pred)

    #PATE
    pate =  PATE(labels, pred, e_buffer, d_buffer, n_jobs= 1, include_zero=False)
    
    score_list_simple = {
                  
                  "VUS_ROC": vus_results["VUS_ROC"],
                  "VUS_PR": vus_results["VUS_PR"],
                  
                  "AUC":auc_result, 
                  "AUC_PR":auc_pr,
                  
                  "PATE":pate,
                  }

    for key in score_list_simple:
        score_list_simple[key] = round(score_list_simple[key], 2)
    
    # return score_list, score_list_simple
    return score_list_simple


