#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


from sklearn.metrics import f1_score
from metrics.f1_score_f1_pa import * # used for PA-F1 
from metrics.AUC import * # Range AUC
from metrics.affiliation.generics import convert_vector_to_events  #Affiliation
from metrics.affiliation.metrics import pr_from_events #Affiliation
from metrics.vus.models.feature import Window #VUS
from metrics.vus.metrics import get_range_vus_roc #VUS
from metrics.eTaPR_pkg import *
from metrics.eTaPR_pkg import f1_score_etapr
from metrics.AUCs_Compute import *
from metrics.Range_Based_PR import *

from PATE_metric import PATE
from PATE_utils import convert_events_to_array_PATE, categorize_predicted_ranges_with_ids


def evaluate_all_metrics(pred, labels, vus_zone_size = 20, e_buffer=20, d_buffer=20):
    
    #Affliation
    events_pred = convert_vector_to_events(pred) 
    events_gt = convert_vector_to_events(labels)
    Trange = (0, len(pred))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    
    #Vus
    vus_results = get_range_vus_roc(pred, labels, vus_zone_size) 
    
    #Auc
    auc_result = compute_auc(labels, pred)
    auc_pr = compute_auprc(labels, pred)

    #Ours
    pate =  PATE(labels, pred, e_buffer, d_buffer, Big_Data = False, n_jobs= 1, num_splits_MaxBuffer = 1, include_zero = False, binary_scores = False)
    pate_f1 = PATE(labels, pred, e_buffer, d_buffer, Big_Data = False, n_jobs= 1, num_splits_MaxBuffer = 1, include_zero = False, binary_scores = True)
    
    #R-based
    Rbased_precision, Rbased_recall, Rbased_f1_score = get_F1Score_RangeBased(labels, pred) 

    #eTaPR
    eTaPR_precision, eTaPR_recall, eTaPR_f1_score = f1_score_etapr.get_eTaPR_fscore(labels, pred, theta_p = 0.5, theta_r = 0.01, delta=0) #Default Settings from the original paper

    #Standard Original F1-Score
    original_F1Score = f1_score(labels, pred)

    #Point-Adj
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred, labels) #Effect on others!!


    score_list_simple = {
                   "original_F1Score":original_F1Score, 

                   "pa_precision":pa_precision, 
                   "pa_recall":pa_recall, 
                   "pa_f_score":pa_f_score,
                  
                   "Rbased_precision":Rbased_precision, 
                   "Rbased_recall":Rbased_recall,     
                   "Rbased_f1score":Rbased_f1_score, 
                  
                  "eTaPR_precision":eTaPR_precision, 
                  "eTaPR_recall":eTaPR_recall,     
                  "eTaPR_f1_score":eTaPR_f1_score,  
                  
                  "Affiliation precision": affiliation['precision'], 
                  "Affiliation recall": affiliation['recall'],
                  "Affliation F1score":  get_f_score(affiliation['precision'],affiliation['recall'] ),
                  
                  "VUS_ROC": vus_results["VUS_ROC"],
                  "VUS_PR": vus_results["VUS_PR"],
                  
                  "AUC":auc_result, 
                   "AUC_PR":auc_pr,
                  
                  "PATE":pate,
                  "PATE-F1":pate_f1,
                  }

    for key in score_list_simple:
        score_list_simple[key] = round(score_list_simple[key], 2)
    
    # return score_list, score_list_simple
    return score_list_simple





def synthetic_generator(label_anomaly_ranges, predicted_ranges, vus_zone_size = 20, e_buffer = 20, d_buffer = 20, time_series_length = 500):
    """
    Runs a synthetic data experiment given label and prediction ranges.
    
    Parameters:
    - label_anomaly_ranges: List of [start, end] ranges for actual anomalies.
    - predicted_ranges: List of [start, end] ranges for detected anomalies.
    - time_series_length: Total length of the time series.
    - vus_zone_size: Size of the VUS method buffer zone.
    - e_buffer: Eaely prebuffer size.
    - d_buffer: Delayed postbuffer size.
    
    Returns:
    - A dictionary containing the categorized ranges with IDs, predicted array, and label array.
    """
    categorized_ranges_with_ids = categorize_predicted_ranges_with_ids(
        predicted_ranges, label_anomaly_ranges, e_buffer, d_buffer, time_series_length)
    
    predicted_array = convert_events_to_array_PATE(predicted_ranges, time_series_length)
    label_array = convert_events_to_array_PATE(label_anomaly_ranges, time_series_length)
    
    return {
        "categorized_ranges_with_ids": categorized_ranges_with_ids,
        "predicted_array": predicted_array,
        "label_array": label_array
    }




