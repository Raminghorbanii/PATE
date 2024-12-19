#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve
from joblib import Parallel, delayed

from pate.PATE_utils import convert_vector_to_events_PATE, categorize_predicted_ranges_with_ids, apply_weights, generate_buffer_points, clean_and_compute_auc_pr



def compute_adjusted_scores(i, threshold, y_score, actual_anomaly_ranges, e_buffer, d_buffer, time_series_length):
    # Convert scores to binary classification based on the selected threshold
    binary_predicted = (y_score >= threshold).astype(int)
    predicted_ranges = convert_vector_to_events_PATE(binary_predicted)
    
    # Here we will categorize the ranges (Based on the selected threshold)
    categorized_predicted_ranges = categorize_predicted_ranges_with_ids(predicted_ranges, actual_anomaly_ranges, e_buffer, d_buffer, time_series_length)
    
    precision, recall = apply_weights(categorized_predicted_ranges, actual_anomaly_ranges)
    
    return precision, recall
    
    
def compute_f1_score(precision, recall):
# Calculate the F1 score from precision and recall
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)



def handle_binary_scores(y_true, y_score, e_buffer, d_buffer, num_splits_MaxBuffer, include_zero):
    """
    Handles the evaluation of binary scores for anomaly detection in time series data.
    
    This function iterates over combinations of early and delayed buffer sizes to
    compute precision and recall for the given binary scores. It calculates the F1 score
    for each combination, ultimately returning the average F1 score across all combinations.
    
    Args:
        y_true (np.ndarray): The true labels for the time series data.
        y_score (np.ndarray): The binary scores indicating predicted anomalies.
        e_buffer (int): The maximum size of the window for early detection buffering.
        d_buffer (int): The maximum size of the window for delayed detection buffering.
        num_splits_MaxBuffer (int): The number of buffer size splits to evaluate.
        include_zero (bool): Whether to include a buffer size of zero in the evaluation.
    
    Returns:
        float: The average F1 score across all combinations of buffer sizes.
    """

    # Convert true labels to ranges of actual anomalies
    actual_anomaly_ranges = convert_vector_to_events_PATE(y_true)
    time_series_length = len(y_true)

    f1_score_list = []
    
    # Generate combinations of early and delayed buffer sizes
    for selected_early_buffer in generate_buffer_points(e_buffer, num_splits_MaxBuffer, include_zero):
        for selected_delayed_buffer in generate_buffer_points(d_buffer, num_splits_MaxBuffer, include_zero):
            
            # Convert binary scores to predicted ranges based on the buffer sizes
            predicted_ranges = convert_vector_to_events_PATE(y_score)
            categorized_predicted_ranges = categorize_predicted_ranges_with_ids(
                predicted_ranges, actual_anomaly_ranges, selected_early_buffer, selected_delayed_buffer, time_series_length)
            
            # Calculate precision and recall for the current combination of buffer sizes
            precision, recall = apply_weights(categorized_predicted_ranges, actual_anomaly_ranges)
            
            # Compute the F1 score from the precision and recall values
            f1_score = compute_f1_score(precision, recall)
            f1_score_list.append(f1_score)

    # Calculate the average F1 score across all buffer size combinations
    if f1_score_list:
        average_f1_score = np.mean(f1_score_list)
    else:
        average_f1_score = 0.0
    
    return average_f1_score




def handle_continuous_scores(y_true, y_score, e_buffer, d_buffer, pos_label, sample_weight, n_jobs, drop_intermediate, Big_Data, num_desired_thresholds, num_splits_MaxBuffer, include_zero):
    """
    Evaluates continuous scores for anomaly detection, calculating the AUC of the
    modified precision-recall curve across different thresholds and buffer sizes.
    
    This function processes continuous prediction scores, applying a range of thresholds
    to binarize these scores. For each threshold and combination of buffer sizes,
    it calculates precision and recall, and subsequently computes the AUC of the
    precision-recall curve.
    
    Args:
        y_true (np.ndarray): The true labels for the time series data.
        y_score (np.ndarray): The continuous scores indicating the likelihood of anomalies.
        e_buffer (int): The maximum size of the window for early detection buffering.
        d_buffer (int): The maximum size of the window for delayed detection buffering.
        pos_label (int): The label considered as positive/anomalous.
        sample_weight (np.ndarray, optional): Weights for each sample.
        n_jobs (int): The number of jobs to run in parallel during threshold processing.
        drop_intermediate (bool): Whether to drop intermediate thresholds for faster processing.
        Big_Data (bool): Whether to use a reduced set of percentiles for thresholds to speed up processing.
        num_desired_thresholds (int): The desired number of thresholds to evaluate.
        num_splits_MaxBuffer (int): The number of buffer size splits to evaluate.
        include_zero (bool): Whether to include a buffer size of zero in the evaluation.
    
    Returns:
        float: The average AUC-PR across all combinations of buffer sizes.
    """

    # Initialize list to hold AUC-PR values for each buffer size combination
    auc_pr_list = []
    
    # Convert true labels to ranges of actual anomalies
    actual_anomaly_ranges = convert_vector_to_events_PATE(y_true)
    time_series_length = len(y_true)

    # Generate thresholds
    fps_orig, tps_orig, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    
    if drop_intermediate and len(tps_orig) > 2:
        # Drop unnecessary thresholds for simplification
        optimal_idxs = np.where(np.concatenate([[True], np.logical_or(np.diff(tps_orig[:-1]), np.diff(tps_orig[1:])), [True]]))[0]
        fps_orig, tps_orig, thresholds = fps_orig[optimal_idxs], tps_orig[optimal_idxs], thresholds[optimal_idxs]
    
    if Big_Data:
        # Reduce the number of thresholds for large datasets
        percentiles = np.linspace(100, 0, num_desired_thresholds)
        thresholds = np.percentile(thresholds, percentiles)
    
    # Iterate over combinations of buffer sizes
    for selected_early_buffer in generate_buffer_points(e_buffer, num_splits_MaxBuffer, include_zero):
        for selected_delayed_buffer in generate_buffer_points(d_buffer, num_splits_MaxBuffer, include_zero):
            
            # Use parallel processing to compute scores for each threshold
            results = Parallel(n_jobs=n_jobs)(delayed(compute_adjusted_scores)(
                i, threshold, y_score, actual_anomaly_ranges, selected_early_buffer, selected_delayed_buffer, time_series_length) for i, threshold in enumerate(thresholds))
            
            # Unpack precision and recall values
            precision, recall = zip(*results)
            
            # Prepare precision and recall for AUC-PR calculation
            precision = np.array(precision)
            recall = np.array(recall)
            precision = np.hstack(([1], precision))
            recall = np.hstack(([0], recall))
            
            # Calculate and append AUC-PR for current buffer size combination
            auc_pr = clean_and_compute_auc_pr(recall, precision)
            auc_pr_list.append(auc_pr)
    
    # Calculate the average AUC-PR across all combinations
    average_auc_pr = np.mean(auc_pr_list) if auc_pr_list else 0.0
    
    return average_auc_pr




def PATE(y_true, y_score, e_buffer=100, d_buffer=100, pos_label=1, sample_weight=None, n_jobs=1, drop_intermediate=True, Big_Data=True, num_desired_thresholds=250, num_splits_MaxBuffer=1, include_zero=True, binary_scores=False):
    """
    Primary function to evaluate anomaly detection in time series data, calculating either
    the average F1 score for binary scores or the average AUC of the precision-recall curve
    for continuous scores, across varying buffer sizes.
    
    This function decides whether to process the scores as binary or continuous based on the
    binary_scores flag and then calls the respective handler function for detailed processing.
    
    Args:
        y_true (np.ndarray): True binary labels for each data point in the time series.
        y_score (np.ndarray): Predicted scores; can be binary or continuous depending on the model.
        e_buffer (int): Early detection buffer size, specifying how far back to consider an early detection.
        d_buffer (int): Delayed detection buffer size, specifying how far forward to consider a detection still relevant.
        pos_label (int): The label of the positive class (typically 1 for anomalies).
        sample_weight (np.ndarray, optional): Weights for the samples, if applicable.
        n_jobs (int): Number of jobs to run in parallel for threshold processing.
        drop_intermediate (bool): Whether to drop intermediate thresholds to speed up processing.
        Big_Data (bool): Adjusts processing for large datasets by reducing the number of thresholds.
        num_desired_thresholds (int): The number of thresholds to evaluate for continuous scores.
        num_splits_MaxBuffer (int): The granularity of buffer size variation.
        include_zero (bool): Whether to include a buffer size of zero.
        binary_scores (bool): Flag indicating whether the provided scores are binary (True) or continuous (False).
    
    Returns:
        float: The average F1 score across all buffer size combinations for binary scores, or the average
        AUC-PR for continuous scores.
    """
    
    
    # Check if scores are binary and call the appropriate handler
    if binary_scores:
        result =handle_binary_scores(y_true, y_score, e_buffer, d_buffer, num_splits_MaxBuffer, include_zero)
    else:
        result = handle_continuous_scores(y_true, y_score, e_buffer, d_buffer, pos_label, sample_weight, n_jobs, drop_intermediate, Big_Data, num_desired_thresholds, num_splits_MaxBuffer, include_zero)
        


    return result





