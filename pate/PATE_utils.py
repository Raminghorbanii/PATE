#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import Libraries
import numpy as np
from itertools import groupby
from operator import itemgetter
from sklearn.metrics import auc
import logging
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def convert_vector_to_events_PATE(vector_array):
    
    """
    Convert a binary NumPy array (indicating 1 for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an interval [i, i] which is one point.

    :param vector: a NumPy array of elements belonging to {0, 1}
    :return: a list of tuples, each tuple representing the start and stop of each event
    """
    
    # Find indices where the value is 1
    positive_indexes = np.where(vector_array > 0)[0]

    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix : ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    
    # Consistent conversion in case of range anomalies (for indexes):
    events = [(x, y) for (x,y) in events]
        
    return(events)



def convert_events_to_array_PATE(ranges, time_series_length):
    """
    Convert a list of ranges (events) into a NumPy array representation.

    Args:
    ranges: List of lists, where each inner list represents a range [start, end].
    time_series_length: Total length of the time series.

    Returns:
    A NumPy array with ones in the indices of the ranges and zeros elsewhere.
    """
    array = np.zeros(time_series_length, dtype=int)
    for start, end in ranges:
        array[start:end + 1] = 1
    return array




def categorize_predicted_ranges_with_ids(prediction_ranges, label_anomaly_ranges, e_buffer, d_buffer, time_series_length):
    
    """
    Categorize predicted anomaly ranges in a time series relative to label anomaly ranges and buffer zones,
    and associate them with unique IDs.

    Args:
    prediction_ranges: List of tuples, each tuple representing the start and end of a predicted range (event).
    label_anomaly_ranges: List of tuples, each tuple representing the start and end of an actual (labeled) anomaly range.
    e_buffer: Integer representing the max size of the pre-buffer zone (before each labeled anomaly range).
    d_buffer: Integer representing the max size of the post-buffer zone (after each labeled anomaly range).
    time_series_length: Total length of the time series.

    Returns:
    - A dictionary categorizing each segment of the predicted ranges into 'pre_buffer', 'true_detection', 'post_buffer', 'partial_missed', and 'outside' along with their corresponding IDs.
    - Anomaly Events with their ids
    """


    # Initialize the dictionary to hold categorized ranges with IDs
    categorized_ranges_with_ids = {'pre_buffer': [], 'true_detection': [], 'post_buffer': [], 'outside': [], 'partial_missed': []}

    # Assign an ID to each actual labeled anomaly range
    label_anomaly_ids = {i: (start, end) for i, (start, end) in enumerate(label_anomaly_ranges)}

    # Dictionary to track covered points in each actual labeled anomaly
    covered_points_in_label = {anomaly_id: set() for anomaly_id in label_anomaly_ids}

    # Initialize a dictionary to track the end of each post-buffer zone
    post_buffer_end_points = {}

    # Loop through each predicted range        
    for pred_start, pred_end in prediction_ranges:
        segment_start = pred_start

        # Compare with each label anomaly range
        for anomaly_id, (label_start, label_end) in label_anomaly_ids.items():
            
            ##############
            ##############
            
            # Define the end of buffer zones considering the next actual start
            
            next_label_start = label_anomaly_ranges[anomaly_id + 1][0] if anomaly_id + 1 < len(label_anomaly_ranges) else time_series_length # we check if there is a next label anomaly. If yes, then we get access to the start of that label anomaly range
            
            post_buffer_end = min(label_end + d_buffer, next_label_start - 1) # post-buffer zone can be ended sooner if the zone reaches the next label anomaly range 
            
            post_buffer_end_points[anomaly_id] = post_buffer_end #Keep tracking of the end of each post-buffer zone to have the size of the post-buffer zone
            
            # Determine pre_buffer_start considering the previous post-buffer's end if applicable
            previous_post_buffer_end = post_buffer_end_points.get(anomaly_id - 1, -1) #Return -1 when we process the first one and there is not any buffer zone before 
            
            pre_buffer_start = max(0, label_start - e_buffer, previous_post_buffer_end + 1) 
            pre_buffer_end = label_start - 1

            ############## 
            ##############
            
            # Categorize the predicted range with ID
            
            # Categorizing the outside range before the pre-buffer ranges
            if segment_start < pre_buffer_start:
                outside_end = min(pred_end, pre_buffer_start - 1)
                if segment_start <= outside_end:
                    categorized_ranges_with_ids['outside'].append({'range': [segment_start, outside_end], 'id': 'outside'})
        
        
            # Categorizing the Pre-buffer range
            if segment_start <= pre_buffer_end: 
                pre_buffer_segment_start = max(segment_start, pre_buffer_start)
                pre_buffer_segment_end = min(pred_end, pre_buffer_end)

                if pre_buffer_segment_start <= pre_buffer_segment_end:
                    categorized_ranges_with_ids['pre_buffer'].append({'range': [pre_buffer_segment_start, pre_buffer_segment_end], 'pre_buffer_start': pre_buffer_start, 'id': f"{anomaly_id}-pre"})
                    
                segment_start = pre_buffer_segment_end + 1 # This updated puts us in the next zones


            # Categorizing the true detected anoamly ranges 
            if segment_start <= label_end:
                actual_segment_end = min(pred_end, label_end)
                if segment_start <= actual_segment_end:
                    categorized_ranges_with_ids['true_detection'].append({'range': [segment_start, actual_segment_end], 'id': f"{anomaly_id}-true_detection"})
                    
                    # Mark these points as covered in the label anomaly range
                    covered_points_in_label[anomaly_id].update(range(segment_start, actual_segment_end + 1)) # Plus 1 is due to the python range function as we would like to also include the last point in the detected range as teh covered point
                    
                segment_start = actual_segment_end + 1 # This updated puts us in the next zones


            # Categorizing the post_buffer range
            if segment_start <= post_buffer_end:
                buffer_segment_end = min(pred_end, post_buffer_end)
                if segment_start <= buffer_segment_end:
                    categorized_ranges_with_ids['post_buffer'].append({'range': [segment_start, buffer_segment_end], 'post_buffer_end': post_buffer_end, 'id': f"{anomaly_id}-buf"})
                    
                segment_start = buffer_segment_end + 1 # This updated puts us in the next zones


        # Categorizing the outside range after the post buffer range
        if segment_start <= pred_end:
            categorized_ranges_with_ids['outside'].append({'range': [segment_start, pred_end], 'id': 'outside'})


    # Identify and categorize missed points in partially detected label anomalies
    for anomaly_id, (start, end) in label_anomaly_ids.items():
        labaled_points = set(range(start, end + 1))
        if labaled_points != covered_points_in_label[anomaly_id] and covered_points_in_label[anomaly_id]:
            missed_points = labaled_points - covered_points_in_label[anomaly_id]
            for point in missed_points:
                # Represent each missed point as a range
                categorized_ranges_with_ids['partial_missed'].append({'range': [point, point], 'id': f"{anomaly_id}-partial_missed"})


    return categorized_ranges_with_ids





def cal_Wtp_postbuffer(predicted_range, label_anomaly_range, buffer_end_point):
    """
    Calculate weights for predicted points within the post-buffer range of an anomaly.

    Args:
    predicted_range: Tuple representing the start and end of the predicted range.
    label_anomaly_range: Tuple representing the start and end of the label anomaly range.
    buffer_end_point: Integer indicating the end point of the buffer zone following the anomaly.

    Returns:
    A list of weights for each point in the predicted range within the post-buffer zone.
    """

    label_start, label_end = label_anomaly_range
    post_buffer_start = label_end + 1
    post_buffer_end = buffer_end_point
    
    # Calculate the maximum possible distance for normalization
    max_possible_distance = sum([abs(post_buffer_end - y) for y in range(label_start, label_end + 1)])
    weights = []

    # Assign weights to points in the predicted range within the post-buffer zone
    for x in range(predicted_range[0], predicted_range[1] + 1):
        if post_buffer_start <= x <= post_buffer_end: #Double check if we are in the post buffer zone
            distance = sum([abs(x - y) for y in range(label_start, label_end + 1)])
            normalized_distance = distance / max_possible_distance
            weight = 1 - normalized_distance  # Weight decreases with distance from anomaly
            weights.append(weight)
    
    return weights




def cal_Wtp_prebuffer(predicted_range, label_anomaly_range, prebuffer_start_point):
    """
    Calculate weights for predicted points within the pre-buffer range of an anomaly.

    Args:
    predicted_range: Tuple representing the start and end of the predicted range.
    label_anomaly_range: Tuple representing the start and end of the label anomaly range.
    prebuffer_start_point: Integer indicating the start point of the pre-buffer zone before the anomaly.

    Returns:
    A list of weights for each point in the predicted range within the pre-buffer zone.
    """

    label_start, label_end = label_anomaly_range
    pre_buffer_start = prebuffer_start_point
    pre_buffer_end = label_start - 1
    
    # Calculate the maximum possible distance for normalization
    max_possible_distance = sum([abs(y - pre_buffer_start) for y in range(label_start, label_end + 1)])
    weights = []

    # Assign weights to points in the predicted range within the pre-buffer zone
    for x in range(predicted_range[0], predicted_range[1] + 1):
        if pre_buffer_start <= x <= pre_buffer_end: #Double check if we are actually in the pre-buffer zone
            distance = sum([abs(y - x) for y in range(label_start, label_end + 1)])
            normalized_distance = distance / max_possible_distance
            weight = 1 - normalized_distance  # Weight decreases with proximity to anomaly
            
            weights.append(weight)
            
    return weights




def cal_Wfn_partial_missed(predicted_range, label_anomaly_range, delta_buffer):

    """
    Calculate weights for partially missed anomaly points in a prediction.
    
    Args:
    predicted_range: Tuple representing the start and end of the predicted range.
    label_anomaly_range: Tuple representing the start and end of the label anomaly range.
    delta_buffer: Integer indicating the size of the buffer zone around the label anomaly based on the coverage level of actual anomaly.
    
    Returns:
    A list of weights for each point in the predicted range
    """
    
    label_start, label_end = label_anomaly_range
    

    buffer_end = label_start + delta_buffer # The bigger the size of this buffer, the less strict we are on detecting onset response time. So, if we detect more coverage of the actual anomaly, in fact we are less strict as we could detect more, then we make this buffer range bigger. So this is why I connected this buffer range by the coverage of the detection in the actual anomaly event. 
    
    weights_FN = []

    # Calculate the maximum possible distance for normalization
    max_distance = sum([abs(label_end - y) for y in range(label_start, label_end + 1)])

    for point in range(predicted_range[0], predicted_range[1] + 1):
        if point <= buffer_end:
            weight = 1  # Weight is 1 within the buffer zone
        else:
            distance = sum([abs(point - y) for y in range(label_start, buffer_end + 1)])

            normalized_distance = distance / max_distance if max_distance > 0 else 0
            weight = 1 - normalized_distance  # Weight decreases beyond the buffer zone
        weights_FN.append(weight)
    
    return weights_FN





def extract_id(id_string):
    """
    Extracts the numeric part from an ID string.

    This function parses a given string, extracting and returning the numeric portion as an integer. 
    If the string contains no numeric characters, the function returns None.

    Args:
    id_string: A string containing the ID from which the numeric part needs to be extracted.

    Returns:
    An integer representing the numeric part of the ID string, or None if no numeric part is found.
    """

    # Extracts the numeric part from the ID string
    numeric_part = ''.join(filter(str.isdigit, id_string))
    return int(numeric_part) if numeric_part else None




   
    
def apply_weights(categorized_ranges_with_ids, label_anomaly_ranges):

    """
    Apply weights to categorized predictions based on their categories and corresponding IDs. This function is crucial for computing precision and recall in our time-series anomaly detection.

    Args:
    categorized_ranges_with_ids: Dictionary containing predictions categorized as 'actual', 'partial_missed', 
                                 'post-buffer', 'pre_buffer', and 'outside', each associated with unique IDs.
    label_anomaly_ranges: List of tuples, where each tuple indicates the start and end of an actual anomaly range.

    Returns:
    Precision and recall values calculated based on the sum of the weighted True Positives (TPs), False Positives (FPs), and False Negatives (FNs) across all categories.
    """


    # Initialize weight dictionaries
    
    weights = {'TP': [], 'FP': [], 'FN': []}

    for category, ranges_with_ids in categorized_ranges_with_ids.items():

        for item in ranges_with_ids:
            pred_range, id = item['range'], item['id']
            label_id = extract_id(id)  # Extracts the numeric part from the ID string
    
            ############################
            
            if category == 'true_detection': # For actual we only have TPs as we detected correctly.
                weights['TP'].extend([1] * (pred_range[1] - pred_range[0] + 1))
            
            ############################
            
            if category == 'partial_missed': # For partial missed one we only have FNs 
            
                # Find corresponding actual range with the same label_id
                actual_ranges = [range_info['range'] for range_info in categorized_ranges_with_ids['true_detection'] if extract_id(range_info['id']) == label_id]
                if actual_ranges:
                    # There's only one actual range per label_id
                    actual_range = actual_ranges[0]  
                    actual_range_length = actual_range[1] - actual_range[0] + 1
                    
                    # Use the actual range length as delta_buffer for this partial_missed
                    weights['FN'].extend(cal_Wfn_partial_missed(pred_range, label_anomaly_ranges[label_id], actual_range_length))
                    
            
            ############################
                
            elif category in ['post_buffer', 'pre_buffer']:
                
                if category == 'post_buffer': # For buffer zones we have TPs and FPs (1-TPs)
                    weights['TP'].extend(cal_Wtp_postbuffer(pred_range, label_anomaly_ranges[label_id], item['post_buffer_end']))

                    weights['FP'].extend(1 - w for w in cal_Wtp_postbuffer(pred_range, label_anomaly_ranges[label_id], item['post_buffer_end'])) 
                    
                    
                else:
                    # Check if pre_buffer is followed by an actual range because if it is not followed by an detected actual category then we have to cnosider this as Outside. 
                    if any(extract_id(item['id']) == label_id for item in categorized_ranges_with_ids['true_detection']):

                        weights['TP'].extend(cal_Wtp_prebuffer(pred_range, label_anomaly_ranges[label_id], item['pre_buffer_start']))

                        weights['FP'].extend(1 - w for w in cal_Wtp_prebuffer(pred_range, label_anomaly_ranges[label_id], item['pre_buffer_start']))
                    
                    else: # If we did not have any correct detection in actual range, the pre-buffer is assummed as total FP (As it is now considered in Outside category)
                        weights['FP'].extend([1] * (pred_range[1] - pred_range[0] + 1))

            
            ############################
            
            elif category == 'outside':
                weights['FP'].extend([1] * (pred_range[1] - pred_range[0] + 1))

    ############################
    
    # Check for total miss of the label anomaly (assigning the FNs)
    for i, label_range in enumerate(label_anomaly_ranges):
        if not any(extract_id(item['id']) == i for item in categorized_ranges_with_ids['true_detection']):
            
            weights['FN'].extend([1] * (label_range[1] - label_range[0] + 1))
                
    ############################
    
    # Summing the weights for each category
    summed_weights = {key: sum(value) for key, value in weights.items()}

    ############################
    ############################
    
    precision = summed_weights['TP'] / (summed_weights['TP'] + summed_weights['FP']) if (summed_weights['TP'] + summed_weights['FP']) > 0 else 0
    recall = summed_weights['TP'] / (summed_weights['TP'] + summed_weights['FN']) if (summed_weights['TP'] + summed_weights['FN']) > 0 else 0
    
    return precision, recall
    





def clean_and_compute_auc_pr(recall, precision):
    """
    Calculate the Area Under the Curve for the Precision-Recall curve (AUC-PR)
    on cleaned data, where recall values are non-decreasing.

    This function is specifically designed to handle the case where recall
    values might not be monotonic due to unique characteristics of the
    evaluation metric. It cleans the precision-recall curve by removing
    points where recall decreases. This ensures the monotonicity of recall
    values which is a requirement for calculating a meaningful AUC-PR.

    Args:
        recall (list or numpy.ndarray): Array of recall values. Should correspond
                                        to precision array. It might contain
                                        non-monotonic values.
        precision (list or numpy.ndarray): Array of precision values corresponding
                                           to recall values.

    Returns:
        float: The calculated AUC-PR on the cleaned precision-recall data.

    Example:
        >>> recall = [0.0, 0.1, 0.2, 0.15, 0.25]
        >>> precision = [1.0, 0.9, 0.8, 0.85, 0.75]
        >>> auc_pr = clean_and_compute_auc_pr(recall, precision)
        >>> print(auc_pr)
    """

    # Initialize lists to store cleaned precision and recall values
    clean_precision = []
    clean_recall = []
    prev_recall = -1  # Initial previous recall value set to -1

    # Iterate through each precision-recall pair
    for p, r in zip(precision, recall):
        # Include the point only if the recall is non-decreasing
        if r >= prev_recall:
            clean_precision.append(p)
            clean_recall.append(r)
            prev_recall = r

    # Calculate and return the AUC-PR on the cleaned data
    return auc(clean_recall, clean_precision)




def generate_buffer_points(max_buffer_size, num_splits, include_zero=True):
    """
    Generate evenly spaced buffer points.

    Args:
    max_buffer_size (int): The max size of the buffer.
    num_splits (int): Number of segments to split max_buffer_size into.
    include_zero (bool): Whether to include 0 in the range.

    Returns:
    np.ndarray: Array of buffer points.
    """
    
    if include_zero:
        start_point = 0
        num_points = num_splits + 1
    else:
        start_point = max_buffer_size / num_splits
        num_points = num_splits

    # Use np.linspace to generate evenly spaced points
    buffer_points = np.linspace(start_point, max_buffer_size, num=num_points, dtype=int)

    return buffer_points






def ACF_find_buffer_size(data):
    """
    Determines the buffer size around anomalies in time series data based on autocorrelation analysis,
    inspired by the method proposed in the VUS paper [1] for finding the appropriate sliding window. 
    This function computes the autocorrelation function (ACF) for a given time series and identifies 
    the most significant local maximum as the suggested buffer size. If no significant local maxima 
    are found, a default buffer size is used, which is commonly applied in research when working with 
    large benchmark datasets. For further details on default values and methodology, please 
    visit the official GitHub page of the VUS paper.

    Parameters:
    - data (array): The input time series data as a one-dimensional NumPy array.

    Returns:
    - int: The buffer size based on ACF local maxima or a default value if no suitable maxima are found.

    Reference:
    [1] Paparrizos, J., Boniol, P., Palpanas, T., Tsay, R. S., Elmore, A., & Franklin, M. J. (2022). Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. Proceedings of the VLDB Endowment, 15(11), 2774-2787.
    """
    # Ensure data is one-dimensional
    if data.ndim != 1:
        return 0  # Only process if data is one-dimensional

    # Limit data size for performance
    data = data[:min(20000, len(data))]

    # Calculate autocorrelation
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    # Find local maxima in the ACF
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        # Select the most prominent local maximum
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        # Validate local maxima within an acceptable range
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            raise ValueError("No suitable local maxima found.")
        return local_max[max_local_max] + base
    except:
        # Return default setting with an informative message if no local maxima are found
        print("No local maxima found; returning the default buffer size of 100.")
        return 100
    
    
