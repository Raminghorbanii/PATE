#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from metrics.eTaPR_pkg.DataManage import Range
from metrics.eTaPR_pkg import etapr, tapr
from metrics.f1_score_f1_pa import * # used for PA-F1 and general F1 calculation

from itertools import groupby
from operator import itemgetter

import numpy as np


def get_eTaPR_fscore(gt, pred, theta_p = 0.5, theta_r = 0.01, delta=0):
    
    def convert_vector_to_events_eTaPR(vector_array):
        
        """
        Convert a binary NumPy array (indicating 1 for the anomalous instances)
        to a list of events. The events are considered as durations,
        i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).
    
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
        # A positive index i is considered as the interval [i, i], so the last end index is the end of the range.
        events = [Range.Range(x, y, '') for (x,y) in events]
            
        return events


    events_eTaPR_pred = convert_vector_to_events_eTaPR(pred) 
    events_eTaPR_gt = convert_vector_to_events_eTaPR(gt)

    result = etapr.evaluate_w_ranges(events_eTaPR_gt, events_eTaPR_pred, theta_p=theta_p, theta_r=theta_r, delta=delta)
    precision_eTaPR = result['eTaP']
    recall_eTaPR = result['eTaR']
    f1_score_eTaPR = get_f_score(precision_eTaPR, recall_eTaPR)
    
    return precision_eTaPR, recall_eTaPR, f1_score_eTaPR