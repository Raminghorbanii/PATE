#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc, precision_recall_curve

def compute_auc(y_true, scores):
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_score = auc(fpr, tpr)
    
    return auc_score 


# Compute the AUPRC for the model
def compute_auprc(y_true, scores):
    
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    auc_pr_score = auc(recall, precision)
    return auc_pr_score
