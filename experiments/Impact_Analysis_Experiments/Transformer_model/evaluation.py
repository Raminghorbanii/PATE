#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Utils import calculate_reconstruction_errors, evaluate_rec



def Reconstruction_Errors_Clculation(model, train_dataloader, test_dataloader, cfg):
    """
    Calculate reconstruction errors for both training and test datasets.
    
    Args:
        model: The trained model for which the reconstruction errors are to be calculated.
        train_dataloader: DataLoader object containing the training dataset.
        test_dataloader: DataLoader object containing the test dataset.
        cfg: Configuration object containing necessary parameters.
        
    Returns:
        tuple: Contains reconstruction errors for training dataset, reconstruction errors for test dataset, and true labels for test dataset.
    """
    
    reconstruction_errors_point_tr = calculate_reconstruction_errors(model, train_dataloader, cfg)
    reconstruction_errors_point_te, true_labels_point = calculate_reconstruction_errors(model, test_dataloader, cfg, test_mode=True)

    return reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point



def evaluate_based_on_reconstruction(reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg):
    """
    Evaluate the model's performance based on the calculated reconstruction errors.
    
    Args:
        reconstruction_errors_point_tr: Reconstruction errors for the training dataset.
        reconstruction_errors_point_te: Reconstruction errors for the test dataset.
        true_labels_point: True labels for the test dataset.
        cfg: Configuration object containing necessary parameters.
        
    Returns:
        dict: Evaluation results based on point-wise reconstruction errors.
    """
    
    point_wise_results = evaluate_rec(reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg, thresh_type_list=['ratio'], adjustment_mode_list=[True])
    
    return point_wise_results


            