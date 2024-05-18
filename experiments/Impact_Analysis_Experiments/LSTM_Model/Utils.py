#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle

import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support, f1_score, accuracy_score
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import random

from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf


import hashlib
import torch




def hash_model_parameters(model):
    """
    Compute a hash of the model's parameters.

    Parameters:
    - model (torch.nn.Module): The PyTorch model.

    Returns:
    - str: The hash value of the model's parameters.
    """
    # Convert model parameters to a byte string
    model_params = []
    for param in model.parameters():
        model_params.append(param.data.cpu().numpy().tobytes())
    model_byte_string = b''.join(model_params)
    
    # Compute the hash of the byte string
    return hashlib.sha256(model_byte_string).hexdigest()




def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



def load_config(working_dir, config_file_name="base_config"):
    with initialize_config_dir(version_base=None, config_dir=f"{working_dir}/configs"):
        cfg = compose(config_name = config_file_name)
        if cfg.device == "auto":
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print('Config File Loaded Dataset:', cfg.dataset.name)
        print('Config File Loaded Model:',cfg.model.type)
        print('Using Device:',cfg.device)

        # If using CUDA, print the GPU model
        if cfg.device == "cuda" and torch.cuda.is_available():
            print('GPU Model:', torch.cuda.get_device_name(0))
            
        return cfg
    
    
    
def get_data(cfg, do_normalization=True):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    
    prefix = cfg.dataset.data_prefix
    dataset = cfg.dataset.name
    max_train_size = cfg.dataset.max_train_size
    max_test_size = cfg.dataset.max_test_size
    train_start = cfg.dataset.train_start
    test_start = cfg.dataset.test_start
    x_dim = cfg.dataset.x_dim
    
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
        
    print('load data of:', dataset)
    print("train Start and End: ", train_start, train_end)
    print("test Start and End: ", test_start, test_end)
    
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_normalization:
        scaler = StandardScaler() # fit on training data
        train_data = scaler.fit_transform(train_data)  # transform training data
        test_data = scaler.transform(test_data)  # transform test data
        
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    
    return (train_data, None), (test_data, test_label)



# Function to create windows from the data
def create_windows(data, window_size, step=1):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)


        
def generate_loaders(train_data, test_data, test_labels, cfg, seed=42):
    """
    Generate DataLoader objects for training and testing datasets.
    
    Parameters:
    - train_data: Training dataset.
    - test_data: Testing dataset.
    - test_labels: Labels for the test dataset.
    - cfg: Configuration object containing batch size, window size, and other parameters.
    - seed: Random seed for reproducibility.
    
    Returns:
    - train_dataloader: DataLoader object for the training dataset.
    - test_dataloader: DataLoader object for the test dataset.
    """
    
    # Extract relevant parameters from configuration object
    batch_size = cfg.model.batch_size
    window_size = cfg.dataset.window_size
    step = cfg.dataset.step
    anomaly_proportion_window = cfg.dataset.anomaly_proportion_window
    
    # Segment the data into overlapping windows
    train_data = create_windows(train_data, window_size, step)
    train_data = shuffle(train_data, random_state=seed)

    # Create dummy labels for training data to match its shape
    dummy_train_labels_point = np.zeros_like(train_data, dtype=int)
    dummy_train_labels_window = np.zeros((train_data.shape[0],), dtype=int)


    test_data = create_windows(test_data, window_size, step)
    test_labels_point = create_windows(test_labels, window_size, step)
    
    # Label windows as anomalous if the proportion of anomalous points within them exceeds a threshold
    test_labels_window = (np.mean(test_labels_point, axis=1) > anomaly_proportion_window).astype(int)
        
    
    # Convert data and labels into PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels_point = torch.tensor(test_labels_point, dtype=torch.long)
    test_labels_window = torch.tensor(test_labels_window, dtype=torch.long)

    # Print the shapes of the data tensors (useful for debugging and understanding data dimensions)
    print("train window shape: ", train_data.shape)
    print("test window shape: ", test_data.shape)
    print("test window label shape (point-level): ", test_labels_point.shape)
    print("test window label shape (window-level): ", test_labels_window.shape)

    # Prepare the training data using a TensorDataset (combining data and labels)
    train_data = TensorDataset(train_data, torch.tensor(dummy_train_labels_point, dtype=torch.long), torch.tensor(dummy_train_labels_window, dtype=torch.long))
    # Create DataLoader objects for both training and testing data
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    
    # For testing data, shuffling isn't needed, so we just specify the batch size
    test_dataset = TensorDataset(test_data, test_labels_point, test_labels_window)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader




def get_encoder_output(model, dataloader, device):
    model.eval()
    encoder_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            _, enc_out, _ = model(inputs)
            
            encoder_outputs.append(enc_out.cpu().numpy())
    return np.concatenate(encoder_outputs)


def function_data_mean(encoder_output):
    
    encoder_output = torch.from_numpy(encoder_output)
    encoder_output_flat = encoder_output.reshape(-1, encoder_output.size(-1))    
    data_mean = torch.mean(encoder_output_flat, dim=0)

    return data_mean



# Calculate the reconstruction error for each window
def calculate_reconstruction_errors(model, dataloader, cfg, test_mode=False):

    model.eval()
    reconstruction_errors_point = []
    true_labels_point = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(cfg.device)
            outputs, _, _ = model(inputs)
                
            batch_error_point = torch.mean((inputs - outputs) ** 2, dim=2).cpu().numpy() # error per point
            
            reconstruction_errors_point.extend(batch_error_point.flatten())

            if test_mode:
                labels_point = batch[1].numpy()
                true_labels_point.extend(labels_point.flatten())

    if test_mode:
        return np.array(reconstruction_errors_point), np.array(true_labels_point)
    else:
        return np.array(reconstruction_errors_point)





def evaluate_rec(reconstruction_errors_tr, reconstruction_errors_te, true_labels, cfg, thresh_type_list=['ratio', 'f1-score'], adjustment_mode_list=[True, False], Analysis_Mode = False, Save_adj = False):
    print("##############################")
    results = {}
    
    for thresh_type in thresh_type_list:
        for adjustment_mode in adjustment_mode_list:
            
            print(f"## Calculating evaluate_rec for thresh_type: {thresh_type}, adjustment_mode: {adjustment_mode}")


            if thresh_type == 'ratio':
                #Find threshold based on ratio
                thresh = find_threshold(reconstruction_errors_tr, reconstruction_errors_te, cfg.dataset.anormly_ratio)
        
            elif thresh_type == 'f1-score':
                #Find threshold based on F1-Score
                thresh = calculate_best_f1_threshold(true_labels, reconstruction_errors_te, max_range = 100)


            np.save(f'AnomalyScores_LSTM_seed{cfg.seed}_dataset_{cfg.dataset.name}.npy', reconstruction_errors_te)
            np.save(f'labels_LSTM_seed{cfg.seed}_dataset_{cfg.dataset.name}.npy', true_labels)  
            
            adj_pred = (reconstruction_errors_te > thresh).astype(int)
            adj_pred, y_true = detection_adjustment(adj_pred, true_labels)
            
            #Save adj_pred and labels
            if Save_adj:
                print('Saving the Predictions and Labels for furthur analysis')            
                np.save(f'adj_pred_Rec_{cfg.dataset.name}_{cfg.model.type}_{cfg.initialization}_seed{cfg.seed}.npy', adj_pred)
                np.save(f'y_true_Rec_{cfg.dataset.name}_{cfg.model.type}_{cfg.initialization}_seed{cfg.seed}.npy', y_true)
             
            #Calculate Metrics
            accuracy, precision, recall, f1 = calculate_metrics(true_labels, reconstruction_errors_te, adj_thresh = thresh , adjustment = adjustment_mode)    
        
            # Calculate AUC
            auc = compute_auc(true_labels, reconstruction_errors_te,  adj_thresh = thresh , adjustment = adjustment_mode)
        
            # Calculate AUPRC
            auprc = compute_auprc(true_labels, reconstruction_errors_te,  adj_thresh = thresh , adjustment = adjustment_mode)
            
            results[(thresh_type, adjustment_mode)] = {'accuracy':accuracy, 'precision': precision, 'recall': recall, 'f1-score':f1, 'AUC': auc, 'AUC-PR': auprc}
    

    if Analysis_Mode:
        return thresh, reconstruction_errors_te
    
    else:
        return results




    
    
def find_threshold(rec_error_train, rec_error_test, anormly_ratio):
    combined_rec = np.concatenate([rec_error_train, rec_error_test], axis=0)

    thresh = np.percentile(combined_rec, 100 - anormly_ratio)
    print("Selected Threshold :{:0.10f}".format( thresh))

    return thresh
    


def detection_adjustment(pred, gt):
    
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    
    pred = np.array(pred)
    gt = np.array(gt)
    
    return pred, gt



def compute_auc(y_true, scores, adj_thresh, adjustment = False):
    
    if adjustment:
        scores = (scores > adj_thresh).astype(int)
        scores, y_true = detection_adjustment(scores, y_true)
            
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_score = auc(fpr, tpr)
    print('AUC: {:0.4f}'.format( auc_score))
    
    return auc_score 


# Compute the AUPRC for the model
def compute_auprc(y_true, scores, adj_thresh, adjustment = False):
    
    if adjustment:
        scores = (scores > adj_thresh).astype(int)
        scores, y_true = detection_adjustment(scores, y_true)
           
        
    _, recall, thresholds = precision_recall_curve(y_true, scores)
    auc_pr_score = average_precision_score(y_true, scores)
    print('AUC-PR: {:0.4f}'.format( auc_pr_score))
    return auc_pr_score



def calculate_best_f1_threshold(y_true, scores, max_range = 100):

    # Get precision, recall, and threshold values
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Get the threshold that gives the maximum F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    print("Selected Threshold :", best_threshold)
    print("Best F1Score :", np.argmax(f1_scores))
    
    return best_threshold



def calculate_metrics(y_true, scores, adj_thresh, adjustment = False):
    y_pred = (scores > adj_thresh).astype(int)
    
    if adjustment:
        y_pred, y_true = detection_adjustment(y_pred, y_true)
        
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

    return accuracy, precision, recall, f_score



def flatten_dict(d, prefix=''):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, prefix=f"{prefix}{k}_"))
        else:
            clean_key = prefix + str(k)
            clean_key = clean_key.replace(",", "").replace("(", "").replace(")", "").replace(" ", "_").replace("'", "")
            flat_dict[clean_key] = v
    return flat_dict



def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0].to(device)
            extracted_features = model.extract_features(inputs)
            features_list.append(extracted_features.cpu().numpy())

    return np.concatenate(features_list, axis=0)


