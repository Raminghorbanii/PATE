#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The code is based on the official published code from the original paper.

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



def load_config(working_dir):
    with initialize_config_dir(version_base=None, config_dir=f"{working_dir}/configs"):
        cfg = compose(config_name="base_config")
        if cfg.device == "auto":
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print('Config File Loaded Dataset:', cfg.dataset.name)
        print('Config File Loaded Model:',cfg.model.type)
        print('Using Device:',cfg.device)
        
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


