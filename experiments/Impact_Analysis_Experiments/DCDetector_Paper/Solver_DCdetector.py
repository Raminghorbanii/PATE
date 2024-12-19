#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The code is based on the official published code from the original paper.

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support, f1_score, accuracy_score
from Utils import  set_seed



# from utils.utils import *
from model.DCdetector import DCdetector
# from einops import rearrange
# from metrics.metrics import *
import warnings
warnings.filterwarnings('ignore')



def my_kl_loss(p, q): #Same
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_): #Same
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping: #Same
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

    
    
    
    
    
def solver(cfg, train_loader, test_loader, seed, temperature_value):
    
    set_seed(seed)
    
    dataset_name = cfg.dataset.name
    lr = cfg.model.lr
    num_epochs = cfg.model.num_epochs
    anormly_ratio = cfg.dataset.anormly_ratio
    window_length = cfg.dataset.window_size
    input_size = cfg.dataset.x_dim

    patch_size = cfg.dataset.patch_size
    

    
    print("======================Model Definition======================")        
    
    
    model = DCdetector(win_size=window_length, enc_in=input_size, c_out=input_size, n_heads=1, d_model=256, e_layers=3, patch_size=patch_size, channel=input_size)
        
    model = model.to(cfg.device)
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    set_seed(seed)
    
    def vali(vali_loader):
        model.eval()
    
        loss_1 = []
        loss_2 = []
        for batch, inputs in enumerate(test_loader):
            inputs = inputs[0].to(cfg.device)
            series, prior = model(inputs)
            series_loss = 0.0
            prior_loss = 0.0
            
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, window_length)).detach())) + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, window_length)).detach(),series[u])))
                
                prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)),series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(),(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)))))
                
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
    
            loss_1.append((prior_loss - series_loss).item())
    
        return np.average(loss_1), np.average(loss_2)
        
    
    
                                                                                                                                                                                                                         
    print("======================TRAIN MODE======================")

    time_now = time.time()  
    
    path = 'checkpoints'
    if not os.path.exists(path):
        os.makedirs(path)
    
    early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name='selected_dataset')
    train_steps = len(train_loader)
    
    
    set_seed(seed)
    
    for epoch in range(num_epochs):
            iter_count = 0

            epoch_time = time.time()
            model.train()
            
            for batch, inputs in enumerate(train_loader):

                optimizer.zero_grad()
                iter_count += 1
                inputs = inputs[0].to(cfg.device)

                series, prior = model(inputs)


                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)).detach())) + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)).detach(),series[u])))
                    
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)),series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)))))
                    
                    
                    
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss 


                if (batch + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((num_epochs - epoch) * train_steps - batch)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer.step()


            vali_loss1, vali_loss2 = vali(test_loader)
            torch.cuda.empty_cache()
            

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            early_stopping(vali_loss1, vali_loss2, model, path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, lr)    



    print("======================TEST MODE======================")    

    set_seed(seed)
    model.load_state_dict(
    torch.load(
        os.path.join(str(path), str('selected_dataset') + '_checkpoint.pth')))
    model.eval()   
    temperature = temperature_value


    # (1) stastic on the train set
    attens_energy = []
    for batch, inputs in enumerate(train_loader):
        inputs = inputs[0].to(cfg.device)
        series, prior = model(inputs)
        series_loss = 0.0
        prior_loss = 0.0
        
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)).detach()) * temperature
                prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)).detach()) * temperature
                prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)),series[u].detach()) * temperature
                      
                
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric.detach().cpu().numpy()
        attens_energy.append(cri)
    
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)
    

    model.eval() # to be sure 
    # (2) find the threshold
    attens_energy = []
    for batch, inputs in enumerate(test_loader):
        inputs = inputs[0].to(cfg.device)
        series, prior = model(inputs)
        
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)),
                    series[u].detach()) * temperature

        # Metric
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric.detach().cpu().numpy()
        attens_energy.append(cri)
    
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    
    thresh = np.percentile(combined_energy, 100 - anormly_ratio)
    print("Threshold :", thresh)
    
    
    
    # (3) evaluation on the test set
    test_labels = []
    attens_energy = []
    model.eval() # to be sure 
    
    for batch, inputs in enumerate(test_loader):
        labels = inputs[1]
        inputs = inputs[0].to(cfg.device)

        series, prior = model(inputs)
    
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,window_length)),
                    series[u].detach()) * temperature
                
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric.detach().cpu().numpy()
        attens_energy.append(cri)
        test_labels.append(labels)
    
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    test_labels = np.array(test_labels)
    
    pred = (test_energy > thresh).astype(int)
    gt = test_labels.astype(int)
    
    np.save(f'AnomalyScores_DCdetector_seed{seed}_dataset_{dataset_name}.npy', test_energy)
    np.save(f'labels_DCdetector_seed{seed}_dataset_{dataset_name}.npy', gt)
    
    # detection adjustment
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
    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)
    

    def compute_auc(y_true, scores):
        
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_score = auc(fpr, tpr)
        print('AUC: {:0.4f}'.format( auc_score))
        
        return auc_score 
    
    
    # Compute the AUPRC for the model
    def compute_auprc(y_true, scores):
        
        _, recall, thresholds = precision_recall_curve(y_true, scores)
        auc_pr_score = average_precision_score(y_true, scores)
        print('AUC-PR: {:0.4f}'.format( auc_pr_score))
        return auc_pr_score
    
    
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    auc_result = compute_auc(gt, pred)
    auc_pr = compute_auprc(gt, pred)
    
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} , AUC : {:0.4f}, AUC-PR : {:0.4f}".format(
            accuracy, precision,
            recall, f_score, auc_result, auc_pr))
    
    return precision, recall, f_score, auc_result, auc_pr, thresh
