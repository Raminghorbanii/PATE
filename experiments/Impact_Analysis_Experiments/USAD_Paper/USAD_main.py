#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The code is based on the official published code from the original paper.

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support, f1_score, accuracy_score


####### Utilities Files folder ############

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def plot_history(history):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()
    
def histogram(y_test,y_pred):
    plt.figure(figsize=(12,6))
    plt.hist([y_pred[y_test==0],
              y_pred[y_test==1]],
            bins=20,
            color = ['#82E0AA','#EC7063'],stacked=True)
    plt.title("Results",size=20)
    plt.grid()
    plt.show()
    
def ROC(y_test,y_pred):
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]
    
    

##########################################
############# Model ######################
##########################################
    
    
class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
    

class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))



##########################################
##########################################
##########################################


def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(torch.flatten(batch[0], start_dim=1), device), n) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for batch in train_loader:
            batch = torch.flatten(batch[0], start_dim=1) # I added this
            batch = to_device(batch,device)
            
            #Train AE1
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
            
        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing(model, test_loader, alpha=.5, beta=.5):
    results=[]
    with torch.no_grad():
        for batch in test_loader:
            batch = torch.flatten(batch[0], start_dim=1) # I added this
            batch=to_device(batch,device)
            w1=model.decoder1(model.encoder(batch))
            w2=model.decoder2(model.encoder(w1))
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results



def testing_pointwise(model, test_loader, alpha=.5, beta=.5):
    all_errors = []

    with torch.no_grad():
        for batch in test_loader:
            original_shape = batch[0].shape
            batch_flat = torch.flatten(batch[0], start_dim=1).to(device)
            
            w1 = model.decoder1(model.encoder(batch_flat))
            w2 = model.decoder2(model.encoder(w1))
            
            # Reshape reconstructions to original shape
            w1 = w1.view(original_shape)
            w2 = w2.view(original_shape)

            # Move batch[0] to the same device as w1 and w2
            batch_on_device = batch[0].to(device)  # Move batch[0] to the GPU
            
            # Compute point-wise errors
            errors = alpha * (batch_on_device - w1)**2 + beta * (batch_on_device - w2)**2
            
            # Take the mean over the features
            mean_errors = torch.mean(errors, dim=2)
            
            all_errors.extend(mean_errors.view(-1).tolist())

    return all_errors



def getting_labels(data_loader):
    all_labels = []

    for batch in data_loader:
        labels = batch[1].numpy()
        all_labels.extend(labels.flatten())

    # Convert list to numpy array
    all_labels = np.array(all_labels)

    return all_labels


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


        
from sklearn.metrics._ranking import _binary_clf_curve
def calculate_best_f1_threshold(y_true, scores):

    # Get precision, recall, and threshold values
    fps_orig, tps_orig, thresholds = _binary_clf_curve(y_true, scores, pos_label=1)
    drop_intermediate = True
    
    if drop_intermediate and len(tps_orig) > 2:
        # Drop thresholds corresponding to points where true positives (tps)
        # do not change from the previous or subsequent point. This will keep
        # only the first and last point for each tps value. All points
        # with the same tps value have the same recall and thus x coordinate.
        # They appear as a vertical line on the plot.
        optimal_idxs = np.where(
            np.concatenate(
                [[True], np.logical_or(np.diff(tps_orig[:-1]), np.diff(tps_orig[1:])), [True]]
            )
        )[0]
        fps_orig = fps_orig[optimal_idxs]
        tps_orig = tps_orig[optimal_idxs]
        thresholds = thresholds[optimal_idxs]
        
    # Calculate F1 scores for each threshold
    F_scores_list = []
    print(f"Length threshold: {len(thresholds)}")
    for i in thresholds:
        
        y_pred = (scores > i).astype(int)
        y_pred, y_true = detection_adjustment(y_pred, y_true)
        
        _, _, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        F_scores_list.append(f_score)
    
    best_threshold = thresholds[np.argmax(np.array(F_scores_list))]
    print("Selected Threshold :", best_threshold)
    
    y_pred = (scores > best_threshold).astype(int)
    y_pred, y_true = detection_adjustment(y_pred, y_true)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division = 1)

    print(
        "Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(precision, recall, f_score))    
        
    return precision, recall, f_score, best_threshold

##########################################
##########################################
##########################################

from Utils import get_data, generate_loaders, set_seed, load_config
import os
import pickle


device = get_default_device()    
seed = 0

#Load Config
working_dir = os.getcwd()   
cfg = load_config(working_dir)

# Set the seed for all relevant libraries
set_seed(42)

#Prepare Dataset    
(x_train, _), (x_test, y_test_point) = get_data(cfg)
train_loader, test_loader = generate_loaders(x_train, x_test, y_test_point, cfg)

data_name = cfg.dataset.name
data_channels = cfg.dataset.x_dim
window_size = cfg.dataset.window_size
N_EPOCHS = cfg.dataset.Epochs
latent_size = cfg.dataset.latent_size
anormly_ratio = cfg.dataset.anormly_ratio

w_size= window_size * data_channels
z_size= window_size * latent_size

model = UsadModel(w_size, z_size)
print(model)
model = to_device(model,device)

history = training(N_EPOCHS,model,train_loader,test_loader)

results_point_wise=testing_pointwise(model,test_loader)
test_rec = np.array(results_point_wise)
labels = getting_labels(test_loader)

results_point_wise_train = testing_pointwise(model, train_loader)
train_rec = np.array(results_point_wise_train)


def find_threshold(rec_error_train, rec_error_test, anormly_ratio):
    combined_rec = np.concatenate([rec_error_train, rec_error_test], axis=0)
    
    thresh = np.percentile(combined_rec, 100 - anormly_ratio)
    print("Selected Threshold :{:0.10f}".format( thresh))
    
    return thresh

thresh = find_threshold(train_rec, test_rec, 1)
pred = (test_rec > thresh).astype(int)
gt = labels.astype(int)
print("pred:   ", pred.shape)
print("gt:     ", gt.shape)
pred, gt = detection_adjustment(pred, gt)
precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
results_ratio_thre = precision, recall, f_score, thresh

print("Precision**: {:0.4f}, Recall** : {:0.4f}, F-score** : {:0.4f} ".format(precision, recall, f_score))    
    
# np.save(f'AnomalyScores_USAD_seed{seed}_dataset_{data_name}.npy', test_rec)
# np.save(f'labels_USAD_seed{seed}_dataset_{data_name}.npy', labels)   
    
# results_best_f1 = calculate_best_f1_threshold(labels, test_rec)


