
# The code is based on the official published code from the original paper.


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from patchad_model.models import PatchMLPAD
from einops import rearrange,repeat
import warnings
warnings.filterwarnings('ignore')
from tkinter import _flatten
from Utils_personal import  set_seed
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support, f1_score, accuracy_score


def my_kl_loss(p, q):
    # B N D
    res = p * (torch.log(p + 0.0000001) - torch.log(q + 0.0000001))
    # B N
    return torch.sum(res, dim=-1)

def inter_intra_dist(p,q,w_de=True,train=1,temp=1):
    # B N D
    if train:
        if w_de:
            p_loss = torch.mean(my_kl_loss(p,q.detach()*temp)) + torch.mean(my_kl_loss(q.detach(),p*temp))
            q_loss = torch.mean(my_kl_loss(p.detach(),q*temp)) + torch.mean(my_kl_loss(q,p.detach()*temp))
        else:
            p_loss = -torch.mean(my_kl_loss(p,q.detach())) 
            q_loss = -torch.mean(my_kl_loss(q,p.detach())) 
    else:
        if w_de:
            p_loss = my_kl_loss(p,q.detach()) + my_kl_loss(q.detach(),p)
            q_loss = my_kl_loss(p.detach(),q) + my_kl_loss(q,p.detach())

        else:
            p_loss = -(my_kl_loss(p,q.detach())) 
            q_loss = -(my_kl_loss(q,p.detach())) 

    return p_loss,q_loss


def normalize_tensor(tensor):
    # tensor: B N D
    sum_tensor = torch.sum(tensor,dim=-1,keepdim=True)
    normalized_tensor = tensor / sum_tensor
    return normalized_tensor

def anomaly_score(patch_num_dist_list,patch_size_dist_list, win_size, train=1, temp=1, w_de=True):
    for i in range(len(patch_num_dist_list)):
        patch_num_dist = patch_num_dist_list[i]
        patch_size_dist = patch_size_dist_list[i]


        patch_num_dist = repeat(patch_num_dist,'b n d -> b (n rp) d',rp=win_size//patch_num_dist.shape[1])
        patch_size_dist = repeat(patch_size_dist,'b p d -> b (rp p) d',rp=win_size//patch_size_dist.shape[1])

        patch_num_dist = normalize_tensor(patch_num_dist)
        patch_size_dist = normalize_tensor(patch_size_dist)

        patch_num_loss,patch_size_loss = inter_intra_dist(patch_num_dist,patch_size_dist,w_de,train=train,temp=temp)

        if i==0:
            patch_num_loss_all = patch_num_loss
            patch_size_loss_all = patch_size_loss
        else:
            patch_num_loss_all += patch_num_loss
            patch_size_loss_all += patch_size_loss

    return patch_num_loss_all,patch_size_loss_all



def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
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
        score = val_loss
        score2 = val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        print('Save model')
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

        

    

 

    
def solver(cfg, train_loader, test_loader, seed, temperature_value):
    
    set_seed(seed)
    
    dataset_name = cfg.dataset.name
    lr = cfg.model.lr
    num_epochs = cfg.model.num_epochs
    anormly_ratio = cfg.dataset.anormly_ratio
    input_size = cfg.dataset.x_dim

    patch_size = cfg.dataset.patch_size
    patch_mx = cfg.dataset.patch_mx
    win_size = cfg.dataset.window_size

    
    print("======================Model Definition======================")        
    
    
    model = PatchMLPAD(win_size=win_size, e_layer= 3 , patch_sizes=patch_size, dropout=0.0, activation="gelu", output_attention=True,
                                channel=input_size, d_model= 40, cont_model=win_size, norm='ln')
        
    model = model.to(cfg.device)
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    set_seed(seed)
    
    @torch.no_grad()
    def vali(test_loader):
        model.eval()
        loss_1 = []
        loss_2 = []
        for batch, inputs in enumerate(test_loader):
            inputs = inputs[0].to(cfg.device)
            patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list = model(inputs)

            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=1)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            # loss3 = patch_size_loss + patch_num_loss
                
            p_loss = patch_size_loss 
            q_loss = patch_num_loss

            loss_1.append((p_loss).item())
            loss_2.append((q_loss).item())

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
            
            patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list = model(inputs)


            loss = 0.

            cont_loss1,cont_loss2 = anomaly_score(patch_num_dist_list,patch_size_mx_list,win_size=win_size,train=1,temp=1)
            cont_loss_1 = cont_loss1 - cont_loss2
            loss += patch_mx *cont_loss_1

            cont_loss12,cont_loss22 = anomaly_score(patch_num_mx_list,patch_size_dist_list,win_size=win_size,train=1,temp=1)
            cont_loss_2 = cont_loss12 - cont_loss22
            loss += patch_mx *cont_loss_2

            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=1,temp=1)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            loss3 = patch_num_loss - patch_size_loss

            
            loss += loss3 * (1-patch_mx)

            if (batch + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((num_epochs - epoch) * train_steps - batch)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

                epo_left = speed * (len(train_loader))
                print('Epoch time left: {:.4f}s'.format(epo_left))
 
            loss.backward()
            optimizer.step()


        vali_loss1, vali_loss2 = vali(test_loader)
        print('Vali',vali_loss1, vali_loss2)

        print(
            "Epoch: {0}, Cost time: {1:.3f}s ".format(
                epoch + 1, time.time() - epoch_time))
        early_stopping(vali_loss1, vali_loss2, model, path)
        if early_stopping.early_stop:
            break
        adjust_learning_rate(optimizer, epoch + 1, lr)


    print("======================TEST MODE======================")    


    set_seed(seed)
    model.load_state_dict(
    torch.load(
        os.path.join(str(path), str('selected_dataset') + '_checkpoint.pth')))
    model.eval()   
    temperature = temperature_value

    use_project_score = 0
    
    # (1) stastic on the train set
    attens_energy = []
    for batch, inputs in enumerate(train_loader):
        inputs = inputs[0].to(cfg.device)
        patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list = model(inputs)

        if use_project_score:
            patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list,patch_size_mx_list,win_size=win_size,train=0,temp=temperature)
        else:
            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=0)
        patch_num_loss = patch_num_loss / len(patch_num_dist_list)
        patch_size_loss = patch_size_loss / len(patch_num_dist_list)

        loss3 = patch_size_loss - patch_num_loss

        
        # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        metric = torch.softmax((-patch_num_loss), dim=-1)
        cri = metric.detach().cpu().numpy()
        attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)



    model.eval() # to be sure 
    # (2) find the threshold
    attens_energy = []
    for batch, inputs in enumerate(test_loader):
        inputs = inputs[0].to(cfg.device)
        patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list = model(inputs)

        if use_project_score:
            patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list,patch_size_mx_list,win_size=win_size,train=0,temp=temperature)
        else:
            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=0)
        patch_num_loss = patch_num_loss / len(patch_num_dist_list)
        patch_size_loss = patch_size_loss / len(patch_num_dist_list)

        loss3 = patch_size_loss - patch_num_loss

        # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        metric = torch.softmax((-patch_num_loss ), dim=-1)
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

    test_data = []
    for batch, inputs in enumerate(test_loader):
        labels = inputs[1]
        inputs = inputs[0].to(cfg.device)
        patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list = model(inputs)

        if use_project_score:
            patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list,patch_size_mx_list,win_size=win_size,train=0,temp=temperature)
        else:
            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=0)
        patch_num_loss = patch_num_loss / len(patch_num_dist_list)
        patch_size_loss = patch_size_loss / len(patch_num_dist_list)

        loss3 = patch_size_loss - patch_num_loss

        # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        metric = torch.softmax((-patch_num_loss ), dim=-1)
        cri = metric.detach().cpu().numpy()

        attens_energy.append(cri)
        test_labels.append(labels)
        test_data.append(inputs.cpu().numpy().reshape(-1,inputs.shape[-1]))
        
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    test_labels = np.array(test_labels)
    test_data = np.concatenate(test_data,axis=0)


    pred = (test_energy > thresh).astype(int)
    gt = test_labels.astype(int)
    
    
    np.save(f'AnomalyScores_PatchAD_seed{seed}_dataset_{dataset_name}.npy', test_energy)
    np.save(f'labels_PatchAD_seed{seed}_dataset_{dataset_name}.npy', gt)
    
    
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

