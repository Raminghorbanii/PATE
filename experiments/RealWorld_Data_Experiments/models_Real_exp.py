#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import os
from sklearn.preprocessing import MinMaxScaler

# Models
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import tensorflow as tf



#############################
#############################
# Predictive Models
#############################
#############################



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)  # Set the random seed for TensorFlow/Keras
    # print(f"Random seed set as {seed}")


def iforest_model(train_window, test_window, window_size, n_estimators=100,
                  max_samples="auto",contamination="auto",
                  max_features=1.,bootstrap=False,
                  n_jobs=None,random_state=42):
    
    modelName='IForest'
    iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination, n_jobs=n_jobs, random_state=random_state)
    iforest.fit(train_window)
    anomaly_scores = -iforest.decision_function(test_window)
    # anomaly_scores = window_to_points_Scores(test_data, anomaly_scores, window_size)
    # anomaly_scores = np.array([anomaly_scores[0]]*math.ceil((window_size-1)/2) + list(anomaly_scores) + [anomaly_scores[-1]]*((window_size-1)//2))
    anomaly_scores = np.pad(anomaly_scores, (window_size - 1, 0), 'constant', constant_values=(anomaly_scores[0],))
    anomaly_scores_plot = MinMaxScaler(feature_range=(0,1)).fit_transform(anomaly_scores.reshape(-1,1)).ravel()
    # plotFig(test_data, test_label, anomaly_scores_plot, window_size, fileName='Data', modelName=modelName) 
    
    return anomaly_scores, anomaly_scores_plot



def mvn_model(train_window, test_window, window_size, n_components, random_state):
    
    modelName='MVN'
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    gmm.fit(train_window)

    logpdf_mvn_test = gmm.score_samples(test_window)
    anomaly_scores = -logpdf_mvn_test
    
    # anomaly_scores = window_to_points_Scores(test_data, anomaly_scores, window_size)
    # anomaly_scores = np.array([anomaly_scores[0]]*math.ceil((window_size-1)/2) + list(anomaly_scores) + [anomaly_scores[-1]]*((window_size-1)//2))
    anomaly_scores = np.pad(anomaly_scores, (window_size - 1, 0), 'constant', constant_values=(anomaly_scores[0],))
    anomaly_scores_plot = MinMaxScaler(feature_range=(0,1)).fit_transform(anomaly_scores.reshape(-1,1)).ravel()
    # plotFig(test_data, test_label, anomaly_scores_plot, window_size, fileName='Data', modelName=modelName) 
    
    return anomaly_scores, anomaly_scores_plot




def LOF_model(x_train, x_test, window_size, n_neighbors=20, contamination="auto", novelty=True, n_jobs=None):
    
    modelName='LOF'
    set_seed()
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=novelty, n_jobs=n_jobs)
    clf.fit(x_train)

    # y_pred = clf.predict(x_test)
    anomaly_scores = -clf.decision_function(x_test)

    # anomaly_scores = window_to_points_Scores(test_data, anomaly_scores, window_size)
    # anomaly_scores = np.array([anomaly_scores[0]]*math.ceil((window_size-1)/2) + list(anomaly_scores) + [anomaly_scores[-1]]*((window_size-1)//2))
    anomaly_scores = np.pad(anomaly_scores, (window_size - 1, 0), 'constant', constant_values=(anomaly_scores[0],))

    anomaly_scores_plot = MinMaxScaler(feature_range=(0,1)).fit_transform(anomaly_scores.reshape(-1,1)).ravel()
        
    return anomaly_scores, anomaly_scores_plot
    

def pca_model(x_train, x_test, window_size, n_components= 0.95):
    
    modelName='PCA'
    pca = PCA(n_components=n_components).fit(x_train)

    X_pca = pca.transform(x_test)
    X_recon = pca.inverse_transform(X_pca)

    anomaly_scores = np.mean((x_test - X_recon)**2, axis=1)

    # anomaly_scores = window_to_points_Scores(test_data, anomaly_scores, window_size)
    # anomaly_scores = np.array([anomaly_scores[0]]*math.ceil((window_size-1)/2) + list(anomaly_scores) + [anomaly_scores[-1]]*((window_size-1)//2))
    anomaly_scores = np.pad(anomaly_scores, (window_size - 1, 0), 'constant', constant_values=(anomaly_scores[0],))
    anomaly_scores_plot = MinMaxScaler(feature_range=(0,1)).fit_transform(anomaly_scores.reshape(-1,1)).ravel()
        
    return anomaly_scores, anomaly_scores_plot    
    

def ocsvm_model(x_train, x_test, window_size):
    
    modelName='OC-SVM'
    clf = OneClassSVM(gamma="auto", kernel = 'rbf', nu = 0.05).fit(x_train)
    
    anomaly_scores = -clf.decision_function(x_test)

    # anomaly_scores = window_to_points_Scores(test_data, anomaly_scores, window_size)
    # anomaly_scores = np.array([anomaly_scores[0]]*math.ceil((window_size-1)/2) + list(anomaly_scores) + [anomaly_scores[-1]]*((window_size-1)//2))
    anomaly_scores = np.pad(anomaly_scores, (window_size - 1, 0), 'constant', constant_values=(anomaly_scores[0],))
    anomaly_scores_plot = MinMaxScaler(feature_range=(0,1)).fit_transform(anomaly_scores.reshape(-1,1)).ravel()
        
    return anomaly_scores, anomaly_scores_plot        
    

    
# The AE architecture is inspired by the AE model used in the 'VUS'[1] study. 
# [1]: Paparrizos, J., Boniol, P., Palpanas, T., Tsay, R. S., Elmore, A., & Franklin, M. J. (2022). Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. Proceedings of the VLDB Endowment, 15(11), 2774-2787.

class AE_MLP:  
    def __init__(self, slidingWindow = 100,  contamination = 0.1, epochs = 10, batch_size = 128, verbose=0, seed_num=20):
        self.slidingWindow = slidingWindow
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed_num = seed_num
        self.model_name = 'AE_MLP'
        
    def fit(self, X_clean, X_dirty, ratio = 0.15):

        TIME_STEPS =  self.slidingWindow
        epochs = self.epochs
        

        X_train = self.create_dataset(X_clean,TIME_STEPS)
        X_test = self.create_dataset(X_dirty,TIME_STEPS)
        
        set_seed(self.seed_num)
        X_train = MinMaxScaler().fit_transform(X_train.T).T
        X_test = MinMaxScaler().fit_transform(X_test.T).T
        
        set_seed(self.seed_num)
        model = Sequential()
        model.add(layers.Dense(32,  activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(TIME_STEPS, activation='relu'))
 
        model.compile(optimizer='adam', loss='mse')     
        
        set_seed(self.seed_num)
        history = model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size= self.batch_size,
                        shuffle=False,
                        validation_split=0.15,verbose=self.verbose,
                        callbacks=[EarlyStopping(monitor="val_loss", verbose=self.verbose, patience=5, mode="min")])

        test_predict = model.predict(X_test)
        test_mae_loss = np.mean(np.abs(test_predict - X_test), axis=1)
        nor_test_mae_loss = MinMaxScaler().fit_transform(test_mae_loss.reshape(-1,1)).ravel()
        score = np.pad(nor_test_mae_loss, (self.slidingWindow - 1, 0), 'constant', constant_values=(nor_test_mae_loss[0],))
        
        self.decision_scores_ = score
        
        return self
        
    
    # Generated training sequences for use in the model.
    def create_dataset(self, X, time_steps):
        output = []
        for i in range(len(X) - time_steps + 1):
            output.append(X[i : (i + time_steps)])
        return np.stack(output)
    
    
    
    
def AE_model(train_data, test_data, window_size,  epochs=100, batch_size = 128):
    
    modelName='AE'
    clf = AE_MLP(slidingWindow = window_size, epochs=epochs, batch_size = batch_size, verbose=1)
    clf.fit(train_data.reshape(train_data.shape[0],), test_data.reshape(test_data.shape[0],))
        
    anomaly_scores = clf.decision_scores_
    anomaly_scores_plot = MinMaxScaler(feature_range=(0,1)).fit_transform(anomaly_scores.reshape(-1,1)).ravel()

    return anomaly_scores, anomaly_scores_plot   


#############################
#############################
#############################
