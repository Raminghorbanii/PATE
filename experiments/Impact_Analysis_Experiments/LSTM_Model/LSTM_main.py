#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Utils import get_data, generate_loaders, set_seed, load_config
import pickle
import os
from solver import run_experiment

##############################

#Load Config
working_dir = os.getcwd()   
cfg = load_config(working_dir)


# Set the seed for all relevant libraries
set_seed(42)

#Prepare Dataset    
(x_train, _), (x_test, y_test_point) = get_data(cfg)
train_dataloader, test_dataloader = generate_loaders(x_train, x_test, y_test_point, cfg)


results = {}  # Dictionary to save the results for each seed
        
seeds = [0]  # List of seeds to use for the experiments

for seed in seeds:
    seed_results = run_experiment(cfg, train_dataloader, test_dataloader, seed = seed, save_mode = False)
    results[seed] = seed_results



# # Save the results
# with open(f'{cfg.dataset.save_prefix}/results_{cfg.dataset.name}_LSTM_Results.pkl', 'wb') as f:
#     pickle.dump(results, f)
    

