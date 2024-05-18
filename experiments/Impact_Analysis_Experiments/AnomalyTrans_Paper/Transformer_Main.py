#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The code is based on the official published code from the original paper.


from Utils import get_data, generate_loaders, set_seed, load_config
from Solver_AnomalyTransformer import solver

import os
import pickle


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1000'  # Adjust the value as needed


#Load Config
working_dir = os.getcwd()   
cfg = load_config(working_dir)


# Set the seed for all relevant libraries
set_seed(42)

#Prepare Dataset    
(x_train, _), (x_test, y_test_point) = get_data(cfg)
train_loader, test_loader = generate_loaders(x_train, x_test, y_test_point, cfg)


results = {}  # Dictionary to save the results for each seed
        
seeds = [0]  # List of seeds to use for the experiments

for seed in seeds:
    seed_results = solver(cfg, train_loader, test_loader, seed, temperature_value = 50)
    results[seed] = seed_results


# # Save the results
# with open(f'{cfg.dataset.save_prefix}/results_{cfg.dataset.name}_AnomalyTransformer_PaperResults.pkl', 'wb') as f:
#     pickle.dump(results, f)
    

    
    
    
