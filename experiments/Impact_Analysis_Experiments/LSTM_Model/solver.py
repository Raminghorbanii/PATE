#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from evaluation import Reconstruction_Errors_Clculation, evaluate_based_on_reconstruction
import torch


from Utils import set_seed, get_encoder_output
from LSTM_Model import LSTM
# from Transformer_Model import Transformer
from Training import Trainer
import torch


def train_selected_model(cfg, train_dataloader, test_dataloader, seed, save_mode = False):
    
    """
    Pre-trains the model without the Radial Basis Function (RBF) layer.
    
    This function handles the pre-training process of models without the RBF layer. 
    The pre-training is done using the model type specified in the configuration. 
    After training, the encoder's output is extracted, which can be used for 
    subsequent training phases. The function also provides an option to save 
    the encoder output and the trained model's state dictionary for future use.
    
    Args:
        cfg (object): Configuration object containing model and training settings.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        seed (int): Seed value for reproducibility.
        save_mode (bool, optional): If True, saves the encoder output and the trained model's 
                                    state dictionary. Defaults to True.
        
    Returns:
        trained_model (object): Model object after pre-training without the RBF layer.
    """

    print("####################")
    print(f"Starting pre-training without RBF using model type: {cfg.model.type}")
    
    base_model_name = cfg.model.base_model_type  # Name of the model used in the first phase

    # Set the seed for all relevant libraries
    set_seed(seed)
    print(f"Running experiment with seed {seed}")

    # Determine the model type based on the config
    if cfg.model.type == "LSTM":
        model = LSTM(cfg).to(cfg.device)
    # elif cfg.model.type == "Transformer":
        # model = Transformer(cfg).to(cfg.device)

    print(model)

    trainer = Trainer(model, train_dataloader, test_dataloader, cfg, use_rbf=False)
    trained_model = trainer.train()
    print("Training completed.")
    print("####################")
    
    encoder_output = get_encoder_output(trained_model, train_dataloader, cfg.device)
    print("Encoder output extracted.")
    
    if save_mode:
        # Save the encoder output and the model's state dictionary for the next phase
        torch.save(encoder_output, f'{cfg.dataset.save_prefix}/encoder_output_{cfg.dataset.name}_{base_model_name}_{seed}_{cfg.initialization}.pth')
        torch.save(trained_model.state_dict(), f'{cfg.dataset.save_prefix}/trained_model_{cfg.dataset.name}_{base_model_name}_{seed}_{cfg.initialization}.pth')
            
    return trained_model



def run_experiment(cfg, train_dataloader, test_dataloader, seed, save_mode):

    """
    Run the experiment based on the provided configuration, data loaders, and seed.
    
    Args:
        cfg: Configuration object containing necessary parameters.
        train_dataloader: DataLoader object containing the training dataset.
        test_dataloader: DataLoader object containing the test dataset.
        seed: Seed value for reproducibility.
        
    Returns:
        dict: Results of the experiment for the given seed.
    """
    
    results = {}

    print(f"Selected experiment with seed: {seed}")
                
    print("Starting pre-training phase...")
            
    trained_model = train_selected_model(cfg, train_dataloader, test_dataloader, seed, save_mode = save_mode)
    reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point = Reconstruction_Errors_Clculation(trained_model, train_dataloader, test_dataloader, cfg)
    Rec_point_wise_results = evaluate_based_on_reconstruction(reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg)
    results[seed] = {'Point Results': Rec_point_wise_results} # Save the results for this seed
            
    print("Pre-training phase completed.")

    print(f"Experiment with seed {seed} completed.")

    return results