#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau



class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, cfg, mean_data = None, use_rbf=False, optimizer=None, criterion=None, patience=100, save_path='best_model_early.pth'):
        # Initialize the trainer with the model, data, configuration, and training options
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.cfg = cfg
        
        self.use_rbf = use_rbf  # Flag to indicate whether to use the RBF layer

        # Set up the optimizer and loss function
        self.optimizer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=self.cfg.model.lr, weight_decay=self.cfg.model.weight_decay)
        # Define the scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.criterion = criterion if criterion else nn.MSELoss()
        
        # Set up early stopping
        self.best_loss = float('inf')
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best_model = None
        
        self.mean_data = mean_data.to(self.cfg.device) if mean_data is not None else None
        
    def train(self):
        # Main training loop
        for epoch in range(self.cfg.model.num_epochs):
            self.model.train()
            running_loss = 0.0
            
            for batch in self.train_dataloader:
                inputs = batch[0].to(self.cfg.device)
                self.optimizer.zero_grad()
                all_outputs = self.model(inputs)
                
                # Separate the outputs based on whether the RBF layer and/or VAE are used

                outputs, _, _ = all_outputs
                                               
                # Compute the main loss
                loss = self.criterion(outputs, inputs)


                # Backpropagation and optimization
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.model.clip_grad) # Can be removed!

                self.optimizer.step()
                running_loss += loss.item()

            # Compute the average loss for this epoch
            epoch_loss = running_loss / len(self.train_dataloader)

            # Evaluate the model on the test set
            test_loss = self.test_model()
            
            # Step the scheduler
            self.scheduler.step(test_loss)


            # Print the loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.cfg.model.num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Check for early stopping
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), self.save_path)

            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    # Load the best model before returning
                    self.model.load_state_dict(torch.load(self.save_path))
                    break
        
        # Load the best model after training
        self.model.load_state_dict(torch.load(self.save_path))

        # Delete the model file
        os.remove(self.save_path)
        
        return self.model

        

    def test_model(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs in self.test_dataloader:
                inputs = inputs[0].to(self.cfg.device)
                outputs, _, _ = self.model(inputs)
                    
                loss = self.criterion(outputs, inputs)
                running_loss += loss.item()

        return running_loss / len(self.test_dataloader)



