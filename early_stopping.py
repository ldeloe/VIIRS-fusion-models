#import wandb
#import torch
import math

#from functions import save_best_model

class EarlyStopping:
#    def __init__(self, patience=10, min_delta=0):
    def __init__(self, patience=10, grace_period=20, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait for improvement before stopping.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.grace_period = grace_period
        self.epochs_since_start = 0

    def __call__(self, val_loss):
        self.epochs_since_start += 1
        if self.epochs_since_start <= self.grace_period:
            print(f"Within grace period ({self.epochs_since_start}/{self.grace_period}). Continuing training.")
            return False   
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            print(f"Validation loss improved to {self.best_loss:.4f}")

        else:
            self.counter += 1
            print(f"No improvement in validation loss for {self.counter} epoch(s).")
        
        if math.isnan(val_loss): 
            self.counter = self.patience + 1
            
        return self.counter >= self.patience

#    def __call__(self, val_loss, model):
#        if val_loss < self.best_loss - self.min_delta:
#            self.best_loss = val_loss
#            self.counter = 0
#        elif val_loss > 12.0:
#            self.counter += 1
#
#        if self.counter >= self.patience:
#            print("Early stopping triggered")
#            return True
#
#        return False
