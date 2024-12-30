import torch

class EarlyStopCriterion:
    """
    Stop training if the validation loss hasn't improved for a fixed number of
    epochs. 
    """
    def __init__(self, early_stop_epochs):
        self.early_stop_epochs = early_stop_epochs
        self.best_val_loss = torch.inf
        self.best_val_loss_epoch = 0
        self.current_epoch = 0
    
    def epoch_step(self, epoch, val_loss):
        self.current_epoch = epoch
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_loss_epoch = epoch

    def check_early_stop(self):
        if self.current_epoch - self.best_val_loss_epoch > self.early_stop_epochs:
            return True
        else:
            return False