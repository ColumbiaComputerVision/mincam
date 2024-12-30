import torch
import torch.nn as nn
from training import camera_models

class Loss(nn.Module):
    def __init__(self, task_loss):
        super().__init__()
        self.task_loss = task_loss


    def forward(self, model, y_pred, y, class_frequencies=None):
        ## Task loss
        loss_task = self.task_loss(y_pred, y)
        # Scale task loss by each item's class frequencies
        if class_frequencies is not None:
            if y.ndim == 2: # y is a PDF
                class_idx = y.argmax(1)
            else: # y is a class index
                class_idx = y.to(torch.int64)
            sample_frequencies = class_frequencies[class_idx]
            w = 1 / sample_frequencies
            loss_task *= w / (1 / class_frequencies).sum()
        loss_task = loss_task.mean()
        
        ## Sum all loss components (currently only using the task loss)
        loss = loss_task

        loss_dict = {
            "loss_batch": loss,
            "loss_task": loss_task,
        }

        return loss_dict
