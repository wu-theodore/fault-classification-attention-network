import numpy as np
import torch

class EarlyStopping(object):
    def __init__(self, patience=10, delta=0.01, checkpoint_path='early_stop_model.pt', save_checkpoints=True):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.save_checkpoints = save_checkpoints
        self.counter = 0
        self.min_loss = np.Inf

    def __call__(self, val_loss, model):
        if val_loss > self.min_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping due to no improvement in validation loss for {self.patience} epochs")
                model.load_state_dict(torch.load(self.checkpoint_path))
                return True
        else:
            self.min_loss = val_loss
            self.counter = 0
            if self.save_checkpoints:
                self.save_checkpoint(model)
            return False

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
