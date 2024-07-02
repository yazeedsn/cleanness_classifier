from pathlib import Path
import torch

from ai.config import DEFAULT_MODEL, DEFAULT_MODEL_FC

class CModel():
    def __init__(self, weights:str|Path=None):
        self.model = DEFAULT_MODEL
        self.model.fc = DEFAULT_MODEL_FC
        if weights:
            state_dict = torch.load(weights)
            self.model.load_state_dict(state_dict)
    
    def __call__(self, X):
        return self.model(X)
    
    def __repr__(self) -> str:
        return self.model.__repr__()

    def set_model(self, model:torch.nn.Module):
        self.model = model

    def set_model_fc(self, model:torch.nn.Module):
        self.model.fc = model
    
    def save_state(self, path):
        torch.save(self.model.state_dict(), path)

    def load_state(self, state_dict_file):
        state_dict = torch.load(state_dict_file)
        self.model.load_state_dict(state_dict)

    # only use for inference, never for training.
    def predict(self, X):
        return self.model(X).argmax(dim=1).detach()

    def freezeCNN(self, freeze=True):
        for param in self.model.parameters():
            param.requires_grad_(freeze)
