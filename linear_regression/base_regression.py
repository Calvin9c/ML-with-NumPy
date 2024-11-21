import os
import numpy as np
from .path_utils import *

class BaseRegression:

    def __init__(self, num_features: int):
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def forward(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        return self.forward(features)

    def load(self, weights: str) -> None:
        
        if not os.path.exists(weights):
            raise ValueError(f"{weights} doesn't exists.")

        data = np.load(weights, allow_pickle=True).item()
        self.weights = data['weights']
        self.bias = data['bias']

    def save(self, weights: str='regression_model/weights.npy') -> None:
        
        weights = Path(weights)
        
        if is_dir(weights):
            weights = weights / "weights.npy"
        
        mkdir(weights.parent)

        np.save(weights, {'weights': self.weights, 'bias': self.bias})