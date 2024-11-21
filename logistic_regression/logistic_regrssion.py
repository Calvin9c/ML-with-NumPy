from .base_regression import BaseRegression
import numpy as np

class LogisticRegression(BaseRegression):
    
    def __init__(self, num_features: int, num_classes: int):
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros((1, num_classes))

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        linear_output = super().forward(features)
        return self.softmax(linear_output)

    def predict(self, features: np.ndarray) -> np.ndarray:
        pred = self.forward(features)
        return np.argmax(pred, axis=1)
