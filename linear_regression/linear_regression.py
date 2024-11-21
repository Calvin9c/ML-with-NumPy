from .base_regression import BaseRegression
import numpy as np

class LinearRegression(BaseRegression):

    def __init__(self, num_features: int):
        super().__init__(num_features)
    
    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        
        """
        solve the weights using normal equation.
        
        normal equation:
            A^{H} * A * x = A^{H} * b
        rewrite it as:
            features^{H} * features * weights = features^{H} * targets
        thus, we have
            weights = (features^{H} * features)^{-1} * features^{H} * targets
        """

        num_samples, _ = features.shape
        
        bias = np.ones((num_samples, 1))
        features = np.hstack([bias, features])
        
        # Calculate weights using the normal equation
        weights = np.linalg.pinv(features.T @ features) @ features.T @ targets
        
        self.bias = weights[0]
        self.weights = weights[1:]
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        return super().forward(features)