import numpy as np

class Optimizer:
    def __init__(self, model, learning_rate: float = 1e-4):
        self.model = model
        self.learning_rate = learning_rate
    
    def step(self, dw, db) -> None:
        self.model.weights -= self.learning_rate * dw
        self.model.bias -= self.learning_rate * db

class AdamOptimizer:
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m_w, self.v_w = 0, 0
        self.m_b, self.v_b = 0, 0
        self.t = 0  # Time step

    def step(self, dw, db) -> None:
        self.t += 1
        # Update biased first moment estimate
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        # Update biased second moment estimate
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
        # Compute bias-corrected first and second moment estimates
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        # Update weights and bias
        self.model.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        self.model.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)