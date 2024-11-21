import numpy as np

class MSELoss:

    def __init__(self):
        self.features = None
        self.delta = None

    def forward(
        self,
        features: np.ndarray,
        pred: np.ndarray,
        targets: np.ndarray
    ) -> float:

        self.features = features
        self.delta = targets - pred
        loss = np.mean(self.delta ** 2)
        return loss
    
    def backward(self) -> tuple:
        
        # Compute gradients with respect to weights and bias
        
        num_samples, _ = self.features.shape
        dw = -(2 / num_samples) * (self.features.T @ self.delta)
        db = -(2 / num_samples) * np.sum(self.delta)
        
        self.features = None
        self.delta = None

        return dw, db

    def __call__(
        self,
        features: np.ndarray,
        pred: np.ndarray,
        targets: np.ndarray,
    ) -> tuple:
        
        return self.forward(features, pred, targets)

class CrossEntropyLoss:
    def __init__(self):
        self.features = None
        self.predictions = None
        self.targets = None
        self.epsilon = 1e-15

    def forward(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        計算交叉熵損失
        :param predictions: 預測的概率分佈 (softmax 輸出)，形狀為 (num_samples, num_classes)
        :param targets: 真實標籤 (整數標籤)，形狀為 (num_samples,)
        :return: 平均損失值
        """
        
        self.features = features
        self.predictions = predictions
        self.targets = targets

        num_samples, _ = predictions.shape
        # avoid log(0)
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        log_likelihood = -np.log(predictions[np.arange(num_samples), targets])
        loss = np.mean(log_likelihood)
        return loss

    def backward(self) -> tuple:
        """
        計算交叉熵損失的梯度
        :return: 梯度 dw, db
        """
        num_samples, _ = self.predictions.shape
        # One-hot 編碼
        targets_one_hot = np.zeros_like(self.predictions)
        targets_one_hot[np.arange(num_samples), self.targets] = 1

        # 計算梯度
        error = self.predictions - targets_one_hot
        dw = (self.features.T @ error) / num_samples
        db = np.sum(error, axis=0, keepdims=True) / num_samples

        self.features = None
        self.predictions = None
        self.targets = None

        return dw, db
    
    def __call__(self, features: np.ndarray, pred: np.ndarray, targets: np.ndarray):
        return self.forward(features, pred, targets)