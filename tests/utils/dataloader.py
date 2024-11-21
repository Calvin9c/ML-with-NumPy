import numpy as np

class Dataloader:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        self.features = features
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = features.shape[0]
        self.indices = np.arange(self.num_samples)
        self.current_index = 0

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        start_index = self.current_index
        end_index = min(self.current_index + self.batch_size, self.num_samples)
        self.current_index = end_index

        batch_indices = self.indices[start_index:end_index]

        batch_features = self.features[batch_indices]
        batch_targets = self.targets[batch_indices]

        return batch_features, batch_targets
    
    def __len__(self) -> int:
        return self.num_samples