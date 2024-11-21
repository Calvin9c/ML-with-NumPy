from .decision_tree import DecisionTree
import numpy as np
from tqdm import tqdm

class RandomForest:
    
    """
    A Random Forest classifier consisting of multiple decision trees.    
    """

    def __init__(
        self,
        num_estimators: int = 10, # Number of trees in the forest.
        bootstrap: bool = True, # Whether to use bootstrap sampling.
        random_seed: int = None,
        criterion: str = 'gini', # Impurity criterion for the trees ('gini' or 'entropy').
        max_features: int | str = 'sqrt', # Maximum number of features to consider at each split.
        max_depth: int = None, # Maximum depth of each tree.
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0
    ):

        assert num_estimators > 0, "num_estimators must be a positive integer."
        assert criterion in ["gini", "entropy"], "Criterion must be either 'gini' or 'entropy'."

        self.num_estimators = num_estimators
        self.bootstrap = bootstrap
        self.random_seed = random_seed
        self.forest = []
        self.oob_samples = [] 
        
        self.criterion = criterion
        self.max_features = max_features        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

    def _sample(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_size: int = None
    ) -> tuple:
        
        """
        Generate a sample from the dataset, with optional bootstrap sampling.
        
        Parameters:
            features (np.ndarray): Feature matrix.
            labels (np.ndarray): Label vector.
            sample_size (int): Number of samples to draw. Defaults to the size of the original dataset.
            
        Returns:
            tuple: (sampled features, sampled labels, oob_indices)
                - sampled features: Sampled feature matrix.
                - sampled labels: Sampled label vector.
                - oob_indices (np.ndarray): Indices of Out-of-Bag samples, if bootstrap is True.
        """
        
        num_samples, _ = features.shape
        sample_size = sample_size or num_samples

        if sample_size > num_samples and not self.bootstrap:
            raise ValueError("Sample size cannot exceed the number of samples when sampling without replacement.")

        sample_indices = \
            np.random.choice(num_samples, sample_size, replace=self.bootstrap)

        # Identify OOB samples if bootstrap sampling is enabled
        oob_indices = np.setdiff1d(np.arange(num_samples), sample_indices) \
                      if self.bootstrap else None

        return features[sample_indices], labels[sample_indices], oob_indices

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_size: int = None
    ):
        
        """
        Train the random forest on the provided dataset.
        
        Parameters:
            features (np.ndarray): Feature matrix.
            labels (np.ndarray): Label vector.
        """
        
        assert features.shape[0] == labels.shape[0], \
            "Features and labels must have the same number of samples."
        
        self.forest = []
        self.oob_samples = [] 

        if self.random_seed is not None:
            np.random.seed(self.random_seed) 
        
        for _ in tqdm(range(self.num_estimators)):

            tree = DecisionTree(
                criterion=self.criterion,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease
            )

            features_sample, labels_sample, oob_indices = \
                self._sample(features, labels, sample_size)
            
            # Store OOB indices if bootstrap is enabled
            if oob_indices is not None:
                self.oob_samples.append(oob_indices)
            
            # Train the tree on the bootstrap sample
            tree.fit(features_sample, labels_sample)
            self.forest.append(tree)

    def oob_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the Out-of-Bag (OOB) score if bootstrap sampling is used.
        
        Parameters:
            features (np.ndarray): Feature matrix.
            labels (np.ndarray): Label vector.
        
        Returns:
            float: OOB accuracy score.
        """
        if not self.bootstrap:
            raise ValueError("OOB score is only available when bootstrap is set to True.")

        predictions = np.zeros((features.shape[0], self.num_estimators))
                
        # Populate predictions for OOB samples for each tree
        for tree_idx, tree in enumerate(self.forest):
            for sample in self.oob_samples[tree_idx]:
                predictions[sample, tree_idx] = tree.predict_single(features[sample])

        # Majority vote for OOB samples, ignoring zero values (non-predicted)
        def majority_vote(row: np.ndarray):
            non_zero_predictions = row[row > 0].astype(int)
            return np.bincount(non_zero_predictions).argmax() if non_zero_predictions.size > 0 else -1

        final_predictions = np.apply_along_axis(majority_vote, axis=1, arr=predictions)
        
        # Filter out samples with no OOB predictions (-1) and calculate OOB accuracy
        valid_indices = final_predictions != -1
        accuracy = np.mean(final_predictions[valid_indices] == labels[valid_indices])
        
        return accuracy

    def predict_single(self, sample: np.ndarray) -> int:
        """
        Predict the class label for a single sample by aggregating predictions from all trees in the forest.

        Parameters:
            sample (np.ndarray): A single data point (feature vector).
        
        Returns:
            int: Predicted class label based on majority vote.
        """

        pred = [tree.predict_single(sample) for tree in self.forest]

        return np.bincount(pred).argmax()
    
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a batch of samples by calling predict_single on each sample.
        
        Parameters:
            batch (np.ndarray): Feature matrix for multiple data points.
        
        Returns:
            np.ndarray: Array of predicted class labels for each data point.
        """

        return np.array([self.predict_single(sample) for sample in batch])
    
    def __call__(
        self,
        features: np.ndarray,
        labels: np.ndarray = None,
        sample_size: int = None
    ):
        assert isinstance(self.forest, list)
        
        if self.forest:
            return self.predict(features)
        else:
            assert features is not None and labels is not None
            self.fit(features, labels, sample_size)
        
        return