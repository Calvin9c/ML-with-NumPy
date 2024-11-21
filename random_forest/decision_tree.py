import numpy as np
from collections import deque

class Impurity:

    """
    A class to compute the impurity of a dataset \\
    based on Gini or Entropy criterion.
    """

    def __init__(self, criterion: str = "gini"):

        if criterion == 'gini':
            self.criterion = 'gini'
        elif criterion == 'entropy':
            self.criterion = 'entropy'
        else:
            raise NotImplementedError

    def _gini(self, prob):
        """Calculate Gini impurity."""
        return 1 - np.sum(prob ** 2)

    def _entropy(self, prob):
        """Calculate entropy."""
        return -np.sum(prob * np.log2(prob))

    def __call__(self, x: np.ndarray):

        """
        Compute impurity based on the chosen criterion.
        
        Parameters:
            x (np.array): Array of labels or classes.
        
        Returns:
            float: Impurity value.
        """

        num_samples = x.shape
        _, counts = np.unique(x, return_counts=True)
        prob = counts / num_samples

        if self.criterion == 'gini':
            return self._gini(prob)
        elif self.criterion == 'entropy':
            return self._entropy(prob)
        else:
            raise NotImplementedError

class TreeNode:
    def __init__(
        self,
        depth: int,
        split_feature: int = None,
        split_value: float = None,
        label: int = None,
        is_leaf: bool = False
    ):
        self.depth = depth
        self.split_feature = split_feature    # Feature index used for splitting
        self.split_value = split_value        # Threshold value for the split   
        self.label = label                    # Class label (for leaf nodes)
        self.is_leaf = is_leaf 
        self.left = None                      # Left child node
        self.right = None                     # Right child node

class DecisionTree:

    """
    A Decision Tree classifier using Gini or Entropy impurity criteria.
    The tree is built iteratively using a breadth-first approach.
    """

    def __init__(
        self,
        criterion: str = 'gini',
        max_features: int | str = 'sqrt',
        max_depth: int = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0
    ):

        """
        Initialize the Decision Tree.

        Parameters:
            criterion (str): The impurity criterion ('gini' or 'entropy').
            max_features (int): The maximum number of features to consider for splitting.
            max_depth (int): The maximum depth of the tree.
            min_samples_split (int): The minimum number of samples required to split a node.
            min_impurity_decrease (float): Minimum impurity decrease required to split.
        """

        self.root = None
        self.criterion = Impurity(criterion)
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.feature_importance = {}
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

    def _impurity_gain(
        self,
        labels: np.ndarray,
        left_labels: np.ndarray,
        right_labels: np.ndarray
    ) -> float:

        """
        Calculate the impurity gain from a potential split.

        Parameters:
            labels (np.array): Labels of the parent node.
            left_labels (np.array): Labels of the left child node.
            right_labels (np.array): Labels of the right child node.

        Returns:
            float: The impurity gain.
        """

        num_samples = len(labels)
        w_left = len(left_labels) / num_samples
        w_right = len(right_labels) / num_samples

        impurity_parent = self.criterion(labels)
        impurity_children = \
            w_left * self.criterion(left_labels) + w_right * self.criterion(right_labels)
        impurity_gain = impurity_parent - impurity_children

        return impurity_gain

    def _get_masks(
        self,
        feature: np.ndarray,
        threshold: float
    ) -> tuple:
        
        left_mask = feature <= threshold
        right_mask = ~left_mask
        
        return left_mask, right_mask

    def _best_split(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> tuple:

        """
        Find the best feature and value to split on by maximizing impurity gain.

        Parameters:
            features (np.array): Feature matrix.
            labels (np.array): Labels.

        Returns:
            tuple: (best_split_feature, best_split_value)
        """

        _, num_features = features.shape
        best_impurity_gain = self.min_impurity_decrease
        best_split_feature, best_split_val = None, None
        
        if self.max_features is None:
            selected_features = np.arange(num_features)
            
        elif isinstance(self.max_features, int):
            selected_features = \
                np.random.choice(num_features, min(self.max_features, num_features), replace=False)
                
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                selected_features = \
                    np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)
            elif self.max_features == 'log2':
                selected_features = \
                    np.random.choice(num_features, max(1, int(np.log2(num_features))), replace=False)
            else:
                raise NotImplementedError(f"Unsupported max_features option: '{self.max_features}'")

        else:
            raise ValueError("max_features must be an integer, 'sqrt', 'log2', or None.")

        for split_feature in selected_features:
            
            feature = features[:, split_feature]

            for split_val in np.unique(feature):

                left_mask, right_mask = self._get_masks(feature, split_val)
                left_labels, right_labels = labels[left_mask], labels[right_mask]

                if len(left_labels) < self.min_samples_split or len(right_labels) < self.min_samples_split:
                    continue
                
                impurity_gain = \
                    self._impurity_gain(labels, left_labels, right_labels)
                
                if impurity_gain > best_impurity_gain:
                    best_impurity_gain = impurity_gain
                    best_split_feature = split_feature
                    best_split_val = split_val
        
        # Update feature importance based on impurity gain
        if best_split_feature is not None:
            self.feature_importance[best_split_feature] = \
                self.feature_importance.get(best_split_feature, 0) + best_impurity_gain
        
        return best_split_feature, best_split_val

    def _normalize_feature_importance(self):
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            self.feature_importance = {feature: importance / total_importance 
                                    for feature, importance in self.feature_importance.items()}

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> None:

        """
        Build the decision tree using an iterative approach.

        Parameters:
            features (np.array): Feature matrix.
            labels (np.array): Labels.
        """

        assert 0 < len(features) and len(features) == len(labels), \
            "Features and labels must be non-empty and of same length"

        self.root = TreeNode(depth=0)
        num_samples, _ = features.shape
        mask = np.ones(num_samples, dtype=bool)

        queue = deque([(self.root, mask)])

        while queue:

            node, mask = queue.popleft()
            node_labels = labels[mask]

            if node.depth < self.max_depth and 1 < len(np.unique(node_labels)):
                
                node_features = features[mask]

                split_feature, split_value = \
                    self._best_split(node_features, node_labels)

                if split_feature is not None:

                    node.split_feature, node.split_value = \
                        split_feature, split_value
                    
                    # Create left and right masks based on the best split
                    left_mask, right_mask = self._get_masks(features[:, split_feature], split_value)
                    left_mask, right_mask = mask & left_mask, mask & right_mask

                    # Create child nodes and update the queue
                    node.left, node.right = \
                        TreeNode(depth=node.depth+1), TreeNode(depth=node.depth+1)
                    queue.extend([(node.left, left_mask), (node.right, right_mask)])
                    
                    continue # Skip to next iteration if split was successful
            
            # If no valid split, assign label and mark as leaf
            node.label = np.argmax(np.bincount(node_labels))
            node.is_leaf = True

        self._normalize_feature_importance()

    def __call__(
        self,
        features: np.ndarray,
        labels: np.ndarray = None
    ):
        
        if self.root is None:
            assert features is not None and labels is not None
            self.fit(features, labels)
        else:
            assert isinstance(self.root, TreeNode)
            return self.predict(features)
        
        return

    def predict_single(self, sample: np.ndarray) -> int:

        """
        Predict the class label for a single sample.

        Parameters:
            sample (np.ndarray): A single data point (feature vector).
        
        Returns:
            int: Predicted class label.
        """

        if self.root is None:
            raise ValueError("The decision tree has not been trained. Please fit the model first.")

        node = self.root
        while not node.is_leaf:
            if sample[node.split_feature] <= node.split_value:
                node = node.left
            else:
                node = node.right
        return node.label

    def predict(self, batch: np.ndarray) -> np.ndarray:

        """
        Predict class labels for a batch of data points.

        Parameters:
            batch (np.ndarray): Feature matrix for multiple data points.
        
        Returns:
            np.ndarray: Predicted class labels.
        """

        if batch.size == 0:
            return np.array([])

        return np.array([self.predict_single(x) for x in batch])

    def get_feature_importance(self):
        return self.feature_importance