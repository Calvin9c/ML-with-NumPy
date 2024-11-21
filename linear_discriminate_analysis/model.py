from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt

class LDA:
    def __init__(
        self,
        num_components: int = 2,
        num_neighbors: int = 3
    ) -> None:
        
        self.num_components = num_components 
        self.num_neighbors = num_neighbors
        self.mean_vectors = None
        self.projection_matrix = None

    def _compute_mean_vectors(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> dict:
        
        """Calculate mean for each class"""
        
        return {label: np.mean(features[labels == label], axis=0)
                for label in np.unique(labels)}

    def _compute_within_class_scatter(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:

        """Calculate within-class scatter matrix"""

        _, num_features = features.shape

        within_class_scatter = \
            np.zeros((num_features, num_features))
            
        for cls in np.unique(labels):
            Xi = features[labels == cls] - self.mean_vectors[cls]
            within_class_scatter += Xi.T @ Xi

        return within_class_scatter

    def _compute_between_class_scatter(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:

        """Calculate between-class scatter matrix"""

        _, num_features = features.shape
        
        between_class_scatter = \
            np.zeros((num_features, num_features))
        
        overall_mean = np.mean(features, axis=0)
        
        for cls in np.unique(labels):
            n, _ = features[labels == cls].shape
            mean_diff = self.mean_vectors[cls] - overall_mean
            between_class_scatter += n * mean_diff[:, None] @ mean_diff[None, :]
        
        return between_class_scatter

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> None:
        
        """Calculate projection matrix"""
        
        _, num_features = features.shape
        num_labels = len(np.unique(labels))
        assert self.num_components <= min(num_labels-1, num_features)

        self.mean_vectors = \
            self._compute_mean_vectors(features, labels)

        within_class_scatter = \
            self._compute_within_class_scatter(features, labels)

        between_class_scatter = \
            self._compute_between_class_scatter(features, labels)

        eigenvalues, eigenvectors = \
            np.linalg.eig(np.linalg.pinv(within_class_scatter) @ between_class_scatter)

        # Sort eigenvectors by corresponding eigenvalues in descending order
        sorted_indices = np.argsort(-eigenvalues.real) # Negative sign to sort in descending order
        self.projection_matrix = eigenvectors[:, sorted_indices[:self.num_components]].real

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x @ self.projection_matrix

    def predict_nearest_mean(self, features: np.ndarray) -> np.ndarray:
        
        """Predict by nearest mean"""

        if self.projection_matrix is None:
            raise ValueError("")

        proj_features = self.transform(features)
        proj_means = {label: self.transform(mean_vector)
                      for label, mean_vector in self.mean_vectors.items()}
        
        labels = list(self.mean_vectors.keys())
        mean_matrix = \
            np.stack([proj_means[label] for label in labels])

        distances = np.linalg.norm(proj_features[:, None] - mean_matrix, axis=2)
        pred_indices = np.argmin(distances, axis=1)
        return np.array([labels[idx] for idx in pred_indices])
    
    def predict_knn(
        self,
        features: np.ndarray,
        training_features: np.ndarray,
        training_labels: np.ndarray
    ) -> np.ndarray:

        """Predict by kNN"""

        if self.projection_matrix is None:
            raise ValueError("")
        
        # Project the features and training data into the LDA space
        proj_features = self.transform(features)
        proj_training_features = self.transform(training_features)
        
        # Calculate distances and find the nearest neighbors
        distances = np.linalg.norm(proj_features[:, None] - proj_training_features, axis=2)
        nearest_indices = np.argsort(distances, axis=1)[:, :self.num_neighbors]
        nearest_labels = training_labels[nearest_indices]
        
        return mode(nearest_labels, axis=1).mode.flatten()
    
    # ========== ========== ========== ========== #
    # member function for plot
    # ========== ========== ========== ========== #

    def _scatter(
        self,
        ax: plt.Axes,
        features: np.ndarray,
        labels: np.ndarray,
        cmap: str = 'viridis',
        size: int = 32,
        edgecolor: str = 'k',
        marker: str = 'o',
        alpha: float = 0.5
    ) -> None:
        
        num_samples, num_features = features.shape

        if num_features == 1:
            ax.scatter(features[:, 0], np.zeros(num_samples), c=labels,
                       cmap=cmap, s=size, edgecolors=edgecolor, marker=marker, alpha=alpha)
        elif num_features == 2:
            ax.scatter(features[:, 0], features[:, 1], c=labels,
                       cmap=cmap, s=size, edgecolor=edgecolor, marker=marker, alpha=alpha)
        elif num_features == 3:
            ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels,
                       cmap=cmap, s=size, edgecolor=edgecolor, marker=marker, alpha=alpha)
        else:
            print("Only 1D, 2D and 3D features are supported.")
            return

    def _project(self, features: np.ndarray) -> np.ndarray:
        
        _, num_features = features.shape

        lda_normal = self.projection_matrix[:, 0]
        lda_normal /= np.linalg.norm(lda_normal)

        # According to the formula of orthogonal projection:
        #     proj = <x, y> / <y, y> * y
        if num_features == 2:
            projected_points = (features @ lda_normal)[:, None] * lda_normal
        elif num_features == 3:           
            projected_points = features - (features @ lda_normal)[:, None] * lda_normal
        else:
            raise ValueError
                
        return projected_points
                
    # ========== ========== ========== ========== #
    # member function for _plot_2D
    # ========== ========== ========== ========== #

    def _plot_lda_line(self, ax: plt.Axes, features: np.ndarray) -> None:

        direction = self.projection_matrix[:, 0]
        direction /= np.linalg.norm(direction)
        
        normal = np.array([direction[1], (-1)*direction[0]])
        
        overall_mean = np.mean(features, axis=0)[None, :] # [num_features,] -> [1, num_features]
        projected_mean = self._project(overall_mean) # with shape [1, num_features]

        # get the point on the line
        px, py = projected_mean[0][0], projected_mean[0][1]
        
        # L: wx*x + wy*y = k
        wx, wy = normal[0], normal[1]
        k = wx*px + wy*py
        
        # L: y = (-wx/wy) * x + k       
        projected_features = self._project(features)
        x_min = min(features[:, 0].min(), projected_features[:, 0].min())
        x_max = max(features[:, 0].max(), projected_features[:, 1].max())
        x = np.linspace(x_min, x_max, 10)
        
        y = (-wx/wy) * x + k
        ax.plot(x, y, 'k-', linewidth=0.8, alpha=0.5)

    # ========== ========== ========== ========== #
    # member function for _plot_3D
    # ========== ========== ========== ========== #

    def _plot_lda_plane(self, ax: plt.Axes, features: np.ndarray) -> None:
       
        normal = self.projection_matrix[:, 0]
        
        overall_mean = np.mean(features, axis=0)[None, :] # [num_features,] -> [1, num_features]
        projected_mean = self._project(overall_mean) # with shape [1, num_features]
        
        # get the point on the surface
        px, py, pz = projected_mean[0][0], projected_mean[0][1], projected_mean[0][2]
        
        # E: wx*x + wy*y + wz*z = k
        wx, wy, wz = normal[0], normal[1], normal[2]
        k = wx*px + wy*py + wz*pz
        
        # E: z = [-(wx*x+wy*y) + k] / wz
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        x, y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
        z = (-(wx*x+wy*y) + k) / wz
        ax.plot_surface(x, y, z, color='orange', alpha=0.25)       
        
    def plot(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        features_mapping: dict = None,
        labels_mapping: dict = None
    ) -> None:

        num_samples, num_features = features.shape

        if features_mapping is None:
            features_mapping = {i: f"Feature {i}"
                                for i in range(num_features)}
        unique_labels = np.unique(labels)
        if labels_mapping is None:
            labels_mapping = {cls: f"Class {cls}"
                            for cls in unique_labels}

        if num_features in [1, 2, 3]:
            
            fig = plt.figure(figsize=(12, 8))
            
            if num_features == 1:
                ax = fig.add_subplot(111)
                ax.set_xlabel(features_mapping[0])
                ax.set_title("Data Points in 1D")
            elif num_features == 2:
                ax = fig.add_subplot(111)
                ax.set_xlabel(features_mapping[0])
                ax.set_ylabel(features_mapping[1])
                ax.set_title("Data Points in 2D")
            elif num_features == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel(features_mapping[0])
                ax.set_ylabel(features_mapping[1])
                ax.set_zlabel(features_mapping[2])
                ax.set_title("Data Points in 3D")
            
            self._scatter(ax, features, labels)
        
            if num_features == 2 and self.num_components == 1:
                self._plot_lda_line(ax, features)
                projected_points = self._project(features)
                self._scatter(ax, projected_points, labels)
                for i in range(num_samples):
                    ax.plot([features[i, 0], projected_points[i, 0]],
                            [features[i, 1], projected_points[i, 1]],
                            'k--', linewidth=0.6, alpha=0.25)
            elif num_features == 3 and self.num_components == 2:
                self._plot_lda_plane(ax, features)
                projected_points = self._project(features)
                self._scatter(ax, projected_points, labels)
                for i in range(num_samples):
                    ax.plot([features[i, 0], projected_points[i, 0]], 
                            [features[i, 1], projected_points[i, 1]], 
                            [features[i, 2], projected_points[i, 2]], 
                            'k--', linewidth=0.6, alpha=0.25)
            
            plt.savefig('original_data.png')
            plt.show()
        
        if self.num_components in [1, 2, 3]:
            
            fig = plt.figure(figsize=(12, 8))
            
            if self.num_components == 1:
                ax = fig.add_subplot(111)
                ax.set_xlabel(f'lda features 0')
                ax.set_title("1D lda feature")
            elif self.num_components == 2:
                ax = fig.add_subplot(111)
                ax.set_xlabel(f'lda features 0')
                ax.set_ylabel(f'lda features 1')
                ax.set_title("2D lda features")
            elif self.num_components == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel(f'lda features 0')
                ax.set_ylabel(f'lda features 1')
                ax.set_zlabel(f'lda features 2')
                ax.set_title("3D lda features")
                
            transfrom_features = self.transform(features)
            self._scatter(ax, transfrom_features, labels)

            plt.savefig('lda_features.png')
            plt.show()