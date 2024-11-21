import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

class SKLDA:

    '''A wrapper class of LDA implementation by sklearn'''

    def __init__(
        self,
        num_components: int = 2,
        num_neighbors: int = 3
    ):
        self.model = LinearDiscriminantAnalysis(n_components=num_components)
        self.clf = KNeighborsClassifier(n_neighbors=num_neighbors)
      
        self.num_components = num_components

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.model.transform(x)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> None:
        self.model.fit(features, labels)
        self.clf.fit(self.model.transform(features), labels)

    def predict_knn(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.model.transform(features))

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

        lda_normal = self.model.coef_[0]
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

        direction = self.model.coef_[0]
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
       
        normal = self.model.coef_[0]
        
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
            
            plt.savefig('sk_original_data.png')
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

            plt.savefig('sk_lda_features.png')
            plt.show()