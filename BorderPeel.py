from device_fix import (
    ensure_tensor,
    process_batch,
    DeviceAwareModule,
    device_performance_monitor,
    DEVICE
)
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


class BorderPeel(DeviceAwareModule):
    def __init__(self, k=7, max_iterations=100, mean_border_eps=0.01, plot_debug_output_dir=None, verbose=True):
        self.k = k
        self.max_iterations = max_iterations
        self.mean_border_eps = mean_border_eps
        self.plot_debug_output_dir = plot_debug_output_dir
        self.verbose = verbose

    def fit(self, X, X_plot_projection=None):
        # Your existing fit code here
        pass

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels
        """
        # Call fit method
        self.fit(X)

        # Return the cluster assignments
        return self.labels_

    def _compute_border_values(self, X):
        # 计算k近邻
        X_np = X.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(X_np)
        distances, _ = nbrs.kneighbors()

        # 转换为张量并移动到正确设备
        distances = self.ensure_tensor_on_device(distances)

        # 计算局部缩放因子
        k_distances = torch.sort(distances, dim=1)[0][:, self.k - 1]
        k_distances = k_distances.reshape(-1, 1)

        # 计算边界值
        border_values = torch.exp(-distances * distances / (k_distances * k_distances))
        return torch.mean(border_values, dim=1)