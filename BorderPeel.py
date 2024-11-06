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
    def __init__(self, k=7, max_iterations=10, mean_border_eps=0.01,
                 plot_debug_output_dir=None, verbose=True, n_clusters=None,
                 **kwargs):  # 添加 n_clusters 参数和 kwargs
        super().__init__()
        self.k = k
        self.max_iterations = max_iterations
        self.mean_border_eps = mean_border_eps
        self.plot_debug_output_dir = plot_debug_output_dir
        self.verbose = verbose
        self.n_clusters = n_clusters  # 新增

    @device_performance_monitor
    def fit(self, X, X_plot_projection=None):
        # 确保输入数据在正确设备上
        X = self.ensure_tensor_on_device(X)

        # 计算初始边界值
        border_values = self._compute_border_values(X)
        mean_border = torch.mean(border_values)

        # 迭代优化
        for iteration in range(self.max_iterations):
            old_mean_border = mean_border

            # 更新边界值
            border_values = self._compute_border_values(X)
            mean_border = torch.mean(border_values)

            if self.verbose:
                print(f"Iteration {iteration + 1}, Mean Border: {mean_border:.4f}")

            # 检查收敛
            if torch.abs(old_mean_border - mean_border) < self.mean_border_eps:
                break

        return border_values.cpu().numpy()

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