from device_fix import (
    ensure_tensor,
    process_batch,
    device_performance_monitor,
    DEVICE
)
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


@device_performance_monitor
def compute_knn_graph(X, k):
    """计算k近邻图"""
    X = ensure_tensor(X)
    X_np = X.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_np)
    distances, indices = nbrs.kneighbors()
    return ensure_tensor(distances), ensure_tensor(indices)


@device_performance_monitor
def compute_border_values(X, k):
    """计算边界值"""
    distances, _ = compute_knn_graph(X, k)

    # 计算局部缩放因子
    k_distances = torch.sort(distances, dim=1)[0][:, k - 1]
    k_distances = k_distances.reshape(-1, 1)

    # 计算边界值
    border_values = torch.exp(-distances * distances / (k_distances * k_distances))
    return torch.mean(border_values, dim=1)


@device_performance_monitor
def dynamic_3w(data, k=7, max_iterations=10, mean_border_eps=0.01, verbose=True):
    """动态3W聚类算法"""
    # 确保数据在正确设备上
    data = ensure_tensor(data)

    # 初始化边界值
    border_values = compute_border_values(data, k)
    mean_border = torch.mean(border_values)

    # 迭代优化
    for iteration in range(max_iterations):
        old_mean_border = mean_border

        # 更新边界值
        border_values = compute_border_values(data, k)
        mean_border = torch.mean(border_values)

        if verbose:
            print(f"Iteration {iteration + 1}, Mean Border: {mean_border:.4f}")

        # 检查收敛
        if torch.abs(old_mean_border - mean_border) < mean_border_eps:
            break

    return border_values.cpu().numpy()