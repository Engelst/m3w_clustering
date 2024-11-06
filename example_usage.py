import torch

from device_fix import (
    ensure_tensor,
    process_batch,
    device_performance_monitor,
    DEVICE
)
from BorderPeel import BorderPeel
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


@device_performance_monitor
def run_clustering_example():
    # 生成示例数据
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

    # 转换数据到正确设备
    X = ensure_tensor(X)

    # 初始化BorderPeel
    border_peel = BorderPeel(
        k=7,
        max_iterations=10,
        mean_border_eps=0.01,
        verbose=True
    )

    # 运行聚类
    border_values = border_peel.fit(X)

    # 可视化结果
    plt.figure(figsize=(12, 5))

    # 原始数据
    plt.subplot(121)
    plt.scatter(X.cpu().numpy()[:, 0], X.cpu().numpy()[:, 1], c=y_true, cmap='viridis')
    plt.title('Original Data')

    # 聚类结果
    plt.subplot(122)
    plt.scatter(X.cpu().numpy()[:, 0], X.cpu().numpy()[:, 1], c=border_values, cmap='viridis')
    plt.title('Clustering Results')

    plt.tight_layout()
    plt.show()


# 批处理示例
@device_performance_monitor
def process_large_dataset(X, batch_size=1000):
    """处理大规模数据集的示例"""
    total_samples = len(X)
    results = []

    for i in range(0, total_samples, batch_size):
        # 获取当前批次
        batch = X[i:min(i + batch_size, total_samples)]

        # 处理批次数据
        batch = process_batch(batch)

        # 在这里添加你的处理逻辑
        # ...

        results.append(batch)

    return torch.cat(results, dim=0)


if __name__ == "__main__":
    # 运行聚类示例
    run_clustering_example()

    # 大数据集处理示例
    large_X = np.random.randn(10000, 2)
    large_X = ensure_tensor(large_X)
    processed_X = process_large_dataset(large_X)