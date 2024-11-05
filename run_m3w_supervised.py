import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import border_tools as bt
import clustering_tools as ct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n-clusters', type=int, required=True)
    parser.add_argument('--no-labels', action='store_true')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading data...")
    # 修正参数名称从 has_labels 到 hasLabels
    data, labels = ct.read_data(args.input, hasLabels=not args.no_labels)
    print(f"\nGenerated data shape: {data.shape}")
    print(f"\nNumber of labels: {len(labels)}")
    print(f"\nUnique labels: {np.unique(labels)}")

    # 特征提取和降维
    print("Extracting initial features...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    data_embedded = tsne.fit_transform(data_scaled)

    # 运行M3W优化
    print("Running M3W optimization...")
    lambda_estimate = bt.estimate_lambda(data_embedded, k=8)

    final_labels, core_points, _, iterations, _, _, border_values, core_indices = bt.border_peel_rknn_exp_transform_local(
        data_embedded,
        k=7,
        threshold=0.5,
        iterations=150,
        verbose=True
    )

    # 保存结果
    print("Saving results...")
    with open(args.output, "w") as f:
        for label in final_labels:
            f.write(f"{label}\n")

    # 评估结果
    if not args.no_labels:
        ari = adjusted_rand_score(labels, final_labels)
        ami = adjusted_mutual_info_score(labels, final_labels)
        print("\nClustering Quality:")
        print(f"ARI: {ari:.3f}")
        print(f"AMI: {ami:.3f}")


if __name__ == "__main__":
    main()