import argparse
import numpy as np
import torch
import clustering_tools as ct
import BorderPeel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os


def main():
    args = parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\nLoading data...")
    # 修正参数名为hasLabels
    data, labels = ct.read_data(args.input, hasLabels=not args.no_labels)

    print(f"\nLoaded data shape: {data.shape}")

    if labels is not None:
        n_true_clusters = len(np.unique(labels))
        print(f"\nNumber of true clusters: {n_true_clusters}")

    # 数据标准化
    print("\nStandardizing data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 特征提取
    print("\nExtracting features...")
    features = scaled_data

    # 生成伪标签
    print("\nGenerating pseudo labels...")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        n_init=20,
        max_iter=300,
        random_state=42
    )
    pseudo_labels = kmeans.fit_predict(features)
    print(f"\nGenerated {args.n_clusters} pseudo clusters")

    # 运行M3W聚类
    print("\nRunning M3W clustering...")
    m3w = BorderPeel(
        k=args.k,
        max_iterations=args.max_iterations,
        mean_border_eps=args.mean_border_eps,
        plot_debug_output_dir=args.debug_output_dir,
        verbose=args.verbose
    )

    # 添加fit方法
    m3w.fit(features)
    final_labels = m3w.labels_

    # 评估结果（如果有真实标签）
    if labels is not None:
        nmi = normalized_mutual_info_score(labels, final_labels)
        ari = adjusted_rand_score(labels, final_labels)
        print(f"\nClustering Results:")
        print(f"NMI: {nmi:.4f}")
        print(f"ARI: {ari:.4f}")

    # 保存结果
    if args.output:
        print(f"\nSaving results to {args.output}")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            # 写入评估指标
            if labels is not None:
                f.write(f"NMI: {nmi:.4f}\n")
                f.write(f"ARI: {ari:.4f}\n")
            # 写入聚类标签
            f.write("\nCluster Labels:\n")
            for label in final_labels:
                f.write(f"{label}\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Run M3W clustering with supervision')

    # 必需参数
    parser.add_argument('--input', type=str, required=True,
                        help='Input data file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for results')
    parser.add_argument('--n-clusters', type=int, required=True,
                        help='Number of clusters')

    # 可选参数
    parser.add_argument('--k', type=int, default=7,
                        help='Number of neighbors for border peeling (default: 7)')
    parser.add_argument('--max-iterations', type=int, default=100,
                        help='Maximum number of iterations (default: 100)')
    parser.add_argument('--mean-border-eps', type=float, default=0.01,
                        help='Mean border epsilon threshold (default: 0.01)')
    parser.add_argument('--no-labels', action='store_true',
                        help='Input data does not contain labels')
    parser.add_argument('--debug-output-dir', type=str,
                        help='Directory for debug output plots')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


if __name__ == "__main__":
    main()