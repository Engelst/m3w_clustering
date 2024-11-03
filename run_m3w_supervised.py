import argparse
import numpy as np
from .M3W import border_tools as bt, BorderPeel, clustering_tools as ct
from feature_extraction import MultiViewFeatureExtractor
from supervised_clustering import SupervisedClusterNet, SupervisedClusteringTrainer
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n-clusters', type=int, required=True)
    parser.add_argument('--no-labels', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    data, labels = ct.read_data(args.input, has_labels=not args.no_labels)
    
    # 特征提取
    print("Extracting initial features...")
    feature_extractor = MultiViewFeatureExtractor(
        n_components=32,
        n_neighbors=10
    )
    
    # 获取多视角特征和伪标签
    features = feature_extractor.extract_features(data)
    pseudo_labels = feature_extractor.get_pseudo_labels(features, args.n_clusters)
    
    # 初始化监督学习模型
    print("Training supervised clustering model...")
    model = SupervisedClusterNet(
        input_dim=data.shape[1],
        hidden_dims=[256, 128, 64],
        n_clusters=args.n_clusters
    )
    
    trainer = SupervisedClusteringTrainer(model, device=device)
    trainer.train(data, pseudo_labels, epochs=100)
    
    # 获取最终聚类结果
    print("Generating final clustering results...")
    final_labels, final_features = trainer.predict(data)
    
    # 使用M3W进行最终优化
    print("Running M3W optimization...")
    lambda_estimate = bt.estimate_lambda(final_features, k=8)
    bp = BorderPeel.BorderPeel(
        mean_border_eps=0.15,
        max_iterations=3,
        k=8,
        min_cluster_size=2,
        dist_threshold=lambda_estimate,
        convergence_constant=0,
        link_dist_expansion_factor=1.6,
        verbose=True,
        border_precentile=0.1,
        stopping_precentile=0.01,
        core_points_threshold=0.6,
        dvalue_threshold=0
    )
    
    final_membership = bp.fit_predict(final_features, final_labels)
    
    # 保存结果
    print("Saving results...")
    clusters = []
    nonzero_indices = np.nonzero(final_membership)
    with open(args.output, "w") as f:
        for col in range(final_membership.shape[1]):
            if col in nonzero_indices[1]:
                indices = nonzero_indices[1] == col
                clus = nonzero_indices[0][indices]
            else:
                clus = np.array([-1])
            clusters.append(np.amax(clus))
            f.write(f"{np.amax(clus)}\n")
    
    # 评估结果
    if not args.no_labels:
        # 评估伪标签质量
        pseudo_ari = adjusted_rand_score(labels, pseudo_labels)
        pseudo_ami = adjusted_mutual_info_score(labels, pseudo_labels)
        print("\nPseudo Labels Quality:")
        print(f"ARI: {pseudo_ari:.3f}")
        print(f"AMI: {pseudo_ami:.3f}")
        
        # 评估监督学习结果
        supervised_ari = adjusted_rand_score(labels, final_labels)
        supervised_ami = adjusted_mutual_info_score(labels, final_labels)
        print("\nSupervised Clustering Quality:")
        print(f"ARI: {supervised_ari:.3f}")
        print(f"AMI: {supervised_ami:.3f}")
        
        # 评估最终结果
        final_ari = adjusted_rand_score(labels, clusters)
        final_ami = adjusted_mutual_info_score(labels, clusters)
        print("\nFinal Results (after M3W):")
        print(f"ARI: {final_ari:.3f}")
        print(f"AMI: {final_ami:.3f}")

if __name__ == "__main__":
    main()