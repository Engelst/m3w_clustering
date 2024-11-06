import os
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from clustering_tools import read_data, draw_clusters, evaluate_clustering
# from models.autoencoder import EnhancedAutoencoder
from utils.feature_extraction import MultiViewFeatureExtractor
from BorderPeel import BorderPeel
from config import Config
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt

def setup_parser():
    parser = ArgumentParser(description='Run M3W clustering on Pathbased dataset')
    parser.add_argument('--input', type=str, default='res/Pathbased.csv',
                      help='Input CSV file path (default: res/Pathbased.csv)')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory path (default: results)')
    parser.add_argument('--n-clusters', type=int, default=3,
                      help='Number of clusters (default: 3)')
    return parser

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory: {str(e)}")
            raise

def save_results(X, labels, evaluation_results, args):
    """保存所有结果"""
    try:
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(args.output, f"run_{timestamp}")
        ensure_dir(result_dir)
        
        # 1. 保存聚类结果
        results_file = os.path.join(result_dir, "clustering_results.csv")
        results = np.column_stack((X, labels))
        pd.DataFrame(results, columns=['x', 'y', 'cluster_label']).to_csv(
            results_file, index=False, float_format='%.6f'
        )
        
        # 2. 保存评估结果
        evaluation_file = os.path.join(result_dir, "evaluation_results.txt")
        with open(evaluation_file, 'w') as f:
            f.write("Clustering Evaluation Results\n")
            f.write("==========================\n")
            f.write(f"Number of clusters: {args.n_clusters}\n")
            f.write(f"Number of samples: {len(X)}\n\n")
            
            f.write("Evaluation Metrics:\n")
            f.write("-----------------\n")
            for metric, value in evaluation_results.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nCluster Statistics:\n")
            f.write("-----------------\n")
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                f.write(f"Cluster {label}: {count} samples\n")
        
        # 3. 保存可视化结果
        plot_file = os.path.join(result_dir, "clustering_plot.png")
        plt.figure(figsize=(10, 8))
        draw_clusters(X, labels, show_plt=False)
        plt.title('Pathbased Dataset Clustering Results')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return result_dir
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

def get_available_device():
    """检测并返回最适合的设备"""
    if torch.cuda.is_available():
        # 检查CUDA是否真的可用
        try:
            device = torch.device('cuda')
            # 测试CUDA是否真的可用
            torch.zeros(1).cuda()
            print("使用 CUDA 设备")
            return device
        except Exception as e:
            print(f"CUDA初始化失败: {e}")
            print("回退到CPU设备")
    else:
        print("未检测到CUDA设备，使用CPU")
    return torch.device('cpu')

def validate_config(config, data_shape):
    """验证配置参数是否合理"""
    validated_config = config.copy()
    
    # 确保输入维度匹配数据
    validated_config['input_dim'] = data_shape[1]
    
    # 验证其他参数
    if 'hidden_dim' not in validated_config:
        validated_config['hidden_dim'] = 32  # 默认值
    
    if 'latent_dim' not in validated_config:
        validated_config['latent_dim'] = 16  # 默认值
        
    # 移除任何不需要的参数
    valid_params = {'input_dim', 'hidden_dim', 'latent_dim', 'device'}
    return {k: v for k, v in validated_config.items() if k in valid_params}

def setup_torch_defaults(device):
    """设置PyTorch默认配置"""
    torch.set_default_dtype(torch.float32)
    if device.type == 'cuda':
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')

def main():
    args = setup_parser().parse_args()
    
    # 获取可用设备
    device = get_available_device()
    
    try:
        # 设置默认张量类型
        setup_torch_defaults(device)
        
        # 1. 加载数据
        print("\nLoading data...")
        data = pd.read_csv(args.input, header=None)
        X = data.iloc[:, :2].values  # 前两列是特征
        y = data.iloc[:, 2].values   # 最后一列是标签
        
        print(f"Loaded data shape: {X.shape}")
        print(f"Number of true clusters: {len(np.unique(y))}")
        
        # 2. 数据标准化
        print("\nStandardizing data...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 3. 验证并更新配置
        config = validate_config(Config.FEATURE_EXTRACTION_CONFIG, X_scaled.shape)
        config['device'] = device
        
        # 4. 特征提取
        print("\nExtracting features...")
        feature_extractor = MultiViewFeatureExtractor(**config)
        features = feature_extractor.extract_features(X_scaled)
        
        # 5. 生成伪标签
        print("\nGenerating pseudo labels...")
        pseudo_labels = feature_extractor.get_pseudo_labels(features, args.n_clusters)
        print(f"Generated {len(np.unique(pseudo_labels))} pseudo clusters")
        
        # 6. M3W聚类
        print("\nRunning M3W clustering...")
        m3w_config = Config.M3W_CONFIG.copy()
        m3w_config['n_clusters'] = args.n_clusters  # 使用命令行参数的聚类数
        m3w = BorderPeel(**m3w_config)
        final_labels = m3w.fit_predict(features)
        print(f"Final clustering produced {len(np.unique(final_labels))} clusters")
        
        # 7. 评估结果
        print("\nEvaluating clustering results...")
        evaluation_results = evaluate_clustering(features, y, final_labels)
        
        # 8. 保存所有结果
        print("\nSaving results...")
        result_dir = save_results(X, final_labels, evaluation_results, args)
        print(f"\nAll results saved in: {result_dir}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()