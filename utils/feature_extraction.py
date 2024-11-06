import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import numpy as np
from scipy.stats import mode
from models.autoencoder import EnhancedAutoencoder, AutoencoderTrainer
from config import Config

class MultiViewFeatureExtractor:
    def __init__(self, input_dim, hidden_dim, latent_dim, device=None):
        """初始化多视图特征提取器
        
        参数:
            input_dim (int): 输入数据维度
            hidden_dim (int): 自动编码器隐藏层维度
            latent_dim (int): 潜在空间维度
            device (torch.device): 计算设备
        """
        self.device = device if device is not None else \
                     torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化自动编码器
        try:
            self.autoencoder = EnhancedAutoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                device=self.device
            )
        except Exception as e:
            print(f"自动编码器初始化失败: {e}")
            print("尝试使用CPU模式重新初始化")
            self.device = torch.device('cpu')
            self.autoencoder = EnhancedAutoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                device=self.device
            )
        self.autoencoder_trainer = None
        
    def train_autoencoder(self, data):
        """训练自编码器"""
        print("Training autoencoder...")
        input_dim = data.shape[1]
        
        # 创建自编码器模型
        self.autoencoder = EnhancedAutoencoder(
            input_dim=input_dim,
            config=Config.AUTOENCODER_CONFIG
        )
        
        # 创建训练器
        self.autoencoder_trainer = AutoencoderTrainer(
            self.autoencoder,
            Config.AUTOENCODER_CONFIG
        )
        
        # 训练自编码器
        self.autoencoder_trainer.train(data)
        print("Autoencoder training completed.")
    
    def extract_features(self, data):
        """提取特征"""
        self.autoencoder.eval()  # 设置为评估模式
        with torch.no_grad():
            features = self.autoencoder.encode(data)
        return features.cpu().numpy()
    
    def get_pseudo_labels(self, features, n_clusters):
        """生成伪标签"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(features)
    
    def _batch_process(self, data, batch_size=32):
        """分批处理大数据集"""
        n_samples = len(data)
        features_list = []
        
        for i in range(0, n_samples, batch_size):
            batch = data[i:min(i + batch_size, n_samples)]
            batch_features = self._extract_features_impl(batch)
            features_list.append(batch_features)
            
            # 主动清理GPU内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        return torch.cat(features_list, dim=0)