import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(FeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class MultiViewFeatureExtractor:
    def __init__(self, n_components=32, n_neighbors=10):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.pca = PCA(n_components=n_components)
        self.spectral = SpectralEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors
        )
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_autoencoder(self, data, epochs=100, batch_size=64):
        """训练自编码器获取特征"""
        input_dim = data.shape[1]
        self.autoencoder = FeatureExtractor(input_dim, self.n_components).to(self.device)
        
        # 准备数据
        data_tensor = torch.FloatTensor(self.scaler.fit_transform(data)).to(self.device)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练
        optimizer = torch.optim.Adam(self.autoencoder.parameters())
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                inputs = batch[0]
                encoded = self.autoencoder(inputs)
                
                # 计算重构损失
                decoded = torch.nn.functional.linear(encoded, 
                    self.autoencoder.encoder[0].weight.t())
                loss = criterion(decoded, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    def extract_features(self, data):
        """多视角特征提取"""
        # 标准化数据
        scaled_data = self.scaler.transform(data)
        
        # PCA特征
        pca_features = self.pca.fit_transform(scaled_data)
        
        # 谱嵌入特征
        spectral_features = self.spectral.fit_transform(scaled_data)
        
        # 自编码器特征
        self.autoencoder.eval()
        with torch.no_grad():
            ae_features = self.autoencoder(
                torch.FloatTensor(scaled_data).to(self.device)
            ).cpu().numpy()
            
        # 组合特征
        combined_features = np.concatenate([
            pca_features,
            spectral_features,
            ae_features
        ], axis=1)
        
        return combined_features
        
    def get_pseudo_labels(self, features, n_clusters):
        """生成高质量伪标签"""
        # 使用集成策略
        labels_list = []
        
        # 对每个特征视图进行聚类
        feature_views = np.split(features, 3, axis=1)  # 分割成三个视图
        
        for view_features in feature_views:
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=20,
                max_iter=300,
                random_state=42
            )
            labels_list.append(kmeans.fit_predict(view_features))
        
        # 通过投票选择最终标签
        final_labels = np.zeros(len(features), dtype=int)
        for i in range(len(features)):
            labels = [labels[i] for labels in labels_list]
            final_labels[i] = max(set(labels), key=labels.count)
            
        return final_labels
