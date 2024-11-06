__all__ = ['AutoEncoder', 'ClusteringNetwork', 'SelfSupervisedM3W']

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(AutoEncoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        # 解码
        decoded = self.decoder(encoded)
        return encoded, decoded


class ClusteringNetwork(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super(ClusteringNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_clusters)
        )

    def forward(self, x):
        return F.softmax(self.network(x), dim=1)


class SelfSupervisedM3W(nn.Module):
    def __init__(self, input_dim, projection_dim=128, latent_dim=64):
        super(SelfSupervisedM3W, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim)
        )
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections
        
    def contrastive_loss(self, p1, p2, temperature=0.1):
        """计算对比损失"""
        # 归一化投影
        z1 = F.normalize(p1, dim=1)
        z2 = F.normalize(p2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z1, z2.T) / temperature
        
        # 正样本对的标签
        labels = torch.arange(z1.size(0), device=z1.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss