import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

class SupervisedClusterNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], n_clusters=None):
        super(SupervisedClusterNet, self).__init__()
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        
        # 特征提取层
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # 聚类头
        self.cluster_head = nn.Linear(hidden_dims[-1], n_clusters)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.cluster_head(features)
        return features, logits

class SupervisedClusteringTrainer:
    def __init__(self, model, device='cuda', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scaler = StandardScaler()
        
    def train(self, data, pseudo_labels, epochs=100, batch_size=64):
        # 数据预处理
        scaled_data = self.scaler.fit_transform(data)
        data_tensor = torch.FloatTensor(scaled_data).to(self.device)
        labels_tensor = torch.LongTensor(pseudo_labels).to(self.device)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_data, batch_labels in dataloader:
                # 前向传播
                features, logits = self.model(batch_data)
                
                # 计算分类损失
                cls_loss = F.cross_entropy(logits, batch_labels)
                
                # 计算特征聚类损失
                center_loss = self._compute_center_loss(features, batch_labels)
                
                # 总损失
                loss = cls_loss + 0.1 * center_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.model.state_dict()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        # 加载最佳模型
        self.model.load_state_dict(best_state)
        
    def _compute_center_loss(self, features, labels):
        """计算特征中心损失"""
        centers = []
        for i in range(self.model.n_clusters):
            mask = (labels == i)
            if mask.sum() > 0:
                center = features[mask].mean(0)
                centers.append(center)
        
        if not centers:
            return torch.tensor(0.0).to(self.device)
            
        centers = torch.stack(centers)
        
        # 计算类内距离
        intra_dist = 0
        for i in range(self.model.n_clusters):
            mask = (labels == i)
            if mask.sum() > 0:
                cluster_points = features[mask]
                center = centers[i]
                intra_dist += torch.mean(torch.norm(cluster_points - center, dim=1))
        
        # 计算类间距离
        inter_dist = 0
        n_valid_pairs = 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                inter_dist += torch.norm(centers[i] - centers[j])
                n_valid_pairs += 1
                
        if n_valid_pairs == 0:
            return intra_dist
            
        return intra_dist - 0.1 * (inter_dist / n_valid_pairs)
    
    def predict(self, data):
        """预测聚类标签"""
        self.model.eval()
        scaled_data = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(scaled_data).to(self.device)
        
        with torch.no_grad():
            features, logits = self.model(data_tensor)
            predictions = torch.argmax(logits, dim=1)
            
        return predictions.cpu().numpy(), features.cpu().numpy()