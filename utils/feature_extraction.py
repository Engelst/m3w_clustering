# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.manifold import SpectralEmbedding
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np



class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class MultiViewFeatureExtractor:
    def __init__(self, n_components=32, n_neighbors=10):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder = None

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

                # 前向传播
                encoded, decoded = self.autoencoder(inputs)

                # 计算重构损失
                loss = criterion(decoded, inputs)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

    def extract_features(self, data):
        """提取多视角特征"""
        # 标准化数据
        scaled_data = self.scaler.fit_transform(data)

        # 训练自编码器并获取特征
        self.train_autoencoder(scaled_data)

        # 使用自编码器提取特征
        data_tensor = torch.FloatTensor(scaled_data).to(self.device)
        self.autoencoder.eval()
        with torch.no_grad():
            encoded, _ = self.autoencoder(data_tensor)
            autoencoder_features = encoded.cpu().numpy()

        # 提取PCA特征
        pca = PCA(n_components=min(self.n_components, data.shape[1]))
        pca_features = pca.fit_transform(scaled_data)

        # 提取TSNE特征
        tsne = TSNE(n_components=min(self.n_components, data.shape[1]),
                    n_iter=250, random_state=42)
        tsne_features = tsne.fit_transform(scaled_data)

        # 调整特征维度
        features_list = [autoencoder_features, pca_features, tsne_features]
        max_dim = max(f.shape[1] for f in features_list)
        aligned_features = []

        for feat in features_list:
            if feat.shape[1] < max_dim:
                pad_width = ((0, 0), (0, max_dim - feat.shape[1]))
                feat = np.pad(feat, pad_width, mode='constant')
            aligned_features.append(feat)

        # 合并所有特征
        combined_features = np.hstack(aligned_features)
        return combined_features

    def get_pseudo_labels(self, features, n_clusters):
        """获取伪标签"""
        # 确保特征维度能被3整除
        feature_dim = features.shape[1]
        view_dim = feature_dim // 3
        remainder = feature_dim % 3

        if remainder > 0:
            pad_width = ((0, 0), (0, 3 - remainder))
            features = np.pad(features, pad_width, mode='constant')
            feature_dim = features.shape[1]

        # 将特征分成三个视图
        feature_views = np.split(features, 3, axis=1)

        # 对每个视图进行聚类
        kmeans_models = []
        view_labels = []

        for view_features in feature_views:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(view_features)
            kmeans_models.append(kmeans)
            view_labels.append(labels)

        # 使用多数投票确定最终的伪标签
        view_labels = np.array(view_labels)
        pseudo_labels = np.zeros(len(features), dtype=int)

        for i in range(len(features)):
            sample_labels = view_labels[:, i]
            unique_labels, counts = np.unique(sample_labels, return_counts=True)
            pseudo_labels[i] = unique_labels[np.argmax(counts)]

        return pseudo_labels