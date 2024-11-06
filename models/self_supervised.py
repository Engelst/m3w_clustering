import torch
import torch.nn as nn
import torch.nn.functional as F

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
        z1 = F.normalize(p1, dim=1)
        z2 = F.normalize(p2, dim=1)
        similarity_matrix = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
