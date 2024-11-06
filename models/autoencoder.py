import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DataAugmentation:
    """数据增强类"""
    @staticmethod
    def random_rotation(x, max_degrees=10):
        theta = np.random.uniform(-max_degrees, max_degrees) * np.pi / 180
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=torch.float32)
        return torch.matmul(x, rotation_matrix)

    @staticmethod
    def random_noise(x, std=0.01):
        return x + torch.randn_like(x) * std

    @staticmethod
    def random_scaling(x, min_scale=0.8, max_scale=1.2):
        scale = torch.rand(1) * (max_scale - min_scale) + min_scale
        return x * scale

    @staticmethod
    def random_translation(x, max_translation=0.1):
        tx = torch.rand(1) * 2 * max_translation - max_translation
        ty = torch.rand(1) * 2 * max_translation - max_translation
        return x + torch.tensor([tx, ty])

    def apply_augmentations(self, x):
        augmentations = [
            self.random_rotation,
            self.random_noise,
            self.random_scaling,
            self.random_translation
        ]
        aug_x = x.clone()
        for aug in augmentations:
            if torch.rand(1) > 0.5:  # 随机应用增强
                aug_x = aug(aug_x)
        return aug_x

class EnhancedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device=None):
        super().__init__()
        # 设置设备
        self.device = device if device is not None else \
                     torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化网络组件
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 对比学习头
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 数据增强器
        self.augmenter = DataAugmenter()
        
        # 移动模型到指定设备
        try:
            self.to(self.device)
        except Exception as e:
            print(f"警告：模型移动到{self.device}失败: {e}")
            print("回退到CPU")
            self.device = torch.device('cpu')
            self.to(self.device)

    def forward(self, x):
        # 确保输入在正确的设备上
        x = self._ensure_tensor_device(x)
        
        # 生成两个增强视图
        x1 = self.augmenter.apply_augmentations(x)
        x2 = self.augmenter.apply_augmentations(x)
        
        # 编码
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        # 投影
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)
        
        # 解码（使用原始输入的编码）
        z = self.encoder(x)
        x_recon = self.decoder(z)
        
        return z1, z2, p1, p2, x_recon
    
    def _ensure_tensor_device(self, x):
        """确保输入张量在正确的设备上"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.to(self.device)
    
    def encode(self, x):
        """编码数据"""
        x = self._ensure_tensor_device(x)
        return self.encoder(x)

class DataAugmenter:
    def __init__(self, translation_range=0.1, noise_std=0.05):
        self.translation_range = translation_range
        self.noise_std = noise_std
    
    def apply_augmentations(self, x):
        """应用数据增强"""
        x = self.random_translation(x)
        x = self.add_gaussian_noise(x)
        return x
    
    def random_translation(self, x):
        """随机平移"""
        if isinstance(x, torch.Tensor):
            translation = torch.randn_like(x) * self.translation_range
            return x + translation
        return x
    
    def add_gaussian_noise(self, x):
        """添加高斯噪声"""
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        """计算对比损失"""
        batch_size = z1.shape[0]
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z1_norm, z2_norm.T) / self.temperature
        
        # 正样本对的标签
        labels = torch.arange(batch_size, device=z1.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

def reconstruction_loss(x, x_recon):
    """计算重构损失"""
    return F.mse_loss(x_recon, x)

def total_loss(x, x_recon, z1, z2, p1, p2, contrastive_weight=1.0):
    """计算总损失"""
    recon_loss = reconstruction_loss(x, x_recon)
    contrast_loss = ContrastiveLoss()(p1, p2)
    return recon_loss + contrastive_weight * contrast_loss

class AutoencoderTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, data):
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        reconstruction_criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for batch in dataloader:
                inputs = batch[0]
                
                # 前向传播
                z1, z2, p1, p2, x_recon = self.model(inputs)
                
                # 计算损失
                reconstruction_loss = reconstruction_criterion(x_recon, inputs)
                contrastive_loss = self.model.contrastive_loss(p1, p2)
                
                # 总损失
                loss = reconstruction_loss + self.config['contrastive_weight'] * contrastive_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                      f'Average Loss: {avg_loss:.4f}')
    
    def get_features(self, data):
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            z, _, _, _, _ = self.model(data_tensor)
            return z.cpu().numpy()