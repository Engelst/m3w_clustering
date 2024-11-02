import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class EnhancedAutoencoder(nn.Module):
    def __init__(self, input_dim, config):
        super(EnhancedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.config = config
        
        # 构建编码器层
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in config['hidden_dims']:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                getattr(nn, config['activation'])(),
                nn.Dropout(config['dropout_rate'])
            ])
            current_dim = hidden_dim
            
        # 潜在空间映射
        encoder_layers.append(nn.Linear(current_dim, config['latent_dim']))
        
        # 构建解码器层
        decoder_layers = []
        current_dim = config['latent_dim']
        for hidden_dim in reversed(config['hidden_dims']):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                getattr(nn, config['activation'])(),
                nn.Dropout(config['dropout_rate'])
            ])
            current_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
        
    def encode(self, x):
        return self.encoder(x)

class AutoencoderTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, data):
        # 准备数据
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        criterion = nn.MSELoss()
        
        # 训练循环
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for batch in dataloader:
                inputs = batch[0]
                
                # 前向传播
                encoded, decoded = self.model(inputs)
                
                # 计算损失
                reconstruction_loss = criterion(decoded, inputs)
                # 可以添加其他正则化损失
                loss = reconstruction_loss
                
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
            encoded = self.model.encode(data_tensor)
            return encoded.cpu().numpy()