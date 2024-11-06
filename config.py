class Config:
    # 自编码器配置 - 针对Pathbased数据特点优化
    AUTOENCODER_CONFIG = {
        'hidden_dims': [32, 16],     # 保持简单的网络结构
        'latent_dim': 8,             # 降维但保留足够信息
        'projection_dim': 16,         
        'dropout_rate': 0.1,         
        'learning_rate': 0.001,      
        'batch_size': 32,            # 适中的批次大小
        'epochs': 50,                
        'activation': 'ReLU',        
        'contrastive_weight': 0.5    
    }
    
    # 特征提取配置 - 保持2D特征
    FEATURE_EXTRACTION_CONFIG = {
        'input_dim': 2,  # 输入维度
        'hidden_dim': 32,  # 隐藏层维度
        'latent_dim': 16,  # 潜在空间维度
        'device': None,  # 设备将在运行时设置
    }
    
    # K-means配置
    KMEANS_CONFIG = {
        'n_clusters': 3,            # Pathbased有3个簇
        'n_init': 20,              
        'max_iter': 300,           
        'random_state': 42         
    }
    
    # M3W配置 - 修正后的参数
    M3W_CONFIG = {
        'k': 9,                    # 近邻数量
        'n_clusters': 3,           # 聚类数量
        'C': 1.2,                  # 链接距离扩展因子
        'T': 4,                    # 最大迭代次数
        'alpha': 0.7,              # 核心点阈值
        'beta': 0.1,               # 边界控制参数
        'border_percentile': 0.15,  # 边界百分位数
        'mean_border_eps': 0.2,    # 平均边界eps
        'stopping_percentile': 0.02 # 停止条件百分位数
    }
    
    # 监督训练配置
    SUPERVISED_CONFIG = {
        'epochs': 30,              # 减少轮数
        'learning_rate': 0.001,    # 保持不变
        'batch_size': 32,          # 减小批次大小
        'weight_decay': 1e-5       # 保持不变
    }
    
    @staticmethod
    def validate_device_config(device):
        """验证设备配置是否合理"""
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("警告：配置指定CUDA但系统不支持，将使用CPU")
            return torch.device('cpu')
        return device