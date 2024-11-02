class Config:
    # 自编码器配置
    AUTOENCODER_CONFIG = {
        'hidden_dims': [128, 64, 32],  # 编码器各层维度
        'latent_dim': 16,              # 潜在空间维度
        'dropout_rate': 0.1,           # dropout比率
        'learning_rate': 0.001,        # 学习率
        'batch_size': 64,              # 批次大小
        'epochs': 100,                 # 训练轮数
        'activation': 'relu'           # 激活函数
    }
    
    # K-means配置
    KMEANS_CONFIG = {
        'n_init': 20,                  # 初始化次数
        'max_iter': 300,               # 最大迭代次数
        'random_state': 42             # 随机种子
    }
    
    # M3W配置
    M3W_CONFIG = {
        'k': 8,                        # 近邻数
        'C': 1.6,                      # 链接距离扩展因子
        'T': 3,                        # 最大迭代次数
        'alpha': 0.6,                  # 核心点阈值
        'beta': 0,                     # d值阈值
        'border_percentile': 0.1,      # 边界百分位数
        'mean_border_eps': 0.15,       # 平均边界eps
        'stopping_percentile': 0.01    # 停止百分位数
    }