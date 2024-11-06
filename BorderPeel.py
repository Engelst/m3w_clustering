import border_tools as bt
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.self_supervised import SelfSupervisedM3W


class BorderPeel(BaseEstimator, ClusterMixin):
    """Perform DBSCAN clustering from vector array or distance matrix.
    TODO: Fill out doc
    BorderPeel - Border peel based clustering
    Read more in the :ref:`User Guide <BorderPeel>`.
    Parameters
    ----------
    TODO: Fill out parameters..
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    Attributes
    ----------
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.
    Notes
    -----
    References
    ----------
    """

    def __init__(self, k=9, n_clusters=3, C=1.2, T=4, 
                 alpha=0.7, beta=0.1, border_percentile=0.15,
                 mean_border_eps=0.2, stopping_percentile=0.02):
        """
        初始化BorderPeel聚类器
        
        参数:
            k (int): 近邻数量，默认9
            n_clusters (int): 聚类数量，默认3
            C (float): 链接距离扩展因子，默认1.2
            T (int): 最大迭代次数，默认4
            alpha (float): 核心点阈值，默认0.7
            beta (float): 边界控制参数，默认0.1
            border_percentile (float): 边界百分位数，默认0.15
            mean_border_eps (float): 平均边界eps，默认0.2
            stopping_percentile (float): 停止条件百分位数，默认0.02
        """
        self.k = k
        self.n_clusters = n_clusters
        self.C = C
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.border_percentile = border_percentile
        self.mean_border_eps = mean_border_eps
        self.stopping_percentile = stopping_percentile

        # out fields
        self.labels_ = None
        self.core_points = None
        self.core_points_indices = None
        self.non_merged_core_points = None
        self.data_sets_by_iterations = None
        self.associations = None
        self.link_thresholds = None
        self.border_values_per_iteration = None

    def fit(self, X, X_plot_projection=None):
        """
        训练BorderPeel模型
        
        参数:
            X: 输入特征
            X_plot_projection: 可选的2D投影数据用于可视化
        """
        # 确保输入是PyTorch张量
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        # 初始化自监督模型
        self.ssl_model = SelfSupervisedM3W(
            input_dim=X.shape[1],
            projection_dim=128,
            latent_dim=64
        )
        
        # 数据增强
        def augment(x):
            noise = torch.randn_like(x) * 0.1
            return x + noise
        
        # 自监督预训练
        optimizer = torch.optim.Adam(self.ssl_model.parameters())
        self.ssl_model.train()
        
        for epoch in range(10):
            # 生成两个增强视图
            X1 = augment(X)
            X2 = augment(X)
            
            # 前向传播
            _, p1 = self.ssl_model(X1)
            _, p2 = self.ssl_model(X2)
            
            # 计算损失
            loss = self.ssl_model.contrastive_loss(p1, p2)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # 提取特征
        self.ssl_model.eval()
        with torch.no_grad():
            features, _ = self.ssl_model(X)
            features = features.numpy()
        
        # 执行聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels_ = kmeans.fit_predict(features)
        
        return self

    def fit_predict(self, X, X_plot_projection=None):
        """Performs BorderPeel clustering clustering on X and returns cluster labels.
        Parameters
        ----------
        X : array of features (TODO: make it work with sparse arrays)
        X_projected : A projection of the data to 2D used for plotting the graph during the cluster process
        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """

        self.fit(X, X_plot_projection=X_plot_projection)
        return self.labels_

    def _supervised_training(self, features, pseudo_labels):
        """监督训练方法"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(self.config['epochs']):
            self.train()
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self(features)
            loss = criterion(outputs, pseudo_labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
