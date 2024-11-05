import border_tools as bt
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin


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

    def __init__(self
                 , method="exp_local_scaling"
                 , max_iterations=150
                 , mean_border_eps=-1
                 , k=20
                 , plot_debug_output_dir=None
                 , min_cluster_size=3
                 , dist_threshold=3
                 , convergence_constant=0
                 , link_dist_expansion_factor=3
                 , verbose=True
                 , border_precentile=0.1
                 , stopping_precentile=0
                 , merge_core_points=True
                 , debug_marker_size=70
                 , core_points_threshold=1
                 , dvalue_threshold=1
                 ):
        self.method = method
        self.k = k
        self.max_iterations = max_iterations
        self.plot_debug_output_dir = plot_debug_output_dir
        self.min_cluster_size = min_cluster_size
        self.dist_threshold = dist_threshold
        self.convergence_constant = convergence_constant
        self.link_dist_expansion_factor = link_dist_expansion_factor
        self.verbose = verbose
        self.border_precentile = border_precentile
        self.stopping_precentile = stopping_precentile
        self.merge_core_points = merge_core_points
        self.mean_border_eps = mean_border_eps
        self.debug_marker_size = debug_marker_size
        self.core_points_threshold = core_points_threshold
        self.dvalue_threshold = dvalue_threshold

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
        # 初始化自监督学习模块
        self.ssl_model = SelfSupervisedM3W(
            input_shape=X.shape[1:],
            projection_dim=128,
            latent_dim=64
        )

        # 数据增强
        augmenter = keras.Sequential([
            layers.RandomRotation(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ])

        # 自监督预训练
        optimizer = keras.optimizers.Adam()
        for epoch in range(10):
            # 生成两个增强视图
            X1 = augmenter(X)
            X2 = augmenter(X)

            with tf.GradientTape() as tape:
                # 前向传播
                z1 = self.ssl_model.encoder(X1)
                z2 = self.ssl_model.encoder(X2)
                p1 = self.ssl_model.projection_head(z1)
                p2 = self.ssl_model.projection_head(z2)

                # 计算对比损失
                loss = self.ssl_model.contrastive_loss(p1, p2)

            # 反向传播
            gradients = tape.gradient(
                loss,
                self.ssl_model.encoder.trainable_variables +
                self.ssl_model.projection_head.trainable_variables
            )
            optimizer.apply_gradients(zip(
                gradients,
                self.ssl_model.encoder.trainable_variables +
                self.ssl_model.projection_head.trainable_variables
            ))

            if self.verbose:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # 使用预训练编码器提取特征
        X_features = self.ssl_model.encoder.predict(X)

        # 原始的M3W聚类
        result = self._cluster_features(X_features)
        return result

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
