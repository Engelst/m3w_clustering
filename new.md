根据搜索结果和代码分析，我可以为M3W算法设计一个基于对比学习的自监督方案。主要思路参考了[SimSiam](https://keras.io/examples/vision/simsiam/)和[NNCLR](https://keras.io/examples/vision/nnclr/)的设计。以下是具体的技术方案:

1. 首先添加一个新的自监督学习模块 `self_supervised.py`:

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

class SelfSupervisedM3W:
    def __init__(
        self,
        input_shape,
        projection_dim=128,
        latent_dim=64,
        temperature=0.1,
        queue_size=10000,
    ):
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.latent_dim = latent_dim
        
        # 编码器网络
        self.encoder = self._build_encoder(input_shape)
        # 投影头网络
        self.projection_head = self._build_projection_head()
        # 预测器网络
        self.predictor = self._build_predictor()
        
        # 特征队列用于nearest neighbor对比学习
        self.feature_queue = keras.Variable(
            keras.utils.normalize(
                keras.random.normal(shape=(queue_size, projection_dim)),
                axis=1,
                order=2
            ),
            trainable=False,
        )

    def _build_encoder(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(512, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x) 
        x = layers.Dense(128, activation="relu")(x)
        return keras.Model(inputs, x, name="encoder")

    def _build_projection_head(self):
        inputs = layers.Input(shape=(128,))
        x = layers.Dense(self.projection_dim, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(self.projection_dim)(x)
        return keras.Model(inputs, x, name="projection_head")

    def _build_predictor(self):
        inputs = layers.Input(shape=(self.projection_dim,))
        x = layers.Dense(self.latent_dim, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(self.projection_dim)(x)
        return keras.Model(inputs, x, name="predictor")

    def nearest_neighbour(self, projections):
        """查找最近邻样本用于对比学习"""
        support_similarities = tf.matmul(
            projections, tf.transpose(self.feature_queue)
        )
        nn_projections = tf.gather(
            self.feature_queue, 
            tf.argmax(support_similarities, axis=1),
            axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)

    def contrastive_loss(self, proj_1, proj_2):
        """计算对比损失"""
        # 归一化投影
        proj_1 = tf.math.l2_normalize(proj_1, axis=1)
        proj_2 = tf.math.l2_normalize(proj_2, axis=1)
        
        # 获取最近邻
        nn_proj_1 = self.nearest_neighbour(proj_1)
        nn_proj_2 = self.nearest_neighbour(proj_2)
        
        # 计算相似度
        similarities_1 = tf.matmul(nn_proj_1, tf.transpose(proj_2)) / self.temperature
        similarities_2 = tf.matmul(nn_proj_2, tf.transpose(proj_1)) / self.temperature
        
        # 计算对比损失
        batch_size = tf.shape(proj_1)[0]
        labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([labels, labels], axis=0),
            tf.concat([similarities_1, similarities_2], axis=0),
            from_logits=True
        )
        
        # 更新特征队列
        self.feature_queue.assign(
            tf.concat([proj_1, self.feature_queue[:-batch_size]], axis=0)
        )
        
        return loss
```

2. 修改 `BorderPeel.py` 中的 `fit()` 方法,加入自监督学习:

```python
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
    result = bt.dynamic_3w(
        X_features,
        border_func=lambda data: bt.rknn_with_distance_transform(
            data, self.k, bt.exp_local_scaling_transform
        ),
        threshold_func=None,
        max_iterations=self.max_iterations,
        mean_border_eps=self.mean_border_eps,
        plot_debug_output_dir=self.plot_debug_output_dir,
        k=self.k,
        precentile=self.border_precentile,
        dist_threshold=self.dist_threshold,
        link_dist_expansion_factor=self.link_dist_expansion_factor,
        verbose=self.verbose,
        vis_data=X_plot_projection,
        min_cluster_size=self.min_cluster_size,
        stopping_precentile=self.stopping_precentile,
        should_merge_core_points=self.merge_core_points,
        debug_marker_size=self.debug_marker_size,
        core_points_threshold=self.core_points_threshold,
        dvalue_threshold=self.dvalue_threshold
    )
    
    self.labels_, self.core_points, self.non_merged_core_points, \
    self.data_sets_by_iterations, self.associations, self.link_thresholds, \
    self.border_values_per_iteration, self.core_points_indices = result
    
    return self
```

这个方案的主要特点是:

1. 采用了NNCLR的对比学习框架,使用特征队列来存储历史特征,通过最近邻对比学习来学习特征表示。这种方法相比简单的对比学习更稳定,能学到更好的特征表示[参考](https://keras.io/examples/vision/nnclr/)。

2. 使用了编码器-投影头-预测器的网络架构,这是借鉴了SimSiam的设计[参考](https://keras.io/examples/vision/simsiam/)。其中:
- 编码器用于提取特征
- 投影头将特征映射到对比学习空间
- 预测器用于增强特征学习

3. 损失函数采用InfoNCE loss,这是目前对比学习中最常用的损失函数[参考](https://encord.com/blog/guide-to-contrastive-learning/)。

4. 在原始M3W聚类之前,先通过自监督学习预训练特征提取器,然后用预训练的编码器提取特征后再进行聚类。这样可以学到更有判别性的特征表示,有助于提高聚类效果。

5. 数据增强采用了常用的几何变换,包括旋转、平移和缩放,这些变换可以产生不同视角的样本用于对比学习。

使用该方案的优势:

1. 无需标注数据即可学习有效的特征表示
2. 通过对比学习可以学到更具判别性的特征,有助于后续的聚类
3. 结合了最新的自监督学习技术(NNCLR、SimSiam等)
4. 保持了M3W算法原有的优势,同时提升了特征学习能力