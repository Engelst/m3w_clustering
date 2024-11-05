import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
        self.feature_queue = tf.Variable(
            tf.nn.l2_normalize(
                tf.random.normal(shape=(queue_size, projection_dim)),
                axis=1
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
        proj_1 = tf.nn.l2_normalize(proj_1, axis=1)
        proj_2 = tf.nn.l2_normalize(proj_2, axis=1)

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

        return tf.reduce_mean(loss)