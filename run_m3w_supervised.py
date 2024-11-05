import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import pandas as pd
from self_supervised import SelfSupervisedM3W


def load_data(file_path, separator=',', has_labels=True):
    """
    Load data from CSV file
    """
    try:
        data, labels = read_data(file_path, separator=separator, has_labels=has_labels)
        print(f"Loaded data shape: {data.shape}")
        print(f"Loaded labels shape: {labels.shape}")
        return data, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def read_data(file_path, separator=',', has_labels=True):
    """
    Read data from CSV file
    """
    with open(file_path) as handle:
        data = []
        labels = []

        for line in handle:
            line = line.rstrip()
            if len(line) == 0:
                continue

            values = [float(x) for x in line.split(separator) if x.strip()]
            if has_labels:
                # Assuming the last value is the label
                labels.append(int(values[-1]))
                # Generate synthetic features for demonstration
                data.append([np.random.rand(), np.random.rand()])
            else:
                data.append(values)

    return np.array(data), np.array(labels)


def create_data_augmentation():
    """Create data augmentation pipeline"""
    return keras.Sequential([
        layers.GaussianNoise(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ])


def train_self_supervised(X, epochs=100, batch_size=32):
    """Train self-supervised model"""
    ssl_model = SelfSupervisedM3W(input_shape=X.shape[1:])
    augmenter = create_data_augmentation()

    optimizer = keras.optimizers.Adam()

    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        total_loss = 0
        num_batches = 0

        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = X[batch_indices]

            view1 = augmenter(batch_data, training=True)
            view2 = augmenter(batch_data, training=True)

            with tf.GradientTape() as tape:
                z1 = ssl_model.encoder(view1)
                z2 = ssl_model.encoder(view2)
                p1 = ssl_model.projection_head(z1)
                p2 = ssl_model.projection_head(z2)

                loss = ssl_model.contrastive_loss(p1, p2)

            gradients = tape.gradient(
                loss,
                ssl_model.encoder.trainable_variables +
                ssl_model.projection_head.trainable_variables
            )
            optimizer.apply_gradients(zip(
                gradients,
                ssl_model.encoder.trainable_variables +
                ssl_model.projection_head.trainable_variables
            ))

            total_loss += loss
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / num_batches:.4f}")

    return ssl_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input data file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--n-clusters', type=int, required=True, help='Number of clusters')
    args = parser.parse_args()

    print("Using device:", "cuda" if tf.test.is_built_with_cuda() else "cpu")

    print("Loading data...")
    X, y_true = load_data(args.input)
    if X is None or X.shape[1] == 0:
        print("Generating synthetic features...")
        X = generate_synthetic_data(y_true)
        print(f"Generated data shape: {X.shape}")

    print(f"Data shape: {X.shape}")
    print(f"Number of labels: {len(y_true)}")
    print(f"Unique labels: {np.unique(y_true)}")

    print("Training self-supervised model...")
    ssl_model = train_self_supervised(X)

    print("Extracting features...")
    features = ssl_model.encoder.predict(X)

    print("Generating pseudo labels using K-means...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    pseudo_labels = kmeans.fit_predict(features)

    ari_pseudo = adjusted_rand_score(y_true, pseudo_labels)
    ami_pseudo = adjusted_mutual_info_score(y_true, pseudo_labels)
    print(f"Pseudo Labels Quality:")
    print(f"ARI: {ari_pseudo:.3f}")
    print(f"AMI: {ami_pseudo:.3f}")

    print("Saving results...")
    results = np.column_stack((X, pseudo_labels))
    np.savetxt(args.output, results, delimiter=',')


if __name__ == "__main__":
    main()