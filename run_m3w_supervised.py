import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
import os
from BorderPeel import BorderPeel
from clustering_tools import read_data, load_from_file_or_data  # 添加必要的导入


def parse_args():
    parser = argparse.ArgumentParser(description='Run M3W clustering with self-supervised learning')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--n-clusters', type=int, required=True, help='Number of clusters')
    return parser.parse_args()


def create_augmentation_model():
    """Create a simple data augmentation model for 2D data"""

    def augment(x, training=True):
        if not training:
            return x

        # Add random noise
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01)
        x = x + noise

        # Random scaling
        scale = tf.random.uniform([], 0.95, 1.05)
        x = x * scale

        # Random rotation (for 2D point cloud data)
        angle = tf.random.uniform([], -0.1, 0.1)
        cos_angle = tf.cos(angle)
        sin_angle = tf.sin(angle)
        rotation_matrix = tf.stack([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        x = tf.matmul(x, rotation_matrix)

        return x

    return augment


class SelfSupervisedModel(tf.keras.Model):
    def __init__(self, input_dim, projection_dim=64):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
        ])

        self.projector = tf.keras.Sequential([
            tf.keras.layers.Dense(projection_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(projection_dim)
        ])

    def call(self, inputs):
        features = self.encoder(inputs)
        projections = self.projector(features)
        return features, projections


def train_self_supervised(X):
    """Train self-supervised model"""
    print("\nTraining self-supervised model...")

    # Create model
    input_dim = X.shape[1]
    model = SelfSupervisedModel(input_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Create data augmenter
    augmenter = create_augmentation_model()

    # Prepare dataset
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.shuffle(1000).batch(32)

    # Training loop
    for epoch in range(10):
        total_loss = 0
        num_batches = 0

        for batch_data in dataset:
            with tf.GradientTape() as tape:
                # Create two augmented views
                view1 = augmenter(batch_data)
                view2 = augmenter(batch_data)

                # Get features and projections
                _, proj1 = model(view1)
                _, proj2 = model(view2)

                # Normalize projections
                proj1 = tf.math.l2_normalize(proj1, axis=1)
                proj2 = tf.math.l2_normalize(proj2, axis=1)

                # Compute contrastive loss
                temperature = 0.1
                logits = tf.matmul(proj1, proj2, transpose_b=True) / temperature
                labels = tf.range(tf.shape(batch_data)[0])
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
                loss = tf.reduce_mean(loss)

            # Update model
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    return model

def main():
    args = parse_args()
    
    # Print current working directory and check file existence
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Looking for file: {args.input}")
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        print("Please check if the file exists and the path is correct")
        return

    # Check device
    device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    print(f"\nUsing device: {device}")

    # Load data using clustering_tools functions
    print("\nLoading data...")
    try:
        X, y = read_data(args.input)
        
        if y is None:
            print("\nNo labels column found in the input file.")
            print("The input file should have the following format:")
            print("feature1,feature2,...,featureN,label")
            print("Please check your input file format.")
            return

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(f"\nProcessed data:")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of unique labels: {len(np.unique(y))}")
        
        # Continue with the rest of your code...
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()