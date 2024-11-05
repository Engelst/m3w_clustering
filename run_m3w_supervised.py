import os
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from clustering_tools import load_from_file_or_data, draw_clusters


def setup_parser():
    parser = ArgumentParser(description='Run M3W clustering on unlabeled data')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--n-clusters', type=int, required=True, help='Number of clusters')
    parser.add_argument('--separator', default=',', help='CSV separator (default: ,)')
    parser.add_argument('--dim', type=int, default=2, help='Data dimensionality (default: 2)')
    return parser


def load_data(file_path, separator=',', dim=2):
    try:
        # Load the data as a single column
        data = pd.read_csv(file_path, header=None)

        # Convert to numpy array
        data_array = data.values.flatten()

        # Calculate the number of samples
        n_samples = len(data_array)

        # Reshape the data into a 2D array with dim columns
        if n_samples % dim != 0:
            raise ValueError(f"Data length ({n_samples}) is not divisible by dimensions ({dim})")

        n_points = n_samples // dim
        reshaped_data = data_array.reshape(n_points, dim)

        return reshaped_data

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("\nThe input file should be a CSV with numeric values.")
        print(f"Expected format: Single column of values that can be reshaped into {dim} dimensions")
        exit(1)


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Print startup information
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"\nLooking for file: {args.input}")
    print(f"\nUsing device: cpu")
    print("\nLoading data...")

    # Load the data
    data = load_data(args.input, args.separator, args.dim)

    print(f"\nData shape: {data.shape}")

    if data.shape[1] != args.dim:
        print(f"Error: Expected {args.dim} dimensions but got {data.shape[1]} dimensions in the data")
        print("Please check your input file or adjust the --dim parameter")
        exit(1)

    # Implement M3W clustering here
    # For now, we'll just use basic k-means as a placeholder
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)

    # Save results
    results = np.column_stack((data, clusters))
    np.savetxt(args.output, results, delimiter=',', fmt='%.6f')

    # Visualize if 2D
    if args.dim == 2:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'Clustering Results (k={args.n_clusters})')
        plt.savefig(args.output.replace('.txt', '_visualization.png'))
        plt.close()

    print(f"\nResults saved to: {args.output}")
    print(f"Visualization saved as: {args.output.replace('.txt', '_visualization.png')}")


if __name__ == '__main__':
    main()