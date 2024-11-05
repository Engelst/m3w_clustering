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
        # First try loading as a regular CSV with headers
        data = pd.read_csv(file_path, sep=separator)
        return np.matrix(data.values)
    except:
        try:
            # If that fails, try loading without headers
            data = pd.read_csv(file_path, sep=separator, header=None)
            return np.matrix(data.values)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("\nThe input file should be a CSV with numeric features.")
            print(f"Expected format: {dim} columns of numeric values")
            exit(1)


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Print startup information
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"\nLooking for file: {args.input}")
    print(f"\nUsing device: cpu")  # Modify if GPU support is added
    print("\nLoading data...")

    # Load the data
    data = load_data(args.input, args.separator, args.dim)

    if data.shape[1] != args.dim:
        print(f"Error: Expected {args.dim} dimensions but got {data.shape[1]} dimensions in the data")
        print("Please check your input file or adjust the --dim parameter")
        exit(1)

    # Create dummy labels (all zeros) for visualization purposes
    dummy_labels = np.zeros(data.shape[0])

    # Here you would add your M3W clustering implementation
    # For now, we'll just visualize the raw data
    if data.shape[1] == 2:  # Only visualize if 2D
        import matplotlib.pyplot as plt
        ax = draw_clusters(data, dummy_labels, show_plt=True, show_title=True)
        plt.savefig(args.output + '_initial_data.png')
        plt.close()

    # Save the loaded data to confirm it's being read correctly
    np.savetxt(args.output, data, delimiter=',', fmt='%.6f')
    print(f"\nData shape: {data.shape}")
    print(f"Data saved to: {args.output}")


if __name__ == '__main__':
    main()