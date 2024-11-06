import argparse
import numpy as np
import torch
import clustering_tools as ct
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\nLoading data...")
    # Using the corrected read_data function
    data, labels = ct.read_data(args.input, separator=',', has_labels=not args.no_labels)
    print(f"\nLoaded data shape: {data.shape}")

    if labels is not None:
        n_true_clusters = len(np.unique(labels))
        print(f"\nNumber of true clusters: {n_true_clusters}")

    # Standardize data
    print("\nStandardizing data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Extract features
    print("\nExtracting features...")
    features = scaled_data

    # Generate pseudo labels
    print("\nGenerating pseudo labels...")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        n_init=20,
        max_iter=300,
        random_state=42
    )
    pseudo_labels = kmeans.fit_predict(features)
    print(f"\nGenerated {args.n_clusters} pseudo clusters")

    # Save results
    if args.output:
        print(f"\nSaving results to {args.output}")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            # Write evaluation metrics
            if labels is not None:
                nmi = normalized_mutual_info_score(labels, pseudo_labels)
                ari = adjusted_rand_score(labels, pseudo_labels)
                f.write(f"NMI: {nmi:.4f}\n")
                f.write(f"ARI: {ari:.4f}\n")
            # Write cluster labels
            f.write("\nCluster Labels:\n")
            for label in pseudo_labels:
                f.write(f"{label}\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Run M3W clustering with supervision')

    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input data file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for results')
    parser.add_argument('--n-clusters', type=int, required=True,
                        help='Number of clusters')

    # Optional arguments
    parser.add_argument('--no-labels', action='store_true',
                        help='Input data does not contain labels')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


if __name__ == "__main__":
    main()