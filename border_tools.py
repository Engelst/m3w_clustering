import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from UnionFind import UnionFind
import os


def estimate_lambda(data, k=8):
    """
    Estimate lambda parameter for clustering based on k-nearest neighbors distances

    Parameters:
    -----------
    data : array-like
        Input data points
    k : int, optional (default=8)
        Number of nearest neighbors to consider

    Returns:
    --------
    float
        Estimated lambda value
    """
    # Compute pairwise distances
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Get the k-th nearest neighbor distances (exclude self-distance)
    kth_distances = distances[:, 1:k + 1]

    # Calculate mean distance
    mean_kth_distance = np.mean(kth_distances)

    # Estimate lambda as inverse of mean distance
    lambda_estimate = 1.0 / (mean_kth_distance + 1e-10)  # Add small constant to avoid division by zero

    return lambda_estimate


def exp_local_scaling_transform(distances, indices, k):
    """
    Exponential local scaling transform for distance values
    """
    sigma = distances[:, k - 1]
    sigma = np.maximum(sigma, 1e-10)  # Avoid division by zero
    transformed = np.exp(-distances ** 2 / (sigma.reshape(-1, 1) * sigma))
    return transformed


def rknn_with_distance_transform(data, k, transform_func):
    """
    Calculate reverse k-nearest neighbors with distance transform
    """
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Apply distance transform
    transformed = transform_func(distances, indices, k)

    # Calculate border values
    border_values = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(k):
            neighbor_idx = indices[i][j]
            # Add transformed distance if current point is not in neighbor's k-NN
            if i not in indices[neighbor_idx]:
                border_values[i] += transformed[i][j]

    return border_values


def dynamic_3w(data, border_func, threshold_func=None, max_iterations=150,
               mean_border_eps=-1, plot_debug_output_dir=None, k=20,
               precentile=0, dist_threshold=3, link_dist_expansion_factor=3,
               verbose=True, vis_data=None, min_cluster_size=3,
               stopping_precentile=0, should_merge_core_points=True,
               debug_marker_size=70, core_points_threshold=1,
               dvalue_threshold=1):
    """
    Dynamic M3W clustering algorithm
    """
    data_length = len(data)
    cluster_uf = UnionFind(data_length)

    # Initialize data structures
    data_sets_by_iterations = []
    border_values_per_iteration = []
    current_data = data.copy()
    current_indices = np.arange(data_length)

    # Main iteration loop
    iteration = 0
    while len(current_data) > 0 and iteration < max_iterations:
        if verbose:
            print(f"Iteration {iteration}, remaining points: {len(current_data)}")

        # Calculate border values
        border_values = border_func(current_data)
        border_values_per_iteration.append(border_values)

        # Determine threshold
        if threshold_func is not None:
            threshold = threshold_func(border_values)
        else:
            threshold = np.percentile(border_values, precentile)

        # Find core points
        core_points_mask = border_values <= threshold
        core_points = current_data[core_points_mask]
        core_indices = current_indices[core_points_mask]

        if len(core_points) == 0:
            break

        # Update data sets
        data_sets_by_iterations.append(core_indices)  # Store indices instead of points
        current_data = current_data[~core_points_mask]
        current_indices = current_indices[~core_points_mask]

        # Debug plotting
        if plot_debug_output_dir and vis_data is not None:
            plot_iteration(iteration, vis_data, core_points,
                           plot_debug_output_dir, debug_marker_size)

        iteration += 1

    # Process final results
    labels = process_clusters(data_sets_by_iterations, cluster_uf,
                              min_cluster_size, data_length)

    return labels, data_sets_by_iterations, border_values_per_iteration


def plot_iteration(iteration, data, core_points, output_dir, marker_size):
    """
    Plot debug information for current iteration
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.5, s=marker_size)
    plt.scatter(core_points[:, 0], core_points[:, 1], c='red', s=marker_size)
    plt.title(f'Iteration {iteration}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f'iteration_{iteration}.png'))
    plt.close()


def process_clusters(data_sets, union_find, min_size, data_length):
    """
    Process final cluster assignments
    """
    labels = np.full(data_length, -1)
    current_label = 0

    # Process each data set
    for data_set in data_sets:
        if len(data_set) >= min_size:
            for point in data_set:
                if labels[point] == -1:
                    labels[point] = current_label
            current_label += 1

    return labels


def border_peel_rknn_exp_transform_local(data, k, threshold, iterations,
                                         debug_output_dir=None, dist_threshold=3,
                                         link_dist_expansion_factor=3,
                                         precentile=0, verbose=True):
    """
    Convenience function for border peeling with exponential transform
    """
    border_func = lambda rknn_with_distance_transform(
        data, k, exp_local_scaling_transform)
    threshold_func = lambda value: value > threshold

    return dynamic_3w(data, border_func, threshold_func,
                      plot_debug_output_dir=debug_output_dir, k=k,
                      precentile=precentile, dist_threshold=dist_threshold,
                      link_dist_expansion_factor=link_dist_expansion_factor,
                      verbose=verbose)